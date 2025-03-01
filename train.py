# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
from typing import Any

from absl import logging
from flax import jax_utils
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax
import os

import input_pipeline
from input_pipeline import prepare_batch_data
import models.models_ddpm as models_ddpm
from models.models_ddpm import generate, edm_ema_scales_schedules, generate_from_noisy

from utils.info_util import print_params
from utils.vis_util import make_grid_visualization, visualize_cifar_batch, float_to_uint8
from utils.ckpt_util import restore_checkpoint, restore_pretrained, save_checkpoint
import utils.fid_util as fid_util
import utils.sample_util as sample_util

from configs.load_config import sanity_check
from utils.logging_utils import log_for_0, GoodLogger
from utils.metric_utils import Timer, MyMetrics
import wandb, re

def create_model(*, model_cls, half_precision, num_classes, **kwargs):
    """
    Create a model using the given model class.
    """
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == "tpu":
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
    """
    Initialize the model, and return the model parameters.
    """
    fake_bz = 2
    input_shape = (fake_bz, image_size, image_size, model.out_channels)
    label_shape = (fake_bz,)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing params...")
    variables = init(
        {"params": key, "gen": key, "dropout": key},
        jnp.ones(input_shape, model.dtype),
        jnp.ones(label_shape, jnp.int32),
    )
    if "batch_stats" not in variables:
        variables["batch_stats"] = {}
    log_for_0("Initializing params done.")
    return variables["params"], variables["batch_stats"]


def create_learning_rate_fn(
    training_config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int,
):
    """
    Create learning rate schedule.
    """
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=training_config.warmup_epochs * steps_per_epoch,
    )
    if training_config.lr_schedule in ["cosine", "cos"]:
        cosine_epochs = max(training_config.num_epochs - training_config.warmup_epochs, 1)
        sched_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
    elif training_config.lr_schedule == "const":
        sched_fn = optax.constant_schedule(base_learning_rate)
    else:
        raise NotImplementedError
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, sched_fn],
        boundaries=[training_config.warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


class TrainState(train_state.TrainState):
    batch_stats: Any
    ema_params: Any


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
    """
    Create initial training state.
    """
    platform = jax.local_devices()[0].platform
    if config.training.half_precision and platform == "gpu":
        raise NotImplementedError("we only support TPU yet")

    params, batch_stats = initialized(rng, image_size, model)

    # overwrite the ema net initialization
    ema_params = jax.tree_map(lambda x: jnp.array(x), params)
    assert batch_stats == {}  # we don't handle this in ema

    log_for_0("Info of model params:")
    log_for_0(params, logging_fn=print_params)

    if config.training.optimizer == "sgd":
        log_for_0("Using SGD")
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.training.momentum,
            nesterov=True,
        )
    elif config.training.optimizer == "adamw":
        log_for_0(f"Using AdamW with wd {config.training.weight_decay}")
        tx = optax.adamw(
            learning_rate=learning_rate_fn,
            b1=config.training.adam_b1,
            b2=config.training.adam_b2,
            weight_decay=config.training.weight_decay,
            # mask=mask_fn,  # TODO{km}
        )
    elif config.training.optimizer == "radam":
        log_for_0(f"Using RAdam with wd {config.training.weight_decay}")
        assert config.training.weight_decay == 0.0
        tx = optax.radam(
            learning_rate=learning_rate_fn,
            b1=config.training.adam_b1,
            b2=config.training.adam_b2,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    state = TrainState.create(
        apply_fn=functools.partial(model.apply, method=model.forward),
        params=params,
        ema_params=ema_params,
        tx=tx,
        batch_stats=batch_stats,
    )
    return state


def compute_metrics(dict_losses):
    """
    Utils function to compute metrics, used in train_step and eval_step.
    """
    metrics = dict_losses.copy()
    metrics = lax.all_gather(metrics, axis_name="batch")
    metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
    return metrics


def train_step(state, batch, rng_init, learning_rate_fn, ema_scales_fn):
    """
    Perform a single training step.
    """
    # ResNet has no dropout; but maintain rng_dropout for future usage
    rng_step = random.fold_in(rng_init, state.step)
    rng_device = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))
    rng_gen, rng_dropout = random.split(rng_device)

    ema_decay, scales = ema_scales_fn(state.step)

    def loss_fn(wrt):
        """loss function used for training."""
        outputs, new_model_state = state.apply_fn(
            {"params": wrt, "batch_stats": state.batch_stats},
            batch["image"],
            batch["label"],
            batch["augment_label"],
            scales,
            mutable=["batch_stats"],
            rngs=dict(gen=rng_gen, dropout=rng_dropout),
        )
        loss, dict_losses, images = outputs

        return loss, (new_model_state, dict_losses, images)

    # compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    new_model_state, dict_losses, images = aux[1]
    grads = lax.pmean(grads, axis_name="batch")

    # apply gradients
    state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )

    # update ema
    # NOTE{zhh}: this part is error-prone. Check?
    ema_params = jax.tree_map(lambda e, p: ema_decay * e + (1 - ema_decay) * p, state.ema_params, state.params)
    state = state.replace(ema_params=ema_params)

    # for simplicity, we don't all gather images
    lr = learning_rate_fn(state.step)
    metrics = compute_metrics(dict_losses)
    metrics["lr"] = lr
    metrics["ema_decay"] = ema_decay
    metrics["scales"] = scales
    return state, metrics, images


def sample_step(params, sample_idx, model, rng_init, device_batch_size):
    """
    Generate samples from the train state.

    sample_idx: each random sampled image corrresponds to a seed
    """
    rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
    images = generate(params, model, rng_sample, n_sample=device_batch_size)
    images_all = lax.all_gather(images, axis_name="batch")  # each device has a copy
    images_all = images_all.reshape(-1, *images_all.shape[2:])
    return images_all

def half_sample_step(params, sample_idx, model, rng_init, images, start=0.5):
    """
    Generate samples from the train state.

    sample_idx: each random sampled image corrresponds to a seed
    """
    rng_sample = random.fold_in(rng_init, sample_idx)  # fold in sample_idx
    images = generate_from_noisy(params, model, rng_sample, images, start=start)
    images_all = lax.all_gather(images, axis_name="batch")  # each device has a copy
    images_all = images_all.reshape(-1, *images_all.shape[2:])
    return images_all
    

@functools.partial(jax.pmap, axis_name="x")
def cross_replica_mean(x):
    """
    Compute an average of a variable across workers.
    """
    return lax.pmean(x, "x")

def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    if state.batch_stats == {}:
        return state
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

def run_p_sample_step(p_sample_step, state, sample_idx, ema=False):
    # redefine the interface
    images = p_sample_step(
        params={"params": state.params if not ema else state.ema_params, "batch_stats": {}},
        sample_idx=sample_idx,
    )
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return images[0]  # images have been all gathered

def get_fid_evaluater(config, p_sample_step, logger):
    inception_net = fid_util.build_jax_inception()
    stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)

    def eval_fid(state, ema_only=False):
        step = int(jax.device_get(state.step)[0])
        
        # NOTE: logging for FID should be done inside this function
        
        if not ema_only:
            samples_all = sample_util.generate_samples_for_fid_eval(
                state, config, p_sample_step, run_p_sample_step, ema=False
            )
            mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
            fid_score = fid_util.compute_fid(
                mu, stats_ref["mu"], sigma, stats_ref["sigma"]
            )
            num_gathered = samples_all.shape[0]
            k = num_gathered // 1000
            log_for_0(f"w/o ema: FID at {num_gathered} samples: {fid_score}")
            logger.log_dict(step + 1, {f"FID_{k}k": fid_score})

        samples_all = sample_util.generate_samples_for_fid_eval(
            state, config, p_sample_step, run_p_sample_step, ema=True
        )
        mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
        fid_score_ema = fid_util.compute_fid(
            mu, stats_ref["mu"], sigma, stats_ref["sigma"]
        )
        num_gathered = samples_all.shape[0]
        k = num_gathered // 1000
        log_for_0(f" w/ ema: FID at {num_gathered} samples: {fid_score_ema}")
        logger.log_dict(step + 1, {f"FID_{k}k_ema": fid_score_ema})
        return fid_score_ema
    
    return eval_fid

def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    ######################################################################
    #                       Initialize training                          #
    ######################################################################
    sanity_check(config)
    log_for_0(config)
    training_config = config.training
    if jax.process_index() == 0 and config.wandb:
        wandb.init(project="sqa_random", dir=workdir, tags=["half_fid"])
        wandb.config.update(config.to_dict())
        ka = re.search(r"kmh-tpuvm-v[234]-(\d+)(-preemptible)?-(\d+)", workdir).group()
        wandb.config.update({"ka": ka})
    logger = GoodLogger(use_wandb=config.wandb, workdir=workdir)

    rng = random.key(config.training.seed)
    global_batch_size = training_config.batch_size
    log_for_0("config.batch_size: {}".format(global_batch_size))
    if global_batch_size % jax.process_count() > 0:
        raise ValueError("Batch size must be divisible by the number of processes")
    local_batch_size = global_batch_size // jax.process_count()
    log_for_0("local_batch_size: {}".format(local_batch_size))
    log_for_0("jax.local_device_count: {}".format(jax.local_device_count()))

    if local_batch_size % jax.local_device_count() > 0:
        raise ValueError("Local batch size must be divisible by the number of local devices")

    ######################################################################
    #                           Create Dataloaders                       #
    ######################################################################
    config.dataset.use_flip = not config.aug.use_edm_aug
    train_loader, steps_per_epoch = input_pipeline.create_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    log_for_0("steps_per_epoch: {}".format(steps_per_epoch))

    ######################################################################
    #                       Create Train State                           #
    ######################################################################
    base_learning_rate = training_config.learning_rate

    model = create_model(
        model_cls=models_ddpm.SimDDPM,
        half_precision=config.training.half_precision,
        num_classes=config.dataset.num_classes,
        **config.model,
    )

    learning_rate_fn = create_learning_rate_fn(
        training_config, base_learning_rate, steps_per_epoch
    )

    state = create_train_state(rng, config, model, config.model.image_size, learning_rate_fn)

    if config.load_from != "":
        assert os.path.exists(config.load_from), "checkpoint not found. You should check GS bucket"
        log_for_0("Restoring from: {}".format(config.load_from))
        state = restore_checkpoint(state, config.load_from)
    elif config.pretrain != "":
        raise NotImplementedError("Note that this 'pretrained' have a different meaning, see how `restore_pretrained` is implemented")
        log_for_0("Loading pre-trained from: {}".format(config.restore))
        state = restore_pretrained(state, config.pretrain, config)

    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    epoch_offset = step_offset // steps_per_epoch  # sanity check for resuming
    assert epoch_offset * steps_per_epoch == step_offset, "step_offset must be divisible by steps_per_epoch, but get {} vs {}".format(step_offset, steps_per_epoch)
    state = jax_utils.replicate(state)

    ######################################################################
    #                     Prepare for Training Loop                      #
    ######################################################################
    # ema_scales_fn = functools.partial(
    #     edm_ema_scales_schedules, steps_per_epoch=steps_per_epoch, config=config
    # )
    # p_train_step = jax.pmap(
    #     functools.partial(
    #         train_step,
    #         rng_init=rng,
    #         learning_rate_fn=learning_rate_fn,
    #         ema_scales_fn=ema_scales_fn,
    #     ),
    #     axis_name="batch",
    # )
    p_sample_step = jax.pmap(
        functools.partial(
            half_sample_step,
            model=model,
            rng_init=rng,
            start=config.start,
        ),
        axis_name="batch",
    )
    vis_sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())  # for visualization
    log_for_0(f"fixed_sample_idx: {vis_sample_idx}")

    # compile p_sample_step
    log_for_0("Compiling p_sample_step...")
    B = config.training.batch_size
    B1 = jax.local_device_count()
    B2 = B // jax.device_count()
    timer = Timer()
    lowered = p_sample_step.lower(
        params={"params": state.params, "batch_stats": {}},
        sample_idx=vis_sample_idx,
        images=jnp.zeros((B1, B2, 32, 32, 3), jnp.float32),
    )
    p_sample_step = lowered.compile()
    log_for_0("p_sample_step compiled in {}s".format(timer.elapse_with_reset()))

    # prepare for FID evaluation
    if config.fid.on_use:
        # fid_evaluator = get_fid_evaluater(config, p_sample_step, logger)
        inception_net = fid_util.build_jax_inception()
        stats_ref = fid_util.get_reference(config.fid.cache_ref, inception_net)
        # # handle just_evaluate
        # if config.just_evaluate:
            # log_for_0("Sampling for images ...")
            # vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, ema=False)
            # vis = make_grid_visualization(vis)
            # logger.log_image(1, {"vis_sample": vis[0]})
            # fid_score_ema = fid_evaluator(state, ema_only=True)
            # return state

    log_for_0("Initial compilation, this might take some minutes...")

    ######################################################################
    #                           Training Loop                            #
    ######################################################################
    timer.reset()
    samples_all = []
    # print(training_config.num_epochs)
    for epoch in range(0, training_config.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch {}...".format(epoch))
        
        # # training
        # train_metrics = MyMetrics(reduction="last")
        for n_batch, batch in enumerate(train_loader):
            batch = prepare_batch_data(batch, config)
            # state, metrics, vis = p_train_step(state, batch)
            # train_metrics.update(metrics)

            images = p_sample_step(
                params={"params": state.params, "batch_stats": {}},
                sample_idx=vis_sample_idx,
                images=batch["image"],
            )
            images = images[0]  # images have been all gathered
            jax.random.normal(random.key(0), ()).block_until_ready()
            images = float_to_uint8(images)
            samples_all.append(images)

            if n_batch == 0: # visualize the first batch
                vis = make_grid_visualization(images)
                logger.log_image(0, {"vis_train": vis[0]})


            # if epoch == epoch_offset and n_batch == 0:
            #     log_for_0(
            #         "p_train_step compiled in {}s".format(timer.elapse_with_reset())
            #     )
            #     log_for_0("Initial compilation completed. Reset timer.")

            # step = epoch * steps_per_epoch + n_batch
            # ep = epoch + n_batch / steps_per_epoch
            # if training_config.get("log_per_step"):
            #     if (step + 1) % training_config.log_per_step == 0:
            #         # compute and log metrics
            #         summary = train_metrics.compute_and_reset()
            #         summary["steps_per_second"] = training_config.log_per_step / timer.elapse_with_reset()
            #         summary.update({"ep": ep, "step": step})
            #         logger.log_dict(step + 1, summary)

        # # Show training visualization
        # if (epoch + 1) % training_config.visualize_per_epoch == 0:
        #     vis = visualize_cifar_batch(vis)
        #     logger.log_image(step + 1, {"vis_train": vis[0]})

        # # Show samples (eval)
        # if (epoch + 1) % training_config.eval_per_epoch == 0:
        #     log_for_0("Sample epoch {}...".format(epoch))
        #     vis = run_p_sample_step(p_sample_step, state, vis_sample_idx, ema=False)
        #     vis = make_grid_visualization(vis)
        #     logger.log_image(step + 1, {"vis_sample": vis[0]})

        # # save checkpoint
        # if (
        #     (epoch + 1) % training_config.checkpoint_per_epoch == 0
        #     or epoch == training_config.num_epochs
        # ):
        #     state = sync_batch_stats(state)
        #     save_checkpoint(state, workdir)

        # if config.fid.on_use and (
        #     (epoch + 1) % config.fid.fid_per_epoch == 0
        #     or (epoch + 1) == training_config.num_epochs
        # ):
        #     fid_score_ema = fid_evaluator(state)
        #     # this FID can be used for saving the checkpoint with the best FID, i.e. "save_by_fid", which is not implemented yet

        break

    # here we have done one epoch over the dataset
    samples_all = jnp.concatenate(samples_all, axis=0)
    print(f"Samples shape: {samples_all.shape}")
    # eval fid
    mu, sigma = fid_util.compute_jax_fid(samples_all, inception_net)
    fid_score = fid_util.compute_fid(mu, stats_ref["mu"], sigma, stats_ref["sigma"])
    log_for_0(f' w/ ema: FID at {samples_all.shape[0]} samples: {fid_score}')

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
