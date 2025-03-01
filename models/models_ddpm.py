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

"""Flax implementation of ResNet V1.5."""

# See issue #620.
# pytype: disable=wrong-arg-count

from absl import logging
from typing import Any, Sequence

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from functools import partial

from models.models_unet import ContextUnet
from models.models_ncsnpp_edm import NCSNpp as NCSNppEDM
from models.models_ncsnpp import NCSNpp
import models.jcm.sde_lib as sde_lib
from models.jcm.sde_lib import batch_mul
from functools import reduce
def compose(*funcs):
    return lambda x: reduce(lambda v, f: f(v), funcs, x)

ModuleDef = Any


def ct_ema_scales_schedules(step, config, steps_per_epoch):
    start_ema = float(config.ct.start_ema)
    start_scales = int(config.ct.start_scales)
    end_scales = int(config.ct.end_scales)
    total_steps = config.num_epochs * steps_per_epoch

    scales = jnp.ceil(
        jnp.sqrt(
            (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
            + start_scales**2
        )
        - 1
    ).astype(jnp.int32)
    scales = jnp.maximum(scales, 1)
    c = -jnp.log(start_ema) * start_scales
    target_ema = jnp.exp(-c / scales)
    scales = scales + 1
    return target_ema, scales


def edm_ema_scales_schedules(step, config, steps_per_epoch):
    # ema_halflife_kimg = 500  # from edm
    ema_halflife_kimg = 50000  # from flow
    ema_halflife_nimg = ema_halflife_kimg * 1000

    ema_rampup_ratio = 0.05
    ema_halflife_nimg = jnp.minimum(
        ema_halflife_nimg, step * config.training.batch_size * ema_rampup_ratio
    )

    ema_beta = 0.5 ** (config.training.batch_size / jnp.maximum(ema_halflife_nimg, 1e-8))
    scales = jnp.ones((1,), dtype=jnp.int32)
    return ema_beta, scales


# move this out from model for JAX compilation
def generate(params, model, rng, n_sample):
    """
    Generate samples from the model
    """

    # prepare schedule
    num_steps = model.n_T
    step_indices = jnp.arange(num_steps, dtype=model.dtype)
    # t_steps = jnp.linspace(1e-3, 1.0, num_steps + 1, dtype=model.dtype)
    t_steps = jnp.linspace(0.0, 1.0, num_steps + 1, dtype=model.dtype)

    # initialize noise
    x_shape = (n_sample, model.image_size, model.image_size, model.out_channels)
    rng_used, rng = jax.random.split(rng, 2)
    latents = jax.random.normal(
        rng_used, x_shape, dtype=model.dtype
    )  # x_T ~ N(0, 1), sample initial noise

    x_i = latents

    def step_fn(i, inputs):
        x_i, rng = inputs
        rng_this_step = jax.random.fold_in(rng, i)
        rng_z, rng_dropout = jax.random.split(rng_this_step, 2)
        x_i, _ = model.apply(
            params,  # which is {'params': state.params, 'batch_stats': state.batch_stats},
            x_i,
            rng_z,
            i,
            t_steps,
            # rngs={'dropout': rng_dropout},  # we don't do dropout in eval
            rngs={},
            method=model.sample_one_step,
            mutable=["batch_stats"],
        )
        outputs = (x_i, rng)
        return outputs

    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_i, rng))
    images = outputs[0]
    return images

# move this out from model for JAX compilation
def generate_from_noisy(params, model, rng, images, start=0.5):
    """
    Generate samples from the model
    images: clean images
    """

    # prepare schedule
    num_steps = model.n_T
    # step_indices = jnp.arange(num_steps, dtype=model.dtype)
    # t_steps = jnp.linspace(1e-3, 1.0, num_steps + 1, dtype=model.dtype)
    t_steps = jnp.linspace(start, 1.0, num_steps + 1, dtype=model.dtype)

    只能用一次, 别传进去 = jax.random.split(rng, 2)
    只能用一次啊, 别传进去 = jax.random.split(别传进去, 2)

    noise = jax.random.normal(只能用一次, images.shape, dtype=model.dtype)
    B = images.shape[0]
    t_batch = start * jnp.ones((B, ), dtype=model.dtype)
    x_i = batch_mul(images, t_batch) + batch_mul(noise, 1 - t_batch)

    # # initialize noise
    # x_shape = (n_sample, model.image_size, model.image_size, model.out_channels)
    # rng_used, rng = jax.random.split(rng, 2)
    # latents = jax.random.normal(
    #     rng_used, x_shape, dtype=model.dtype
    # )  # x_T ~ N(0, 1), sample initial noise

    # x_i = latents

    def step_fn(i, inputs):
        x_i, rng = inputs
        rng_this_step = jax.random.fold_in(rng, i)
        rng_z, rng_dropout = jax.random.split(rng_this_step, 2)
        x_i, _ = model.apply(
            params,  # which is {'params': state.params, 'batch_stats': state.batch_stats},
            x_i,
            rng_z,
            i,
            t_steps,
            # rngs={'dropout': rng_dropout},  # we don't do dropout in eval
            rngs={},
            method=model.sample_one_step,
            mutable=["batch_stats"],
        )
        outputs = (x_i, rng)
        return outputs

    outputs = jax.lax.fori_loop(0, num_steps, step_fn, (x_i, 只能用一次啊))
    images = outputs[0]
    return images

class SimDDPM(nn.Module):
    """Simple DDPM."""

    image_size: int
    base_width: int
    num_classes: int = 10
    out_channels: int = 1
    P_std: float = 1.2
    P_mean: float = -1.2
    n_T: int = 18  # inference steps
    net_type: str = "ncsnpp"
    dropout: float = 0.0
    dtype: Any = jnp.float32
    use_aug_label: bool = False
    t_cond_method: str = "log999"
    embedding_type: str = "fourier"

    def setup(self):

        if self.net_type == "context":
            raise NotImplementedError
            net_fn = partial(
                ContextUnet,
                in_channels=self.out_channels,
                n_feat=self.base_width,
                n_classes=self.num_classes,
                image_size=self.image_size,
            )
        elif self.net_type == "ncsnpp":
            raise NotImplementedError
            net_fn = partial(
                NCSNpp,
                base_width=self.base_width,
                image_size=self.image_size,
                dropout=self.dropout,
            )
        elif self.net_type == "ncsnppedm":
            net_fn = partial(
                NCSNppEDM,
                base_width=self.base_width,
                image_size=self.image_size,
                dropout=self.dropout,
                embedding_type=self.embedding_type,
            )
        else:
            raise ValueError(f"Unknown net type: {self.net_type}")

        # declare two networks
        self.net = net_fn(name="net")
        self.t_conder = (
            # TODO: I think this still doesn't match the original implementation during training. But maybe it's okay?
            logging.info("Use t-cond: log999t")  or (lambda t: jnp.log(999*((1-1e-3)*t+1e-3)))
        ) if self.t_cond_method == "log999" else (
            logging.info("Use t-cond: not")  or (lambda t: t * 0.0)
        ) if self.t_cond_method == "not" else (
            logging.info("Use t-cond: ???") or exec(f"raise ValueError('Unknown t_cond_method: {self.t_cond_method}')")
        )
        self.t_conder = compose(self.t_conder, lambda x: x.reshape(x.shape[0]))

    def get_visualization(self, list_imgs):
        vis = jnp.concatenate(list_imgs, axis=1)
        return vis

    def compute_losses(self, pred, gt):
        assert pred.shape == gt.shape

        # simple l2 loss
        loss_rec = jnp.mean((pred - gt) ** 2)

        loss_train = loss_rec

        dict_losses = {"loss_rec": loss_rec, "loss_train": loss_train}
        return loss_train, dict_losses

    def sample_one_step(self, x_i, rng, i, t_steps):

        x_cur = x_i
        t_cur = t_steps[i].repeat(x_cur.shape[0])
        t_next = t_steps[i + 1].repeat(x_cur.shape[0])
        
        net_call = self.net(x_cur, self.t_conder(t_cur),  augment_label=None, train=False)
        x_next = x_cur + batch_mul(net_call, t_next - t_cur)

        return x_next

    def compute_t(self, indices, scales):
        sde = self.sde
        t = sde.t_max ** (1 / sde.rho) + indices / (scales - 1) * (
            sde.t_min ** (1 / sde.rho) - sde.t_max ** (1 / sde.rho)
        )
        t = t**sde.rho
        return t

    def forward_consistency_function(self, x, t, pred_t=None):
        raise NotImplementedError
        c_in = 1 / jnp.sqrt(t**2 + self.sde.data_std**2)
        in_x = batch_mul(x, c_in)  # input scaling of edm
        cond_t = 0.25 * jnp.log(t)  # noise cond of edm

        # forward
        denoiser = self.net(in_x, cond_t)

        if pred_t is None:  # TODO: what's this?
            pred_t = self.sde.t_min

        c_out = (t - pred_t) * self.sde.data_std / jnp.sqrt(t**2 + self.sde.data_std**2)
        denoiser = batch_mul(denoiser, c_out)

        c_skip = self.sde.data_std**2 / ((t - pred_t) ** 2 + self.sde.data_std**2)
        skip_x = batch_mul(x, c_skip)

        denoiser = skip_x + denoiser

        return denoiser

    def forward_edm_denoising_function(
        self, x, sigma, augment_label=None, train: bool = True
    ):  # EDM
        raise NotImplementedError
        c_skip = self.sde.data_std**2 / (sigma**2 + self.sde.data_std**2)
        c_out = sigma * self.sde.data_std / jnp.sqrt(sigma**2 + self.sde.data_std**2)

        c_in = 1 / jnp.sqrt(sigma**2 + self.sde.data_std**2)
        c_noise = 0.25 * jnp.log(sigma)

        # forward network
        in_x = batch_mul(x, c_in)
        c_noise = c_noise.reshape(c_noise.shape[0])

        F_x = self.net(in_x, c_noise, augment_label=augment_label, train=train)

        D_x = batch_mul(x, c_skip) + batch_mul(F_x, c_out)
        return D_x

    def forward(self, imgs, labels, augment_label, train: bool = True):
        imgs = imgs.astype(self.dtype)
        gt = imgs
        x = imgs
        bz = imgs.shape[0]

        # -----------------------------------------------------------------

        sigma = jax.random.uniform(
            self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype, minval=0.0, maxval=1.0
        )
        # sigma = jax.random.uniform(
        #     self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype, minval=1e-3, maxval=1.0
        # )
        noise = (
            jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        )
        xn = batch_mul(x, sigma) + batch_mul(noise, 1 - sigma) # convention: flow from noise to data
        gt = x - noise
        
        net_call = self.net(xn, self.t_conder(sigma), augment_label=augment_label, train=True)

        loss = (net_call - gt) ** 2
        loss = jnp.mean(loss, axis=(1, 2, 3))  # sum over pixels # average loss by default
        loss = loss.mean()  # mean over batch

        loss_train = loss

        dict_losses = {}
        dict_losses["loss"] = loss  # legacy
        # dict_losses["loss_train"] = loss_train
        
        # convert the velocity predictor to x-predictor
        pred_x = xn + batch_mul(net_call, 1 - sigma)

        images = self.get_visualization([imgs, xn, pred_x])
        return loss_train, dict_losses, images

    def __call__(self, imgs, labels, train: bool = False):
        # initialization only
        t = jnp.ones((imgs.shape[0],))
        augment_label = (
            jnp.ones((imgs.shape[0], 9)) if self.use_aug_label else None
        )  # fixed augment_dim
        out = self.net(imgs, t, augment_label)
        out_ema = None  # no need to initialize it here
        return out, out_ema
