from absl import logging
import jax
from flax.training import checkpoints
from .logging_utils import log_for_0


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    log_for_0("Saving checkpoint step %d.", step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=2)


def restore_pretrained(state, path, config):
    pretrained = checkpoints.restore_checkpoint(path, target=None)
    log_for_0(f"pretrained model: {pretrained.keys()}")

    assert jax.tree_structure(state.params["Encoder"]) == jax.tree_structure(
        pretrained["params"]["Encoder"]
    )
    assert jax.tree_structure(state.params["Decoder"]) == jax.tree_structure(
        pretrained["params"]["Decoder"]
    )

    state.params["Encoder"] = pretrained["params"]["Encoder"]
    state.params["Decoder"] = pretrained["params"]["Decoder"]

    # just in case
    # assert jax.tree_structure(state.batch_stats) == \
    #   jax.tree_structure(pretrained['batch_stats'])
    # state = state.replace(batch_stats=pretrained['batch_stats'])

    log_for_0("Loaded.")
    return state
