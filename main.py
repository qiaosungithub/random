import jax

jax.distributed.initialize()
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

"""Main file for running the ImageNet example.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

import os
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags

import train
from utils import logging_utils

logging_utils.supress_checkpt_info()

import warnings

warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_enum(
    "mode",
    enum_values=["local_debug", "remote_debug", "remote_run"],
    default="remote_run",
    help="Running mode.",
)  # NOTE: This variable isn't used currently, but maintained for future use. This at least ensures that there is no more variable that must be passed in from the command line.

flags.DEFINE_bool("debug", False, "Debugging mode.")
config_flags.DEFINE_config_file(
    "config",
    help_string="File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    c = FLAGS.config

    def f():
        if FLAGS.debug:
            with jax.disable_jit():
                train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
        else:
            train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    # for n in [10, 50, 100]:
    #     c.model.n_T = n
    #     c.start = 0.5
    #     c.wandb_name = f"sqa_start_0.5_{n}_steps_euler"
    #     f()

    # for n in [10, 50, 100]:
    #     c.model.n_T = n
    #     c.start = 0.8
    #     c.wandb_name = f"sqa_start_0.8_{n}_steps_euler"
    #     f()

    # for n in [0, 1, 2, 5]:
    #     c.model.n_T = n
    #     c.start = 0.99
    #     c.wandb_name = f"sqa_start_0.99_{n}_steps_euler"
    #     f()

    # for start in [0.99]:
    #     c.model.n_T = 0
    #     c.start = start
    #     c.wandb_name = f"sqa_start_{start}_0_step"
    #     f()

    for scale in [0.5, 0.8, 1.2, 1.5]:
        c.model.n_T == 0
        c.start = 1.00
        c.scale = scale
        c.wandb_name = f"sqa_scale_{scale}"
        f()
    
    # c.model.sampler = "heun"
    # for n in [2, 5]:
    #     c.model.n_T = n
    #     c.start = 0.99
    #     c.wandb_name = f"sqa_start_0.99_{n}_steps_heun"
    #     f()

if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir", "mode", "config"])
    app.run(main)