# random
Try random ideas

Based on: Flow Matching (https://github.com/Hope7Happiness/jax_dev_gen_base)

# JAX+linen Development Codebase for Diffusion Models

In this branch, we provide a codebase for training (Naive) Flow Matching (FM)+CIFAR on JAX+linen.


## Setting up the environment

Don't forget to add your `config.sh`: 

```shell
export EU_IMAGENET_FAKE="/kmh-nfs-ssd-eu-mount/code/hanhong/data/imagenet_fake/kmh-nfs-ssd-eu-mount/" # path to the fake imagenet data at EU
export US_IMAGENET_FAKE="/kmh-nfs-ssd-eu-mount/code/hanhong/data/imagenet_fake/kmh-nfs-us-mount/" # path to the fake imagenet data at US
export OWN_CONDA_ENV_NAME=NNX # A useable env at your V3-8
export WANDB_API_KEY= # Your wandb API key
export TASKNAME=fm_linen # Task name
```

## Expected Performance with (naive) FM

| Card | Run Time | FID-50k | Step per sec | url |
| --- | --- | --- | --- | --- |
| V3-32 | 7h 17m 48s | 2.97 | ~3.3 | [woven-wildflower-18](https://wandb.ai/zhh24-massachusetts-institute-of-technology/fm_linen/runs/4trs1b98) |
| V2-32 | ~14h | TBD
| V4-8 | 16h 27m 47s | 2.87 | ~1.3 | [serene-silence-30](https://wandb.ai/zhh24-massachusetts-institute-of-technology/fm_linen/runs/3w00fnpf) |

## Expected Performance with FM **w/o t**

| Card | Run Time | FID-50k | Step per sec | url |
| --- | --- | --- | --- | --- |
| V3-32 | ~7h | TBD
| V2-32 | 14h 22m 34s | 2.63 | ~1.7 | [celestial-glade-17](https://wandb.ai/zhh24-massachusetts-institute-of-technology/fm_linen/runs/cdxbsab5) |

## Subtlety (for naive FM)

How to deal with the `1e-3` number in time condition is a problem. Aside from the current implementation, another alternative implementation is in commit `2986720`. It leads to a better FID around 2.89, in the wandb run [playful-lake-15](https://wandb.ai/zhh24-massachusetts-institute-of-technology/fm_linen/runs/7xn3dtkb) (for train) and [mild-fog-24](https://wandb.ai/zhh24-massachusetts-institute-of-technology/fm_linen_eval/runs/faplgu62/overview) (for eval).

Although it has a better FID, we believe that it is not very intuitive (the time $t$ is switched to $[10^{-3},1]$ instead of $[0,1]$), we will keep the current implementation for now. Actually, this 0.1 improvement in FID can also be achieved in a variety of other ways, so it is not a big deal.

# The following is the original README

# Jax re-implementation of EDM

Written by Kaiming He.

### Introduction

Before you start, please read [He Vision Group's TPU intro repo](https://github.com/KaimingHe/resnet_jax).

### Usage

See `run_script.sh` for a dev run in local TPU VM.

See `run_remote.sh` for preparing a remote job in a remote multi-node TPU VM. Then run `run_staging.sh` to kick off.

### Results

FID: running for 4000 epochs gives 2.1~2.3 FID with N=18 sampling steps. As a reference, running the original Pytorch/GPU code has 2.04 FID, and 1.97 was reported in the paper (unconditional).

Time: set `config.fid.fid_per_epoch` to a bigger number (200 or 500) to reduce FID evaluation frequency. If FID evaluation is not much, training for 4000 epochs should take ~12 hours in a v3-32 VM. Training for 1000 epochs (~3 hours) should give ~2.6 FID.

### Known issues

EDM turns dropout on even at inference time. My code turns dropout off at inference time. This seems to have negligible difference.
