# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file

from .jcm import layers, layerspp, normalization
import flax.linen as nn
import functools
import jax.numpy as jnp
import jax
import numpy as np
import ml_collections

from typing import Any, Sequence


from absl import logging

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


class NCSNpp(nn.Module):
    """NCSN++ model"""

    base_width: int = 128
    image_size: int = 32
    ch_mult: Sequence[int] = (2, 2, 2)
    num_res_blocks: int = 4
    attn_resolutions: Sequence[int] = (16,)
    dropout: float = 0.0
    fir_kernel: Sequence[int] = (1, 3, 3, 1)
    resblock_type: str = "biggan"
    fourier_scale: float = 16.0
    embedding_type: str = "fourier"

    @nn.compact
    def __call__(self, x, time_cond, augment_label=None, train=True, logging_fn=(lambda _: None)):

        assert time_cond.ndim == 1  # only support 1-d time condition
        assert time_cond.shape[0] == x.shape[0]

        # --------------------
        # redefine arguments:
        act = nn.swish

        nf = self.base_width  # config.model.nf
        ch_mult = self.ch_mult
        num_res_blocks = self.num_res_blocks
        attn_resolutions = self.attn_resolutions
        dropout = self.dropout
        resamp_with_conv = True
        num_resolutions = len(ch_mult)

        conditional = True  # noise-conditional
        fir = True
        fir_kernel = self.fir_kernel
        skip_rescale = True
        resblock_type = self.resblock_type
        progressive = "none"
        embedding_type = self.embedding_type
        progressive_input = "residual"
        fourier_scale = self.fourier_scale
        init_scale = 0.0

        assert progressive in ["none", "output_skip", "residual"]
        assert progressive_input in ["none", "input_skip", "residual"]
        assert embedding_type in ["fourier", "positional"]

        combine_method = "sum"
        combiner = functools.partial(Combine, method=combine_method)

        double_heads = False
        # --------------------

        # timestep/noise_level embedding; only for continuous training
        if embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            temb = layerspp.GaussianFourierProjection(
                embedding_size=nf,
                scale=fourier_scale,
                name="map_noise",
            )(time_cond)
        elif embedding_type == "positional":
            # raise NotImplementedError
            # Sinusoidal positional embeddings.
            temb = layers.get_timestep_embedding(time_cond, nf)
        else:
            raise NotImplementedError
            raise ValueError(f"embedding type {embedding_type} unknown.")

        if augment_label is not None:
            aemb = nn.Dense(
                nf * 2,
                kernel_init=default_initializer(),
                use_bias=False,
                name="map_augment",
            )(augment_label)
            temb += aemb

        if conditional:
            temb = nn.Dense(
                nf * 4, kernel_init=default_initializer(), name="map_layer0"
            )(temb)
            temb = nn.Dense(
                nf * 4, kernel_init=default_initializer(), name="map_layer1"
            )(act(temb))
        else:
            raise NotImplementedError
            temb = None

        AttnBlock = functools.partial(
            layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale
        )

        Upsample = functools.partial(
            layerspp.Upsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive == "output_skip":
            raise NotImplementedError
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive == "residual":
            raise NotImplementedError
            pyramid_upsample = functools.partial(
                layerspp.Upsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        Downsample = functools.partial(
            layerspp.Downsample,
            with_conv=resamp_with_conv,
            fir=fir,
            fir_kernel=fir_kernel,
        )

        if progressive_input == "input_skip":
            raise NotImplementedError
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=False
            )
        elif progressive_input == "residual":
            pyramid_downsample = functools.partial(
                layerspp.Downsample, fir=fir, fir_kernel=fir_kernel, with_conv=True
            )

        if resblock_type == "ddpm":
            raise NotImplementedError
            ResnetBlock = functools.partial(
                ResnetBlockDDPM,
                act=act,
                dropout=dropout,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
            )

        elif resblock_type == "biggan":
            ResnetBlock = functools.partial(
                ResnetBlockBigGAN,
                act=act,
                dropout=dropout,
                fir=fir,
                fir_kernel=fir_kernel,
                init_scale=init_scale,
                skip_rescale=skip_rescale,
            )

        else:
            raise ValueError(f"resblock type {resblock_type} unrecognized.")

        # utility function to count number of parameters
        def pms(self, name):
            tree = jax.tree_map(
                lambda x: np.prod(x.shape), self.variables["params"][name]
            )
            return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)

        ps = functools.partial(pms, self)

        # begin of the work
        # Downsampling block

        cur_size = self.image_size

        input_pyramid = None
        if progressive_input != "none":
            input_pyramid = x

        logging_fn(f"Input shape {x.shape}")
        name = f"enc_{cur_size}x{cur_size}_conv"  # 32x32_conv
        hs = [conv3x3(x, nf, name=name)]
        logging_fn(f"{name}: params {ps(name)}, shape {hs[-1].shape}")
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                name = f"enc_{cur_size}x{cur_size}_block{i_block}"
                h = ResnetBlock(out_ch=nf * ch_mult[i_level], name=name)(
                    hs[-1], temb, train
                )
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
                if h.shape[1] in attn_resolutions:
                    name = f"enc_{cur_size}x{cur_size}_block{i_block}_attn"
                    h = AttnBlock(name=name)(h)
                    logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
                hs.append(h)

            if i_level != num_resolutions - 1:
                if resblock_type == "ddpm":
                    raise NotImplementedError
                    h = Downsample()(hs[-1])
                else:
                    cur_size //= 2
                    name = f"enc_{cur_size}x{cur_size}_down"
                    h = ResnetBlock(down=True, name=name)(hs[-1], temb, train)
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

                if progressive_input == "input_skip":
                    raise NotImplementedError
                    input_pyramid = pyramid_downsample()(input_pyramid)
                    h = combiner()(input_pyramid, h)

                elif progressive_input == "residual":
                    name = f"enc_{cur_size}x{cur_size}_aux_residual"
                    input_pyramid = pyramid_downsample(out_ch=h.shape[-1], name=name)(
                        input_pyramid
                    )
                    if skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(
                            2.0, dtype=np.float32
                        )
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                    logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

                # logging_fn(f"Level {i_level}, combined shape {h.shape}")
                hs.append(h)

        h = hs[-1]
        name = f"dec_{cur_size}x{cur_size}_in0"
        h = ResnetBlock(name=name)(h, temb, train)
        logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

        name = f"dec_{cur_size}x{cur_size}_in0_attn"
        h = AttnBlock(name=name)(h)
        logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

        name = f"dec_{cur_size}x{cur_size}_in1"
        h = ResnetBlock(name=name)(h, temb, train)
        logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                name = f"dec_{cur_size}x{cur_size}_block{i_block}"
                h = ResnetBlock(out_ch=nf * ch_mult[i_level], name=name)(
                    jnp.concatenate([h, hs.pop()], axis=-1), temb, train
                )
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

            if h.shape[1] in attn_resolutions:
                name = f"dec_{cur_size}x{cur_size}_block{i_block}_attn"
                h = AttnBlock(name=name)(h)
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

            if progressive != "none":
                raise NotImplementedError
                if i_level == num_resolutions - 1:
                    if progressive == "output_skip":
                        pyramid = conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            x.shape[-1],
                            bias=True,
                            init_scale=init_scale,
                        )
                    elif progressive == "residual":
                        pyramid = conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            h.shape[-1],
                            bias=True,
                        )
                    else:
                        raise ValueError(f"{progressive} is not a valid name.")
                else:
                    if progressive == "output_skip":
                        pyramid = pyramid_upsample()(pyramid)
                        pyramid = pyramid + conv3x3(
                            act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h)),
                            x.shape[-1],
                            bias=True,
                            init_scale=init_scale,
                        )
                    elif progressive == "residual":
                        pyramid = pyramid_upsample(out_ch=h.shape[-1])(pyramid)
                        if skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0, dtype=np.float32)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f"{progressive} is not a valid name")

            if i_level != 0:
                if resblock_type == "ddpm":
                    raise NotImplementedError
                    h = Upsample()(h)
                else:
                    cur_size *= 2
                    name = f"dec_{cur_size}x{cur_size}_up"
                    h = ResnetBlock(up=True, name=name)(h, temb, train)
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")

        assert not hs

        if progressive == "output_skip" and not double_heads:
            raise NotImplementedError
            h = pyramid
        else:
            name = f"dec_{cur_size}x{cur_size}_aux_norm"
            h = act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32), name=name)(h))
            logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
            if double_heads:
                raise NotImplementedError
                h = conv3x3(h, x.shape[-1] * 2, init_scale=init_scale)
            else:
                name = f"dec_{cur_size}x{cur_size}_aux_conv"
                h = conv3x3(h, x.shape[-1], init_scale=init_scale, name=name)
                logging_fn(f"{name}: params {ps(name)}, shape {h.shape}")
        logging_fn(f"Output shape {h.shape}")
        return h
