# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline neural-texture compression implemented with warp-nn.

Nothing in the inference/runtime path imports this module. Training follows the
NTC design: two learned latent grids, positional features, and a tiny per-asset
MLP. A final refinement phase freezes 4-bit-quantized latents so the exported
decoder is optimized for the representation it will actually consume.
"""

from __future__ import annotations

import numpy as np
import warp as wp

try:
    from warp_nn import nn, optimizers
except ImportError as error:  # pragma: no cover - exercised without training extra
    raise ImportError(
        "neural-texture compression requires the optional 'warp-nn' package"
    ) from error

from .format import (
    CompressionResult,
    NetworkLayer,
    NeuralTextureAsset,
    TextureChannel,
    pack_latents,
    unpack_latents,
)
from .reference import _quantize_fp8_e4m3, decode_image


@wp.kernel
def _build_inputs(
    latent0: wp.array3d(dtype=wp.float32),
    latent1: wp.array3d(dtype=wp.float32),
    pixel_indices: wp.array(dtype=wp.int32),
    image_width: int,
    image_height: int,
    output: wp.array2d(dtype=wp.float32),
):
    sample = wp.tid()
    pixel = pixel_indices[sample]
    x = pixel % image_width
    y = pixel // image_width
    u = (wp.float32(x) + 0.5) / wp.float32(image_width)
    v = (wp.float32(y) + 0.5) / wp.float32(image_height)
    for feature in range(8):
        for grid_index in range(2):
            width = latent0.shape[1] if grid_index == 0 else latent1.shape[1]
            height = latent0.shape[0] if grid_index == 0 else latent1.shape[0]
            fx = u * wp.float32(width) - 0.5
            fy = v * wp.float32(height) - 0.5
            x0 = int(wp.floor(fx))
            y0 = int(wp.floor(fy))
            tx = fx - wp.float32(x0)
            ty = fy - wp.float32(y0)
            ix0 = ((x0 % width) + width) % width
            ix1 = (((x0 + 1) % width) + width) % width
            iy0 = ((y0 % height) + height) % height
            iy1 = (((y0 + 1) % height) + height) % height
            if grid_index == 0:
                s00 = latent0[iy0, ix0, feature]
                s10 = latent0[iy0, ix1, feature]
                s01 = latent0[iy1, ix0, feature]
                s11 = latent0[iy1, ix1, feature]
            else:
                s00 = latent1[iy0, ix0, feature]
                s10 = latent1[iy0, ix1, feature]
                s01 = latent1[iy1, ix0, feature]
                s11 = latent1[iy1, ix1, feature]
            output[sample, grid_index * 8 + feature] = (
                s00 * (1.0 - tx) * (1.0 - ty)
                + s10 * tx * (1.0 - ty)
                + s01 * (1.0 - tx) * ty
                + s11 * tx * ty
            )
    px = wp.float32(x) / wp.float32(image_width)
    py = wp.float32(y) / wp.float32(image_height)
    for wave in range(3):
        index = 16 + wave * 4
        output[sample, index] = (px - wp.floor(px)) * 2.0 - 1.0
        output[sample, index + 1] = (py - wp.floor(py)) * 2.0 - 1.0
        output[sample, index + 2] = (px + 0.25 - wp.floor(px + 0.25)) * 2.0 - 1.0
        output[sample, index + 3] = (py + 0.25 - wp.floor(py + 0.25)) * 2.0 - 1.0
        px *= 2.0
        py *= 2.0
    output[sample, 28] = 0.0
    output[sample, 29] = 0.0
    output[sample, 30] = 0.0
    output[sample, 31] = 0.0


@wp.kernel
def _hgelu(input: wp.array2d(dtype=wp.float32), output: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    value = input[i, j]
    output[i, j] = wp.min(value, 3.0) * wp.clamp(value / 3.0 + 0.5, 0.0, 1.0)


@wp.kernel
def _mse(
    prediction: wp.array2d(dtype=wp.float32),
    target: wp.array2d(dtype=wp.float32),
    normalization: float,
    loss: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()
    difference = prediction[i, j] - target[i, j]
    wp.atomic_add(loss, 0, difference * difference * normalization)


@wp.func
def _fake_quantize_fp8_e4m3(value: float):
    magnitude = wp.min(wp.abs(value), 448.0)
    step = 0.001953125
    if magnitude >= 0.015625:
        exponent = wp.floor(wp.log2(magnitude))
        step = wp.pow(2.0, exponent - 3.0)
    scaled = magnitude / step
    lower = wp.floor(scaled)
    fraction = scaled - lower
    rounded = lower
    if fraction > 0.5 or (fraction == 0.5 and int(lower) % 2 == 1):
        rounded += 1.0
    result = wp.min(rounded * step, 448.0)
    if value < 0.0:
        result = -result
    return result


@wp.func_grad(_fake_quantize_fp8_e4m3)
def _adj_fake_quantize_fp8_e4m3(value: float, adj_ret: float):
    wp.adjoint[value] += adj_ret


@wp.kernel
def _fake_quantize_fp8_e4m3_kernel(
    input: wp.array2d(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    i, j = wp.tid()
    output[i, j] = _fake_quantize_fp8_e4m3(input[i, j])


class _FP8Quantizer:
    def __init__(self):
        self._cache = {}

    def __call__(self, input):
        key = (tuple(input.shape), input.dtype)
        if key not in self._cache:
            self._cache[key] = wp.empty(
                input.shape, dtype=wp.float32, device=input.device, requires_grad=True
            )
        output = self._cache[key]
        wp.launch(
            _fake_quantize_fp8_e4m3_kernel,
            dim=input.shape,
            inputs=[input],
            outputs=[output],
            device=input.device,
        )
        return output


class _HGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = {}

    def __call__(self, input):
        key = (tuple(input.shape), input.dtype)
        if key not in self._cache:
            self._cache[key] = wp.empty(
                input.shape, dtype=wp.float32, device=input.device, requires_grad=True
            )
        output = self._cache[key]
        wp.launch(
            _hgelu,
            dim=input.shape,
            inputs=[input],
            outputs=[output],
            device=input.device,
        )
        return output


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(32, 32)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 16)
        self.activation0 = _HGELU()
        self.activation1 = _HGELU()
        self.quantizer0 = _FP8Quantizer()
        self.quantizer1 = _FP8Quantizer()
        self.quantizer2 = _FP8Quantizer()
        self.fp8_inputs = False
        super().__post_init__()

    def __call__(self, inputs):
        if self.fp8_inputs:
            inputs = self.quantizer0(inputs)
        hidden0 = self.activation0(self.fc0(inputs))
        if self.fp8_inputs:
            hidden0 = self.quantizer1(hidden0)
        hidden1 = self.activation1(self.fc1(hidden0))
        if self.fp8_inputs:
            hidden1 = self.quantizer2(hidden1)
        return self.fc2(hidden1)


def _quantized_array(array: wp.array, device: str) -> tuple[wp.array, np.ndarray]:
    host = array.numpy()
    packed = pack_latents(host)
    restored = unpack_latents(packed, 8)
    return wp.array(restored, dtype=wp.float32, device=device), packed


def _project_model_weights(model: _Decoder) -> None:
    for linear in (model.fc0, model.fc1, model.fc2):
        linear.weight.data.assign(_quantize_fp8_e4m3(linear.weight.data.numpy()))


def compress_texture(
    image: np.ndarray,
    *,
    steps: int = 1000,
    refinement_steps: int | None = None,
    learning_rate: float = 1e-2,
    latent_scale: int = 4,
    batch_size: int = 65536,
    device: str = "cuda:0",
    seed: int = 42,
    channel_name: str = "texture",
    color_space: str = "linear",
) -> CompressionResult:
    """Train and export one texture as a v1 neural-texture asset.

    ``image`` must be float-like ``(height, width, channels)`` data normalized
    to ``[0, 1]`` with at most 16 channels. ``steps`` trains latents and decoder;
    ``refinement_steps`` then freezes quantized latents and refines an FP8-
    projected decoder. Training samples at most ``batch_size`` pixels per step.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or not 1 <= image.shape[2] <= 16:
        raise ValueError("image must have shape (height, width, 1..16)")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image dimensions must be positive")
    if not np.all(np.isfinite(image)) or np.min(image) < 0.0 or np.max(image) > 1.0:
        raise ValueError("image values must be finite and normalized to [0, 1]")
    if steps < 0 or (refinement_steps is not None and refinement_steps < 0):
        raise ValueError("training step counts must be non-negative")
    if latent_scale <= 0:
        raise ValueError("latent_scale must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if refinement_steps is None:
        refinement_steps = max(steps // 4, 1) if steps else 0

    height, width, channels = image.shape
    rng = np.random.default_rng(seed)
    latent0_shape = (
        max((height + latent_scale - 1) // latent_scale, 1),
        max((width + latent_scale - 1) // latent_scale, 1),
        8,
    )
    latent1_shape = (
        max((latent0_shape[0] + 1) // 2, 1),
        max((latent0_shape[1] + 1) // 2, 1),
        8,
    )
    target_host = np.zeros((height * width, 16), dtype=np.float32)
    target_host[:, :channels] = image.reshape(-1, channels)

    batch_count = min(batch_size, height * width)
    with wp.ScopedDevice(device):
        wp.rand_init(seed)
        latent0 = wp.array(
            rng.uniform(-0.1, 0.1, latent0_shape).astype(np.float32),
            device=device,
            requires_grad=True,
        )
        latent1 = wp.array(
            rng.uniform(-0.1, 0.1, latent1_shape).astype(np.float32),
            device=device,
            requires_grad=True,
        )
        target = wp.empty((batch_count, 16), dtype=wp.float32, device=device)
        pixel_indices = wp.empty(batch_count, dtype=wp.int32, device=device)
        inputs = wp.empty(
            (batch_count, 32), dtype=wp.float32, device=device, requires_grad=True
        )
        loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        model = _Decoder()
        model.to(device)
        optimizer = optimizers.Adam(
            [latent0, latent1, *model.parameters()],
            lr=learning_rate,
            device=device,
            disable_graph=True,
        )
        losses = []

        def train_step(
            active_latent0, active_latent1, active_optimizer, project_weights=False
        ):
            loss.zero_()
            if batch_count == height * width:
                indices = np.arange(batch_count, dtype=np.int32)
            else:
                indices = rng.integers(0, height * width, batch_count, dtype=np.int32)
            pixel_indices.assign(indices)
            target.assign(target_host[indices])
            if project_weights:
                _project_model_weights(model)

            with wp.Tape() as tape:
                wp.launch(
                    _build_inputs,
                    dim=batch_count,
                    inputs=[
                        active_latent0,
                        active_latent1,
                        pixel_indices,
                        width,
                        height,
                    ],
                    outputs=[inputs],
                    device=device,
                )
                prediction = model(inputs)
                wp.launch(
                    _mse,
                    dim=prediction.shape,
                    inputs=[prediction, target, 1.0 / prediction.size],
                    outputs=[loss],
                    device=device,
                )
            tape.backward(loss)
            active_optimizer.step()
            value = float(loss.numpy()[0])
            tape.zero()
            return value

        for _ in range(steps):
            losses.append(train_step(latent0, latent1, optimizer))

        quantized0, packed0 = _quantized_array(latent0, device)
        quantized1, packed1 = _quantized_array(latent1, device)
        refine_optimizer = optimizers.Adam(
            model.parameters(),
            lr=learning_rate * 0.25,
            device=device,
            disable_graph=True,
        )
        model.fp8_inputs = True
        for _ in range(refinement_steps):
            losses.append(train_step(quantized0, quantized1, refine_optimizer, True))

        _project_model_weights(model)
        layers = []
        for index, linear in enumerate((model.fc0, model.fc1, model.fc2)):
            layers.append(
                NetworkLayer(
                    linear.weight.data.numpy().astype(np.float16),
                    linear.bias.data.numpy().reshape(-1).astype(np.float16),
                    "none" if index == 2 else "hgelu",
                )
            )
        asset = NeuralTextureAsset(
            width=width,
            height=height,
            channels=tuple(
                TextureChannel(
                    channel_name if first == 0 else f"{channel_name}_{first // 4}",
                    first,
                    min(4, channels - first),
                    color_space,
                )
                for first in range(0, channels, 4)
            ),
            latent_features=8,
            latent_mips=(packed0, packed1),
            layers=tuple(layers),
            metadata={
                "compressor": "warp-nn",
                "training_steps": steps,
                "refinement_steps": refinement_steps,
                "latent_scale": latent_scale,
                "batch_size": batch_count,
                "weight_type": "fp8_e4m3",
            },
        )
    decoded = decode_image(asset, batch_size=batch_size)
    mse = float(np.mean((decoded - image) ** 2))
    psnr = float("inf") if mse == 0.0 else float(-10.0 * np.log10(mse))
    return CompressionResult(asset=asset, losses=tuple(losses), psnr=psnr)
