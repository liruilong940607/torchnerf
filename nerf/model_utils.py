# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

# Lint as: python3
"""Helper functions/classes for model definition."""

import functools
from typing import Any, Callable
import math

import torch
import torch.nn as nn


def dense_layer(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    # The initialization matters!
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(
        self,
        net_depth: int = 8,  # The depth of the first part of MLP.
        net_width: int = 256,  # The width of the first part of MLP.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        net_activation: Callable[Ellipsis, Any] = nn.ReLU(),  # The activation function.
        skip_layer: int = 4,  # The layer to add skip layers to.
        num_rgb_channels: int = 3,  # The number of RGB channels.
        num_sigma_channels: int = 1,  # The number of sigma channels.
        input_dim: int = 63,  # The number of input tensor channels.
        condition_dim: int = 27,  # The number of conditional tensor channels.
    ):
        super(MLP, self).__init__() 
        self.net_depth = net_depth
        self.net_width = net_width
        self.net_depth_condition = net_depth_condition
        self.net_width_condition = net_width_condition
        self.net_activation = net_activation
        self.skip_layer = skip_layer
        self.num_rgb_channels = num_rgb_channels
        self.num_sigma_channels = num_sigma_channels
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        self.input_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.input_layers.append(
                dense_layer(in_features, self.net_width)
            )
            if i % self.skip_layer == 0 and i > 0:
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        self.sigma_layer = dense_layer(in_features, self.num_sigma_channels)

        if self.condition_dim > 0:
            self.bottleneck_layer = dense_layer(in_features, self.net_width)
            self.condition_layers = nn.ModuleList()
            in_features = self.net_width + self.condition_dim
            for i in range(self.net_depth_condition):
                self.condition_layers.append(
                    dense_layer(in_features, self.net_width_condition)
                )
                in_features = self.net_width_condition
        self.rgb_layer = dense_layer(in_features, self.num_rgb_channels)

    def forward(self, x, condition=None):
        """Evaluate the MLP.

        Args:
          x: torch.tensor(float32), [batch, num_samples, feature], points.
          condition: torch.tensor(float32), 
            [batch, feature] or [batch, num_samples, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
          raw_rgb: torch.tensor(float32), with a shape of
               [batch, num_samples, num_rgb_channels].
          raw_sigma: torch.tensor(float32), with a shape of
               [batch, num_samples, num_sigma_channels].
        """
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.view([-1, feature_dim])
        inputs = x
        for i in range(self.net_depth):
            x = self.input_layers[i](x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = torch.cat([x, inputs], dim=-1)
        raw_sigma = self.sigma_layer(x).view(
            [-1, num_samples, self.num_sigma_channels]
        )

        if condition is not None:
            # Output of the first part of MLP.
            bottleneck = self.bottleneck_layer(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            if len(condition.shape) == 2:
                condition = condition[:, None, :].repeat(1, num_samples, 1)
            # Collapse the [batch, num_samples, feature] tensor to
            # [batch * num_samples, feature] so that it can be fed into nn.Dense.
            condition = condition.view([-1, condition.shape[-1]])
            x = torch.cat([bottleneck, condition], dim=-1)
            # Here use 1 extra layer to align with the original nerf model.
            for i in range(self.net_depth_condition):
                x = self.condition_layers[i](x)
                x = self.net_activation(x)
        raw_rgb = self.rgb_layer(x).view(
            [-1, num_samples, self.num_rgb_channels]
        )
        return raw_rgb, raw_sigma


def cast_rays(z_vals, origins, directions):
    return (
        origins[Ellipsis, None, :]
        + z_vals[Ellipsis, None] * directions[Ellipsis, None, :]
    )


def sample_along_rays(
    origins, directions, num_samples, near, far, randomized, lindisp
):
    """Stratified sampling along the rays.

    Args:
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      num_samples: int.
      near: float, near clip.
      far: float, far clip.
      randomized: bool, use randomized stratified sampling.
      lindisp: bool, sampling linearly in disparity rather than depth.

    Returns:
      z_vals: torch.tensor, [batch_size, num_samples], sampled z values.
      points: torch.tensor, [batch_size, num_samples, 3], sampled points.
    """
    batch_size = origins.shape[0]

    t_vals = torch.linspace(0.0, 1.0, num_samples, device=origins.device)
    if lindisp:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        z_vals = near * (1.0 - t_vals) + far * t_vals
        
    if randomized:
        mids = 0.5 * (z_vals[Ellipsis, 1:] + z_vals[Ellipsis, :-1])
        upper = torch.cat([mids, z_vals[Ellipsis, -1:]], -1)
        lower = torch.cat([z_vals[Ellipsis, :1], mids], -1)
        t_rand = torch.rand([batch_size, num_samples], device=origins.device)
        z_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast z_vals to make the returned shape consistent.
        z_vals = z_vals.expand([batch_size, num_samples])
        
    coords = cast_rays(z_vals, origins, directions)
    return z_vals, coords


def posenc(x, min_deg, max_deg, legacy_posenc_order=False):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Args:
      x: torch.tensor, variables to be encoded. Note that x should be in [-pi, pi].
      min_deg: int, the minimum (inclusive) degree of the encoding.
      max_deg: int, the maximum (exclusive) degree of the encoding.
      legacy_posenc_order: bool, keep the same ordering as the original tf code.

    Returns:
      encoded: torch.tensor, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], 
                          dtype=x.dtype, device=x.device)
    if legacy_posenc_order:
        xb = x[Ellipsis, None, :] * scales[:, None]
        four_feat = torch.reshape(
            torch.sin(torch.stack([xb, xb + 0.5 * math.pi], -2)), list(x.shape[:-1]) + [-1]
        )
    else:
        xb = torch.reshape(
            (x[Ellipsis, None, :] * scales[:, None]), list(x.shape[:-1]) + [-1]
        )
        four_feat = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def volumetric_rendering(rgb, sigma, z_vals, dirs, white_bkgd):
    """Volumetric Rendering Function.

    Args:
      rgb: torch.tensor(float32), color, [batch_size, num_samples, 3]
      sigma: torch.tensor(float32), density, [batch_size, num_samples, 1].
      z_vals: torch.tensor(float32), [batch_size, num_samples].
      dirs: torch.tensor(float32), [batch_size, 3].
      white_bkgd: bool.

    Returns:
      comp_rgb: torch.tensor(float32), [batch_size, 3].
      disp: torch.tensor(float32), [batch_size].
      acc: torch.tensor(float32), [batch_size].
      weights: torch.tensor(float32), [batch_size, num_samples]
    """
    eps = 1e-10
    dists = torch.cat(
        [
            z_vals[Ellipsis, 1:] - z_vals[Ellipsis, :-1],
            torch.tensor(
                [1e10], dtype=z_vals.dtype, device=z_vals.device
            ).expand(z_vals[Ellipsis, :1].shape),
        ],
        -1,
    )
    dists = dists * torch.linalg.norm(dirs[Ellipsis, None, :], dim=-1)
    # Note that we're quietly turning sigma from [..., 0] to [...].
    alpha = 1.0 - torch.exp(-sigma[Ellipsis, 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[Ellipsis, :1]),
            torch.cumprod(1.0 - alpha[Ellipsis, :-1] + eps, dim=-1),
        ],
        dim=-1,
    )
    weights = alpha * accum_prod

    comp_rgb = (weights[Ellipsis, None] * rgb).sum(dim=-2)
    depth = (weights * z_vals).sum(dim=-1)
    acc = weights.sum(dim=-1)  # Alpha
    # Equivalent to (but slightly more efficient and stable than):
    #  disp = 1 / max(eps, where(acc > eps, depth / acc, 0))
    inv_eps = 1 / eps
    disp = (acc / depth).double()  # torch.where accepts <scaler, double tensor> 
    disp = torch.where((disp > 0) & (disp < inv_eps) & (acc > eps), disp, inv_eps)
    disp = disp.float()
    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[Ellipsis, None])
    return comp_rgb, disp, acc, weights


def piecewise_constant_pdf(bins, weights, num_samples, randomized):
    """Piecewise-Constant PDF sampling.

    Args:
      bins: torch.tensor(float32), [batch_size, num_bins + 1].
      weights: torch.tensor(float32), [batch_size, num_bins].
      num_samples: int, the number of samples.
      randomized: bool, use randomized samples.

    Returns:
      z_samples: torch.tensor(float32), [batch_size, num_samples].
    """
    # Pad each weight vector (only if necessary) to bring its sum to `eps`. This
    # avoids NaNs when the input is zeros or small, but has no effect otherwise.
    eps = 1e-5
    weight_sum = torch.sum(weights, dim=-1, keepdims=True)
    padding = torch.clamp(eps - weight_sum, min=0)
    weights = weights + padding / weights.shape[-1]  # avoid +=
    weight_sum = weight_sum + padding

    # Compute the PDF and CDF for each weight vector, while ensuring that the CDF
    # starts with exactly 0 and ends with exactly 1.
    pdf = weights / weight_sum
    cdf = torch.clamp(torch.cumsum(pdf[Ellipsis, :-1], dim=-1), max=1)
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], dtype=cdf.dtype, device=cdf.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], dtype=cdf.dtype, device=cdf.device),
        ],
        dim=-1,
    )

    # Draw uniform samples.
    if randomized:
        # Note that `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], 
                       dtype=cdf.dtype, device=cdf.device)
    else:
        # Match the behavior of torch.rand() by spanning [0, 1-eps].
        u = torch.linspace(0.0, 1.0 - torch.finfo().eps, num_samples, 
                           dtype=cdf.dtype, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])

    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[Ellipsis, None, :] >= cdf[Ellipsis, :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(torch.where(mask, x[Ellipsis, None], x[Ellipsis, :1, None]), dim=-2)[0]
        x1 = torch.min(torch.where(~mask, x[Ellipsis, None], x[Ellipsis, -1:, None]), dim=-2)[0]
        return x0, x1

    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    # `nan_to_num` exists in pytorch>=1.8.0
    t = torch.clamp(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)

    # Prevent gradient from backprop-ing through `samples`.
    return samples.detach()


def sample_pdf(
    bins, weights, origins, directions, z_vals, num_samples, randomized
):
    """Hierarchical sampling.

    Args:
      bins: torch.tensor(float32), [batch_size, num_bins + 1].
      weights: torch.tensor(float32), [batch_size, num_bins].
      origins: torch.tensor(float32), [batch_size, 3], ray origins.
      directions: torch.tensor(float32), [batch_size, 3], ray directions.
      z_vals: torch.tensor(float32), [batch_size, num_coarse_samples].
      num_samples: int, the number of samples.
      randomized: bool, use randomized samples.

    Returns:
      z_vals: torch.tensor(float32),
        [batch_size, num_coarse_samples + num_fine_samples].
      points: torch.tensor(float32),
        [batch_size, num_coarse_samples + num_fine_samples, 3].
    """
    z_samples = piecewise_constant_pdf(bins, weights, num_samples, randomized)
    # Compute united z_vals and sample points
    z_vals = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)[0]
    coords = cast_rays(z_vals, origins, directions)
    return z_vals, coords


def add_gaussian_noise(raw, noise_std, randomized):
    """Adds gaussian noise to `raw`, which can used to regularize it.

    Args:
      raw: torch.tensor(float32), arbitrary shape.
      noise_std: float, The standard deviation of the noise to be added.
      randomized: bool, add noise if randomized is True.

    Returns:
      raw + noise: torch.tensor(float32), with the same shape as `raw`.
    """
    if (noise_std is not None) and randomized:
        return raw + torch.randn(
            raw.shape, dtype=raw.dtype, device=raw.device) * noise_std
    else:
        return raw
