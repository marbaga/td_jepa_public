# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseConfig
from .nn_filters import IdentityInputFilterConfig, NNFilter


class DrQEncoderArchiConfig(BaseConfig):
    name: tp.Literal["drq"] = "drq"
    feature_dim: int | None = None  # if not None, linearly project the output to feature_dim
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space):
        return DrQEncoder(obs_space, self)


class DrQEncoder(nn.Module):
    """RGB encoder from the DrQ-v2 paper"""

    def __init__(self, obs_space, cfg: DrQEncoderArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 3, "filtered_space must have a 3D shape (image)"

        # courtesy of https://github.com/facebookresearch/drqv2/blob/main/drqv2.py
        self.trunk = nn.Sequential(
            nn.Conv2d(filtered_space.shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # TODO: is there a fundamental reason not to do this?
        with torch.no_grad():
            self.repr_dim = np.prod(self.trunk(torch.zeros(1, *filtered_space.shape)).shape)

        if self.cfg.feature_dim is not None:
            self.proj = nn.Sequential(nn.Linear(self.repr_dim, self.cfg.feature_dim), nn.LayerNorm(self.cfg.feature_dim), nn.Tanh())
            self.repr_dim = self.cfg.feature_dim
        else:
            self.proj = nn.Identity()
            print(
                "WARNING: using a DrQ encoder with feature_dim=None. This yields very large feature vectors that are fed as input to other networks"
            )

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self.input_filter(obs)
        return self.proj(self.trunk(obs))

    @property
    def output_space(self):
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.repr_dim,), dtype=np.float32)


class DreamerEncoderArchiConfig(BaseConfig):
    name: tp.Literal["dreamer"] = "dreamer"
    simnorm_dim: int = 8
    feature_dim: int | None = None  # if not None, linearly project the output to feature_dim
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space):
        return DreamerEncoder(obs_space, self)


class DreamerEncoder(nn.Module):
    """RGB encoder from Dreamer/TD-MPC"""

    def __init__(self, obs_space, cfg: DreamerEncoderArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 3, "filtered_space must have a 3D shape (image)"

        # courtesy of https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
        self.trunk = nn.Sequential(
            nn.Conv2d(filtered_space.shape[0], 32, 7, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
        )

        # TODO: is there a fundamental reason not to do this?
        with torch.no_grad():
            self.repr_dim = np.prod(self.trunk(torch.zeros(1, *filtered_space.shape)).shape)

        if self.cfg.feature_dim is not None:
            self.proj = nn.Sequential(nn.Linear(self.repr_dim, self.cfg.feature_dim), nn.LayerNorm(self.cfg.feature_dim), nn.Tanh())
            self.repr_dim = self.cfg.feature_dim
        else:
            self.proj = nn.Identity()

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self.input_filter(obs)
        # simnorm as per the original implementation
        x = obs.view(obs.shape[0], -1, self.cfg.simnorm_dim)
        x = F.softmax(x, dim=-1)
        x = x.view(x.shape[0], -1)
        return self.proj(x)

    @property
    def output_space(self):
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.repr_dim,), dtype=np.float32)


class ImpalaEncoderArchiConfig(BaseConfig):
    name: tp.Literal["impala"] = "impala"
    feature_dim: int | None = 512  # if not None, linearly project the output to feature_dim
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space):
        return ImpalaEncoder(obs_space, self)


class ResnetStack(nn.Module):
    """ResNet stack module."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.pre_block = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.pre_block(x)
        return self.block(x) + x


class ImpalaEncoder(nn.Module):
    """IMPALA RGB encoder, adapted from OGBench"""

    def __init__(self, obs_space, cfg: ImpalaEncoderArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 3, "filtered_space must have a 3D shape (image)"

        self.trunk = nn.Sequential(
            ResnetStack(filtered_space.shape[0], 16),
            ResnetStack(16, 32),
            ResnetStack(32, 32),
            nn.Flatten(),
            nn.ReLU(),
        )

        # TODO: is there a fundamental reason not to do this?
        with torch.no_grad():
            self.repr_dim = np.prod(self.trunk(torch.zeros(1, *filtered_space.shape)).shape)

        if self.cfg.feature_dim is not None:
            self.proj = nn.Sequential(nn.Linear(self.repr_dim, self.cfg.feature_dim), nn.GELU())
            self.repr_dim = self.cfg.feature_dim
        else:
            self.proj = nn.Identity()

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self.input_filter(obs)
        return self.proj(self.trunk(obs))

    @property
    def output_space(self):
        return gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.repr_dim,), dtype=np.float32)


class AugmentatorArchiConfig(BaseConfig):
    name: tp.Literal["random_shifts"] = "random_shifts"
    pad: int = 4
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space):
        return Augmentator(obs_space, self)


class Augmentator(nn.Module):
    """Image augmentations from DrQ-v2"""

    def __init__(self, obs_space, cfg: AugmentatorArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 3, "filtered_space must have a 3D shape (image)"

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        obs = self.input_filter(obs)
        n, _, h, w = obs.size()
        assert h == w, "Augmentator only supports square images"
        padding = tuple([self.cfg.pad] * 4)
        obs = F.pad(obs, padding, "replicate")
        eps = 1.0 / (h + 2 * self.cfg.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.cfg.pad, device=obs.device, dtype=obs.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.cfg.pad + 1, size=(n, 1, 1, 2), device=obs.device, dtype=obs.dtype)
        shift *= 2.0 / (h + 2 * self.cfg.pad)
        grid = base_grid + shift
        return F.grid_sample(obs, grid, padding_mode="zeros", align_corners=False)
