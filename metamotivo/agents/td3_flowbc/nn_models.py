# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from torch import nn

from ...base import BaseConfig
from ...nn_filters import IdentityInputFilterConfig, NNFilter
from ...nn_models import linear


class NoiseConditionedActorArchiConfig(BaseConfig):
    hidden_dim: int = 1024
    model: str = "simple"  # not used at the moment
    hidden_layers: int = 2
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, action_dim):
        return SimpleNoiseConditionedActor(obs_space, action_dim, cfg=self)


class SimpleNoiseConditionedActor(nn.Module):
    """An actor with a simple MLP architecture and relu non-linearities"""

    def __init__(self, obs_space, action_dim, cfg: NoiseConditionedActorArchiConfig) -> None:
        super().__init__()
        self.cfg: NoiseConditionedActorArchiConfig = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert len(filtered_space.shape) == 1, "obs_space must be 1D box"
        obs_dim = filtered_space.shape[0]

        seq = [linear(obs_dim + action_dim, cfg.hidden_dim), nn.ReLU()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs, noise):
        obs = self.input_filter(obs)
        embedding = torch.cat([obs, noise], dim=-1)
        actions = torch.tanh(self.policy(embedding))
        return actions


class SimpleVectorFieldArchiConfig(BaseConfig):
    # name: tp.Literal["vector_field"] = "vector_field"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, action_dim: int) -> "VectorField":
        return VectorField(obs_space, action_dim, self)


class VectorField(nn.Module):
    def __init__(self, obs_space, action_dim, cfg: SimpleVectorFieldArchiConfig) -> None:
        super().__init__()
        self.cfg: SimpleVectorFieldArchiConfig = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert len(filtered_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        # plus 1 is for time
        seq = [linear(obs_dim + action_dim + 1, cfg.hidden_dim), nn.GELU()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.net = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        obs = self.input_filter(obs)
        embedding = torch.cat([obs, action, t], dim=-1)
        return self.net(embedding)
