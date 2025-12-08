# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import torch
from torch import nn

from ...nn_models import BaseConfig, linear, simple_embedding


class NoiseConditionedActorArchiConfig(BaseConfig):
    name: tp.Literal["noise_conditioned_actor"] = "noise_conditioned_actor"
    model: tp.Literal["simple"] = "simple"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2

    def build(self, obs_space, z_dim: int, action_dim: int) -> "NoiseConditionedActor":
        return NoiseConditionedActor(obs_space, z_dim, action_dim, self)


class NoiseConditionedActor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: NoiseConditionedActorArchiConfig) -> None:
        super().__init__()

        assert len(obs_space.shape) == 1, "obs_space must have a 1D shape"
        obs_dim = obs_space.shape[0]
        self.cfg: NoiseConditionedActorArchiConfig = cfg
        self.embed_z = simple_embedding(obs_dim + z_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = simple_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = []
        for _ in range(cfg.hidden_layers):
            seq += [linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [linear(cfg.hidden_dim, action_dim)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        z_embedding = self.embed_z(torch.cat([obs, z, noise], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(torch.cat([obs, noise], dim=-1))  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        actions = torch.tanh(self.policy(embedding))
        return actions
