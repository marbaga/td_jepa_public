# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ..laplacian.agent import LaplacianAgent, LaplacianAgentConfig, LaplacianAgentTrainConfig
from ..sf_flowbc.agent import SFFlowBCAgentConfig, SFFlowBCAgentMixin, SFFlowBCAgentTrainConfig
from ..sf_flowbc.model import SFFlowBCModelConfig


class LaplacianFlowBCAgentTrainConfig(SFFlowBCAgentTrainConfig, LaplacianAgentTrainConfig):
    pass


class LaplacianFlowBCAgentConfig(SFFlowBCAgentConfig, LaplacianAgentConfig):
    name: tp.Literal["LaplacianFlowBCAgent"] = "LaplacianFlowBCAgent"
    model: SFFlowBCModelConfig
    train: LaplacianFlowBCAgentTrainConfig

    @property
    def object_class(self):
        return LaplacianFlowBCAgent


class LaplacianFlowBCAgent(SFFlowBCAgentMixin, LaplacianAgent):
    config_class = LaplacianFlowBCAgentConfig

    @property
    def optimizer_dict(self):
        return super(LaplacianAgent, self).optimizer_dict | super(LaplacianFlowBCAgent, self).optimizer_dict

    def setup_training(self) -> None:
        LaplacianAgent.setup_training(self)
        SFFlowBCAgentMixin.setup_training(self)
