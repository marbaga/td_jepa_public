# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ..sf_flowbc.agent import SFFlowBCAgentConfig, SFFlowBCAgentMixin, SFFlowBCAgentTrainConfig
from ..spr.agent import SPRAgent, SPRAgentConfig, SPRAgentTrainConfig
from .model import SPRFlowBCModelConfig


class SPRFlowBCAgentTrainConfig(SFFlowBCAgentTrainConfig, SPRAgentTrainConfig):
    pass


class SPRFlowBCAgentConfig(SFFlowBCAgentConfig, SPRAgentConfig):
    name: tp.Literal["SPRFlowBCAgent"] = "SPRFlowBCAgent"
    model: SPRFlowBCModelConfig
    train: SPRFlowBCAgentTrainConfig

    @property
    def object_class(self):
        return SPRFlowBCAgent


class SPRFlowBCAgent(SFFlowBCAgentMixin, SPRAgent):
    config_class = SPRFlowBCAgentConfig

    @property
    def optimizer_dict(self):
        return super(SPRAgent, self).optimizer_dict | super(SPRFlowBCAgent, self).optimizer_dict

    def setup_training(self) -> None:
        SPRAgent.setup_training(self)
        SFFlowBCAgentMixin.setup_training(self)
