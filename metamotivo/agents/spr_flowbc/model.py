# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ..sf_flowbc.model import SFFlowBCModelArchiConfig, SFFlowBCModelConfig, SFFlowBCModelMixin
from ..spr.model import SPRModel, SPRModelArchiConfig, SPRModelConfig


class SPRFlowBCModelArchiConfig(SFFlowBCModelArchiConfig, SPRModelArchiConfig):
    pass


class SPRFlowBCModelConfig(SFFlowBCModelConfig, SPRModelConfig):
    name: tp.Literal["SPRFlowBCModel"] = "SPRFlowBCModel"
    archi: SPRFlowBCModelArchiConfig = SPRFlowBCModelArchiConfig()

    @property
    def object_class(self):
        return SPRFlowBCModel


class SPRFlowBCModel(SFFlowBCModelMixin, SPRModel):
    config_class = SPRFlowBCModelConfig

    def __init__(self, obs_space, action_dim, cfg: SPRModelConfig):
        SPRModel.__init__(self, obs_space, action_dim, cfg)
        SFFlowBCModelMixin.__init__(self, obs_space, action_dim, cfg)
