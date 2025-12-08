# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from metamotivo.agents.byol.agent import BYOLAgentConfig
from metamotivo.agents.byol_flowbc.agent import BYOLFlowBCAgentConfig
from metamotivo.agents.fb.agent import FBAgentConfig
from metamotivo.agents.fb_flowbc.agent import FBFlowBCAgentConfig
from metamotivo.agents.hilp.agent import HilpAgentConfig
from metamotivo.agents.hilp_flowbc.agent import HilpFlowBCAgentConfig
from metamotivo.agents.icvf.agent import ICVFAgentConfig
from metamotivo.agents.icvf_flowbc.agent import ICVFFlowBCAgentConfig
from metamotivo.agents.laplacian.agent import LaplacianAgentConfig
from metamotivo.agents.laplacian_flowbc.agent import LaplacianFlowBCAgentConfig
from metamotivo.agents.rldp.agent import RLDPAgentConfig
from metamotivo.agents.rldp_flowbc.agent import RLDPFlowBCAgentConfig
from metamotivo.agents.td3.agent import TD3AgentConfig
from metamotivo.agents.td3_flowbc.agent import TD3FlowBCAgentConfig
from metamotivo.agents.td_jepa.agent import TDJEPAAgentConfig
from metamotivo.agents.td_jepa_flowbc.agent import TDJEPAFlowBCAgentConfig

Agent = (
    TDJEPAAgentConfig
    | TDJEPAFlowBCAgentConfig
    | FBAgentConfig
    | FBFlowBCAgentConfig
    | RLDPAgentConfig
    | RLDPFlowBCAgentConfig
    | BYOLAgentConfig
    | BYOLFlowBCAgentConfig
    | HilpAgentConfig
    | HilpFlowBCAgentConfig
    | LaplacianAgentConfig
    | LaplacianFlowBCAgentConfig
    | ICVFAgentConfig
    | ICVFFlowBCAgentConfig
    | TD3AgentConfig
    | TD3FlowBCAgentConfig
)
