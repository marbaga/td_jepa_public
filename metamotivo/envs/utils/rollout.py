# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils._pytree import tree_map
from typing import Any, Dict, List, Tuple


def rollout(env: Any, agent: Any, num_episodes: int, ctx: torch.Tensor | None = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    observation, info = env.reset()
    returns, lengths, infos = [0.0], [0], [[info]]
    ctx = {} if ctx is None else {"z": ctx}
    while True:
        input_dict = {"obs": tree_map(lambda x: torch.tensor(x, device=agent.device, dtype=torch.float32)[None], observation), **ctx}
        action = agent.act(**input_dict).cpu().numpy()[0]
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        returns[-1] += reward
        lengths[-1] += 1
        infos[-1] += [info]
        if done:
            if len(returns) >= num_episodes:
                break
            observation, info = env.reset()
            returns.append(0.0)
            lengths.append(0)
            infos.append([info])
    return {"reward": returns, "length": lengths}, infos
