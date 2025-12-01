# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path

import torch

from metamotivo.base import BaseConfig
from metamotivo.buffers.parallel import ParallelBuffer
from metamotivo.buffers.transition import DictBuffer

# TODO: move these helper functions in a shared file
from metamotivo.data_loading.dmc import load_trajectories, load_transitions
from metamotivo.envs.ogbench import ALL_DOMAINS
from metamotivo.pytree_utils import tree_get_batch_size


class OGBenchDataConfig(BaseConfig):
    name: tp.Literal["ogbench"] = "ogbench"

    domain: tp.Literal[tuple(ALL_DOMAINS)]
    dataset_root: str
    load_n_episodes: int = 1_000
    obs_type: tp.Literal["state", "pixels"] = "state"
    buffer_type: tp.Literal["dict", "parallel"] = "dict"
    future: float = 0.99
    num_workers: int = 8  # for parallel buffer

    # TODO: the logic here is almost entirely shared with DMC, it can be unified
    def build(self, buffer_device, batch_size, frame_stack, relabel_fn=None) -> tp.Dict:
        domain = ("visual-" if self.obs_type == "pixels" else "") + self.domain
        path = Path(self.dataset_root) / f"{domain}/buffer"
        print(f"Loading data from: {path}")
        files = list(path.glob("*.npz"))
        num_episodes = min(self.load_n_episodes, len(files))

        buffer_type = self.buffer_type
        if self.obs_type == "pixels":
            print("Enforcing parallel buffer when learning from pixels.")
            buffer_type = "parallel"

        match buffer_type:
            case "dict":
                data = load_transitions(
                    files[:num_episodes],
                    self.obs_type,
                )
                replay_buffer = {"train": DictBuffer(capacity=tree_get_batch_size(data["observation"]), device=buffer_device)}
                replay_buffer["train"].extend(data)
                del data
                if relabel_fn is not None:
                    rewards = relabel_fn(
                        replay_buffer["train"].storage["next"]["physics"].cpu().numpy(),
                        replay_buffer["train"].storage["action"].cpu().numpy(),
                    )
                    replay_buffer["train"].storage["reward"] = torch.tensor(
                        rewards, dtype=torch.float32, device=replay_buffer["train"].device
                    )
            case "parallel":
                replay_buffer = {
                    "train": ParallelBuffer(
                        files[:num_episodes],
                        load_trajectories,
                        batch_size=batch_size,
                        frame_stack=frame_stack,
                        obs_type=self.obs_type,
                        relabel_fn=relabel_fn,
                        device=buffer_device,
                        future=self.future,
                        num_workers=self.num_workers,
                    )
                }
            case _:
                raise ValueError(f"Unsupported buffer type {self.buffer_type}")
        return replay_buffer
