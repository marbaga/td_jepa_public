# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


def np_tree_map(func, tree):
    """
    Recursively apply `func` to all numpy array leaves in a nested dict structure.
    """
    return {k: np_tree_map(func, v) for k, v in tree.items()} if isinstance(tree, dict) else func(tree)


@dataclasses.dataclass(kw_only=True)
class Buffer(IterableDataset):
    episode_fns: Tuple  # list of filenames that need to be loaded across workers
    load_fn: callable  # function that loads the data from the filenames
    relabel_fn: Optional[callable]  # relabeling function
    num_workers: int = 0
    batch_size: int = 1
    frame_stack: int = 1
    frame_stack_key: str = "observation"
    obs_type: str = "pixels"
    output_key: Tuple[str] = ("observation", "physics", "action", "reward")
    output_key_next: Tuple[str] = ("observation", "physics", "terminated")
    end_key: str = "truncated"
    # if future < 1, additionally sample an observation in the future of each transition
    # this is done by geometric distribution with p=1-future
    future: float = 0.99

    def __post_init__(self) -> None:
        self.storage = None
        self.sampleable_idxs = None
        self.timesteps = None
        self.ready = False

    def _fetch(self):
        try:  # find the worker id
            worker_id = torch.utils.data.get_worker_info().id
        except:  # noqa: E722
            worker_id = 0
        # only load the correct share of all episodes
        episode_fns = [fn for i, fn in enumerate(self.episode_fns) if i % max(1, self.num_workers) == worker_id]
        self.storage = self.load_fn(episode_fns, self.obs_type)  # trajectory storage
        self.sampleable_idxs = np.where(~self.storage[self.end_key])[0]  # idxs that contain the first state in a valid transition
        self.timesteps = np.zeros(len(self.storage[self.end_key]), dtype=np.int32)  # temporal distance from first state in the trajectory
        self.lengths = np.zeros(len(self.storage[self.end_key]), dtype=np.int32)  # length of each trajectory
        prev = 0
        for i in range(1, len(self.timesteps)):
            if not self.storage[self.end_key][i - 1]:
                self.timesteps[i] = self.timesteps[i - 1] + 1
            else:
                self.timesteps[i] = 0
                self.lengths[prev:i] = self.timesteps[i - 1]
                prev = i
        self.lengths[prev:] = self.timesteps[-1]
        if self.relabel_fn is not None:
            self.storage["reward"] = np.zeros((len(self.storage["action"]), 1))
            self.storage["reward"][:-1] = self.relabel_fn(self.storage["physics"][1:], self.storage["action"][:-1])
            self.storage["reward"][self.storage[self.end_key]] = 0.0  # set rewards to zero for truncated episodes
        assert not any([isinstance(v, dict) for v in self.storage[self.frame_stack_key].values()]), (
            "Only flat observation dictionaries are allowed for the parallel buffer"
        )
        self.ready = True

    def _sample(self):
        if not self.ready:
            self._fetch()
        idxs = np.random.choice(self.sampleable_idxs, size=self.batch_size)
        timesteps = self.timesteps[idxs]
        offsets = [np.maximum(i, -timesteps) for i in range(-self.frame_stack + 1, 0)] + [0, 1]
        obs = np_tree_map(lambda v: [v[idxs + offset] for offset in offsets], self.storage[self.frame_stack_key])
        batch = {
            self.frame_stack_key: np_tree_map(lambda v: np.concatenate(v[:-1], 1) if self.frame_stack > 1 else v[0], obs),
            "next": {
                self.frame_stack_key: np_tree_map(lambda v: np.concatenate(v[1:], 1) if self.frame_stack > 1 else v[-1], obs),
            },
        }
        for k in self.output_key:
            if k != self.frame_stack_key and k in self.storage:
                batch[k] = np_tree_map(lambda x: x[idxs], self.storage[k])
        for k in self.output_key_next:
            if k != self.frame_stack_key and k in self.storage:
                batch["next"][k] = np_tree_map(lambda x: x[idxs + 1], self.storage[k])

        if self.future < 1:
            future_timesteps = timesteps + np.random.geometric(p=(1 - self.future), size=self.batch_size) - 1
            future_timesteps = np.clip(future_timesteps, a_min=None, a_max=self.lengths[idxs] - 1)
            future_idxs = idxs + future_timesteps - timesteps
            future_offsets = [np.maximum(-i, -future_timesteps) for i in range(1, self.frame_stack - 1)][::-1] + [0, 1]
            future_obs = {k: [v[future_idxs + offset] for offset in future_offsets] for k, v in self.storage[self.frame_stack_key].items()}
            batch["future_observation"] = {k: (np.concatenate(v, 1) if self.frame_stack > 1 else v[-1]) for k, v in future_obs.items()}

        return batch

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    # see https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _collate(batch_list, batch_size):
    batch = {}
    for k in batch_list[0].keys():
        if isinstance(batch_list[0][k], dict):
            batch[k] = _collate([b[k] for b in batch_list], batch_size)
        else:
            batch[k] = torch.cat([b[k] for b in batch_list], 0)[:batch_size]
    return batch


class ParallelBuffer:
    def __init__(
        self, episode_fns, load_fn, batch_size, frame_stack=1, obs_type="pixels", relabel_fn=None, device="cpu", num_workers=8, future=0.99
    ):
        self._batch_size = batch_size  # for compatibility with standard dataloaders, we use a fixed batch size
        iterable = Buffer(
            episode_fns=episode_fns,
            load_fn=load_fn,
            relabel_fn=relabel_fn,
            num_workers=num_workers,
            batch_size=batch_size,
            frame_stack=frame_stack,
            obs_type=obs_type,
            future=future,
        )
        self.loader = torch.utils.data.DataLoader(
            iterable, batch_size=None, num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_init_fn
        )
        self._replay_iter = iter(self.loader)
        self.device = device

    @torch.no_grad
    def sample(self, batch_size):
        if batch_size == self._batch_size:
            return next(self._replay_iter)
        # collate batches to the requested batch size
        # useful for regression of reward embeddings
        n_batches = 1 + batch_size // self._batch_size
        return _collate([next(self._replay_iter) for _ in range(n_batches)], batch_size)
