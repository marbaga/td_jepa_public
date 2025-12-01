# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import pickle

import numpy as np
from dm_control.rl.control import flatten_observation


def set_seed_everywhere(seed):
    import random

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def load_episode(fn):
    with fn.open("rb") as f:
        try:
            episode = pickle.load(f, allow_pickle=True)
        except Exception:
            episode = np.load(f, allow_pickle=True)
            episode = {k: episode[k] for k in episode.keys()}
        return episode


def check_consistency_of_episode(env, episode):
    physics = episode["physics"]
    actions = episode["action"]
    rewards = episode["reward"]
    observations = episode["observation"]

    env.reset()
    with env.physics.reset_context():
        env.physics.set_state(physics[0])
    obs = env.task.get_observation(env.physics)
    obs = flatten_observation(obs)["observations"]
    obs_err = [np.abs(obs - observations[0]).max()]
    phy_err = [np.abs(env.physics.get_state() - physics[0]).max()]
    rew_err = []

    for ii in range(observations.shape[0] - 1):
        action = actions[ii + 1]
        time_step = env.step(action)
        obs = time_step.observation["observations"]
        obs_err.append(np.abs(obs - observations[ii + 1]).max())
        phy_err.append(np.abs(env.physics.get_state() - physics[ii + 1]).max())
        rew_err.append(np.abs(time_step.reward - rewards[ii + 1]).max())
    passed = (np.max(obs_err) < 1e-4) and (np.max(phy_err) < 1e-4) and (np.max(rew_err) < 1e-4)
    return passed
