# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import time
import typing as tp

import gymnasium
import mujoco
import numpy as np
from gymnasium.wrappers import TimeAwareObservation

from metamotivo.base import BaseConfig
from metamotivo.envs.dmc_tasks import dmc
from metamotivo.envs.dmc_tasks.dmc2gym import DmcGymWrapper

from .utils.wrappers import PixelWrapper

CAMERA_IDS = {
    "walker": 0,
    "cheetah": 0,
    "quadruped": 2,
    "pointmass": 0,
}


def create_dmc_env(
    num_envs=1,
    vectorization_mode: str = "async",
    wrappers=None,
    **env_kwargs,
):
    print(env_kwargs)
    import random

    mp_context = env_kwargs.pop("context", None)
    seed = env_kwargs.pop("seed", random.randint(0, 9999))

    def create_single_env(**kwargs):
        def trunk():
            env = dmc.make(f"{kwargs.get('domain')}_{kwargs.get('task')}", seed=kwargs.get("seed", 1))
            env = DmcGymWrapper(
                env,
                height=kwargs.get("render_height", 64),
                width=kwargs.get("render_width", 64),
                camera_id=kwargs.get("camera_id", 0),
                obs_type=kwargs.get("obs_type", "state"),
            )
            if wrappers is not None:
                for wrapper in wrappers:
                    env = wrapper(env)
            return env

        return trunk

    if num_envs > 1:
        envs = [create_single_env(**env_kwargs) for _ in range(num_envs)]
        if vectorization_mode == "sync":
            env = gymnasium.vector.SyncVectorEnv(envs)
        else:
            env = gymnasium.vector.AsyncVectorEnv(envs, context=mp_context)
    else:
        env = create_single_env(**env_kwargs)()

    env.reset(seed=seed)  # this is used to pass the seed to the environment
    return env, {}


class DMCEnvConfig(BaseConfig):
    name: tp.Literal["dmc"] = "dmc"

    domain: tp.Literal["walker", "cheetah", "quadruped", "pointmass"]
    # TODO this is hard to sanity check for, because dmc tries to create task from task string in multiple ways
    task: str

    seed: int = 0
    vectorization_mode: tp.Literal["sync", "async"] = "async"

    # observation type
    obs_type: tp.Literal["state", "pixels", "state_pixels"] = "state"

    # wrappers
    add_time: bool = False  # add time field to the dictionary

    # vision based parameter
    camera_id: int | None = None
    render_height: int = 64
    render_width: int = 64
    frame_stack: int = 1

    def build(self, num_envs: int = 1) -> tp.Tuple[gymnasium.Env, tp.Any]:
        assert num_envs >= 1
        wrappers = []
        if self.obs_type == "pixels":
            wrappers.append(lambda env: PixelWrapper(env, self.frame_stack))
        if self.add_time:
            wrappers.append(lambda env: TimeAwareObservation(env, flatten=False))
        return create_dmc_env(
            num_envs=num_envs,
            vectorization_mode=self.vectorization_mode,
            domain=self.domain,
            task=self.task,
            seed=self.seed,
            wrappers=wrappers,
            obs_type=self.obs_type,
            camera_id=self.camera_id or CAMERA_IDS[self.domain],
            render_height=self.render_height,
            render_width=self.render_width,
        )

    def get_relabel_fn(self, task):
        def fn(next_physics, actions):
            print("Relabeling train buffer.")
            env = dmc.make(f"{self.domain}_{task}")
            start = time.time()
            reward = np.zeros((len(actions), 1))
            for i in range(len(actions)):
                with env._physics.reset_context():
                    env._physics.set_state(next_physics[i])
                    env._physics.set_control(actions[i])
                mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
                reward[i] = env._task.get_reward(env._physics)
            print(f"Relabeling time: {time.time() - start}s")
            return reward

        return fn
