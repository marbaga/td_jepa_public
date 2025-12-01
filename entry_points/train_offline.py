# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OMP_NUM_THREADS"] = "1"

import torch

torch.set_float32_matmul_precision("high")
torch._inductor.config.autotune_local_cache = False

import json
import time
import typing as tp
from pathlib import Path
from typing import Dict, List

import exca as xk
import gymnasium
import numpy as np
import pydantic
import torch
import tyro
import wandb

from metamotivo.base import BaseConfig
from metamotivo.data_loading.dmc import DMCDataConfig
from metamotivo.data_loading.ogbench import OGBenchDataConfig
from metamotivo.envs.dmc import DMCEnvConfig
from metamotivo.envs.ogbench import OGBenchEnvConfig
from metamotivo.evaluations.dmc import DMCRewardEvalConfig
from metamotivo.evaluations.ogbench import OGBenchRewardEvalConfig
from metamotivo.misc.loggers import CSVLogger
from metamotivo.agents.td_jepa.agent import TDJEPAAgentConfig
from metamotivo.agents.td_jepa_flowbc.agent import TDJEPAFlowBCAgentConfig
from metamotivo.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere

TRAIN_LOG_FILENAME = "train_log.txt"

CHECKPOINT_DIR_NAME = "checkpoint"


Agent = (
    TDJEPAAgentConfig
    | TDJEPAFlowBCAgentConfig
)
Env = DMCEnvConfig | OGBenchEnvConfig
DataLoading = DMCDataConfig | OGBenchDataConfig

# Stackoverflow #70914419
Evaluation = tp.Annotated[
    tp.Union[DMCRewardEvalConfig, OGBenchRewardEvalConfig],
    pydantic.Field(discriminator="name"),
]


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")

    env: Env = pydantic.Field(discriminator="name")
    data: DataLoading = pydantic.Field(discriminator="name")
    relabel_dataset: bool = False

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("train_dmc"))

    seed: int = 0
    log_every_updates: int = 10_000
    num_train_steps: int = 3_000_000
    checkpoint_every_steps: int = 100_000

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None

    # misc
    buffer_device: str | None = None  # if None, use the agent's device

    # eval
    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])

    eval_every_steps: int = 100_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    # exca
    infra: xk.TaskInfra = xk.TaskInfra(version="1")

    def model_post_init(self, context):
        if self.relabel_dataset:
            if not isinstance(self.env, (DMCEnvConfig, OGBenchEnvConfig)):
                raise ValueError("Relabeling is only supported for DMC and OGBench environments")

    def build(self):
        """In case of cluster run, use exca and process instead of explivit build"""
        return Workspace(self)

    @infra.apply
    def process(self):
        ws = self.build()
        ws.train()


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0
    if checkpoint_dir.exists():
        # read train status
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = train_status["time"]

        print(f"Loading the agent at time {checkpoint_time}")
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_time


# TODO this can be unified with train_humenv
def init_wandb(cfg: TrainConfig):
    exp_name = "dmc-offline"
    wandb_name = exp_name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=wandb_name, config=wandb_config, dir="./_wandb")


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

        # NOTE we are assuming num_envs returns unvectorized environments
        sample_env, _ = cfg.env.build(num_envs=1)
        self.obs_space = sample_env.observation_space
        assert isinstance(self.obs_space, (gymnasium.spaces.Box, gymnasium.spaces.Dict)), (
            "Only Box and Dict observation spaces are supported"
        )

        self.action_space = sample_env.action_space
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_time = create_agent_or_load_checkpoint(
            self.work_dir,
            self.cfg,
            agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim),
        )
        self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        elif isinstance(self.cfg.evaluations, dict):
            self.evaluations = {name: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0
        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name, eval_cfg in self.evaluations.items()}

        if self.cfg.use_wandb:
            init_wandb(self.cfg)

        with (self.work_dir / "config.json").open("w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        buffer_device = self.agent.device if self.cfg.buffer_device is None else self.cfg.buffer_device
        relabel_fn = self.cfg.env.get_relabel_fn(self.cfg.env.task) if self.cfg.relabel_dataset else None
        replay_buffer = self.cfg.data.build(buffer_device, self.cfg.agent.train.batch_size, self.cfg.env.frame_stack, relabel_fn)
        print(replay_buffer["train"])

        if hasattr(self.agent, "setup_normalizer_from_data"):
            print("Preparing normalizer from data")
            assert self.cfg.data.buffer_type == "dict", (
                f"setup_normalizer_from_data not supported with buffer type {self.cfg.data.buffer_type}"
            )
            self.agent.setup_normalizer_from_data(replay_buffer["train"].storage)

        total_metrics = None
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)

        for t in range(self._checkpoint_time, int(self.cfg.num_train_steps) + 1):
            if (t != self._checkpoint_time) and checkpoint_time_checker.check(t):
                checkpoint_time_checker.update_last_step(t)
                self.save(t, replay_buffer)

            if self.evaluate and eval_time_checker.check(t):
                eval_time_checker.update_last_step(t)
                self.eval(t, replay_buffer=replay_buffer)

            metrics = self.agent.update(replay_buffer, t)

            # we need to copy tensors returned by a cudagraph module
            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if log_time_checker.check(t):
                log_time_checker.update_last_step(t)
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / (1 if t == 0 else self.cfg.log_every_updates)
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
        return

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}

        self.agent._model.train(False)

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations:
            evaluation = self.evaluations[evaluation_name]
            logger = self.eval_loggers[evaluation_name]

            evaluation_metrics, wandb_dict = evaluation.run(
                timestep=t,
                agent_or_model=self.agent,
                replay_buffer=replay_buffer,
                logger=logger,
            )
            # For wandb dict, put it on wandb
            if self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        self.agent._model.train()

        return evaluation_results

    def save(self, time: int, replay_buffer: Dict[str, tp.Any]) -> None:
        print(f"Checkpointing at time {time}")
        self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))
        with (self.work_dir / CHECKPOINT_DIR_NAME / "train_status.json").open("w+") as f:
            json.dump({"time": time}, f, indent=4)


if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    workspace = tyro.cli(Workspace)
    workspace.train()
