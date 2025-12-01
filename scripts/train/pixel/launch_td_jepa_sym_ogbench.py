# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib

os.environ["MUJOCO_GL"] = "egl"  # for headless rendering

import dataclasses

import tyro
from exca.confdict import ConfDict

from entry_points.train_offline import TrainConfig
from metamotivo.envs.ogbench import ALL_TASKS
from metamotivo.misc.launcher_utils import all_combinations_of_nested_dicts_for_sweep, launch_trials
import pathlib

BASE_CFG = ConfDict(
    {
        "num_train_steps": 1_000_000,
        "eval_every_steps": 250_000,
        "checkpoint_every_steps": 250_000,
        "agent": {
            "name": "TDJEPAFlowBCAgent",
            "compile": True,
            "model": {
                "device": "cuda",
                "obs_normalizer": {
                    "normalizers": {
                        "pixels": {
                            "name": "RGBNormalizerConfig",
                        }
                    }
                },
                "archi": {
                    "phi_predictor": {"name": "ForwardArchi", "hidden_dim": 512, "hidden_layers": 2},
                    "actor": {"name": "noise_conditioned_actor", "hidden_dim": 512, "hidden_layers": 2},
                    "actor_vf": {"hidden_layers": 4, "hidden_dim": 512},
                    "phi_mlp_encoder": {"name": "BackwardArchi", "hidden_dim": 512, "hidden_layers": 4, "norm": True},
                    "rgb_encoder": {
                        "name": "drq",
                        "feature_dim": 256,
                        "input_filter": {"name": "DictInputFilterConfig", "key": "pixels"},
                    },
                    "augmentator": {
                        "name": "random_shifts",
                        "pad": 2,
                        "input_filter": {"name": "DictInputFilterConfig", "key": "pixels"},
                    },
                    "phi_dim": 50,
                    "norm_z": True,
                },
                "actor_use_full_encoder": False,
                "symmetric": True,
            },
            "train": {
                "batch_size": 256,
                "discount": 0.99,
                "lr_predictor": 1e-4,
                "lr_phi": 1e-4,
                "lr_actor": 1e-4,
                "phi_ortho_coef": 1.0,
                "train_goal_ratio": 0.5,
                "encoder_target_tau": 0.005,
                "predictor_target_tau": 0.005,
                "predictor_pessimism_penalty": 0.0,
                "actor_pessimism_penalty": 0.0,
                "scale_train_goals": True,
            },
        },
    }
)


def sweep_antmaze():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": [
            "antmaze-medium-navigate-v0",
            "antmaze-large-navigate-v0",
            "antmaze-medium-stitch-v0",
            "antmaze-large-stitch-v0",
            "antmaze-medium-explore-v0",
        ],
        "agent.train.bc_coeff": [0.3],
        "agent.train.phi_ortho_coef": [0.1, 1],
        "agent.train.lr_phi": [1.0e-4, 1.0e-5],
    }
    return conf


def sweep_cube():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": [
            "cube-single-play-v0",
            "cube-double-play-v0",
            "scene-play-v0",
            "puzzle-3x3-play-v0",
        ],
        "agent.train.bc_coeff": [3.0],
        "agent.train.phi_ortho_coef": [1, 10],
        "agent.train.lr_phi": [1.0e-4, 1.0e-5],
    }
    return conf


@dataclasses.dataclass
class LaunchArgs:
    # Instead of launching the experiments, run the first sweep locally to test out the code
    local: bool = False
    # Print out the configs instead of running the experiments
    dry: bool = False
    # wandb config
    use_wandb: bool = False
    wandb_ename: str | None = "unicorns"
    wandb_gname: str | None = "td_jepa"
    wandb_pname: str | None = "tdjepa_main"
    # to run sweeps
    sweep_config: str | None = None


def main(args: LaunchArgs):
    data_path = "PATH TO DATASETS"
    workdir_root = pathlib.Path("test")
    # TODO remove exca?
    exca_infra_args = {}
    exca_infra_args["folder"] = str(workdir_root / "_exca")

    base_config = BASE_CFG.copy()
    base_config.update(
        {
            "data": {
                "name": "ogbench",
                "dataset_root": data_paths["ogbench_data_root_path_processed"],
                "domain": "cube-single-play-v0",
                "obs_type": "pixels",
                "buffer_type": "parallel",
            },
            "work_dir": str(workdir_root),
            "use_wandb": args.use_wandb,
            "wandb_ename": args.wandb_ename,
            "wandb_pname": args.wandb_pname,
            "wandb_gname": args.wandb_gname,
            "infra": exca_infra_args,
            "env": {
                "name": "ogbench",
                "obs_type": "pixels",
                "frame_stack": 3,
                "domain": "cube-single-play-v0",
                "task": "cube-single-play-singletask-task1-v0",
            },
        }
    )

    if args.sweep_config is None:
        sweep_params = {}
    else:
        if args.sweep_config in globals().keys():
            sweep_params = globals()[args.sweep_config]()
        else:
            raise RuntimeError("Unknown sweep configuration")

    base_config = TrainConfig(**base_config)
    trials = all_combinations_of_nested_dicts_for_sweep(sweep_params)
    for i, trial in enumerate(trials):
        trial["work_dir"] = f"{str(workdir_root)}/{i}"
        trial["data.domain"] = trial["env.domain"]
        trial["env"]["task"] = ALL_TASKS[trial["env.domain"]][0]
        extra_kwargs = {
            "env": {
                "name": "ogbench",
                "domain": trial["env.domain"],
                "task": ALL_TASKS[trial["env.domain"]][0],
                "obs_type": "pixels",
                "frame_stack": base_config["env"]["frame_stack"],
            },
            "tasks": ALL_TASKS[trial["env.domain"]],
            "num_envs": 1,
            "num_episodes": 10,
            "num_inference_samples": 10_000,
        }
        trial["evaluations"] = [
            extra_kwargs | {"name": "ogbench_reward_eval", "name_in_logs": "reward_shift", "shift_reward": 1},
        ]
    launch_trials(base_config, trials, args.local, args.dry)


if __name__ == "__main__":
    args = tyro.cli(LaunchArgs)
    main(args)
    # uv run -m scripts.replearn.pixel.launch_td_jepa_sym_ogbench --wandb_gname td_jepa_antmaze --sweep_config sweep_antmaze --use_wandb
    # uv run -m scripts.replearn.pixel.launch_td_jepa_sym_ogbench --wandb_gname td_jepa_cube --sweep_config sweep_cube --use_wandb
