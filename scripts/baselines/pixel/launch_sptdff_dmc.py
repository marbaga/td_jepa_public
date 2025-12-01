# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import tyro
from exca.confdict import ConfDict

from entry_points.train_offline import TrainConfig
from metamotivo.envs.dmc_tasks import ALL_TASKS
from metamotivo.misc.launcher_utils import all_combinations_of_nested_dicts_for_sweep, launch_trials

BASE_CFG = ConfDict(
    {
        "num_train_steps": 2_000_000,
        "eval_every_steps": 250_000,
        "checkpoint_every_steps": 250_000,
        "agent": {
            "name": "SPTDFFAgent",
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
                "actor_std": 0.2,
                "archi": {
                    "fw_predictor": {"name": "ForwardArchi", "hidden_dim": 1024, "hidden_layers": 1},
                    "bw_predictor": {"name": "ForwardArchi", "hidden_dim": 1024, "hidden_layers": 1},
                    "actor": {"hidden_dim": 1024, "hidden_layers": 1, "name": "simple"},
                    "left_encoder": {"name": "BackwardArchi", "hidden_dim": 256, "hidden_layers": 0, "norm": True},
                    "right_encoder": {"name": "BackwardArchi", "hidden_dim": 256, "hidden_layers": 2, "norm": True},
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
                    "L_dim": 256,
                    "R_dim": 50,
                    "norm_z": True,
                },
            },
            "train": {
                "batch_size": 512,
                "discount": 0.98,
                "lr_predictor": 1e-4,
                "lr_left": 1e-4,
                "lr_right": 1e-4,
                "lr_actor": 1e-4,
                "left_ortho_coef": 0.1,
                "right_ortho_coef": 0.1,
                "train_goal_ratio": 0.5,
                "predictor_target_tau": 0.001,
                "encoder_target_tau": 0.001,
                "sptdff_pessimism_penalty": 0.0,
                "actor_pessimism_penalty": 0.0,
            },
        },
    }
)


def sweep_walker():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["walker"],
        "agent": {
            "train": {"lr_right": [1e-4, 1e-5], "right_ortho_coef": [0.01, 0.1, 1]},
        },
    }
    return conf


def sweep_cheetah():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["cheetah"],
        "agent": {
            "train": {"lr_right": [1e-4, 1e-5], "right_ortho_coef": [0.01, 0.1, 1]},
        },
    }
    return conf


def sweep_quadruped():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["quadruped"],
        "agent": {
            "train": {"lr_right": [1e-4, 1e-5], "right_ortho_coef": [0.01, 0.1, 1]},
        },
    }
    return conf


def sweep_pointmass():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["pointmass"],
        "env.task": ["reach_top_left"],
        "agent": {
            "train": {"lr_right": [1e-4, 1e-5], "right_ortho_coef": [0.01, 0.1, 1], "discount": [0.99]},
        },
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
    wandb_gname: str | None = "sptdff"
    wandb_pname: str | None = "replearn_dmc_pixel_paper"
    # to run sweeps
    sweep_config: str | None = None


def main(args: LaunchArgs):
    # Get default slurm arguments and location to store results
    exca_infra_args = get_default_exca_infra_args_for_current_cluster()
    workdir_root = get_workdir_root(args.wandb_pname, args.wandb_gname)
    # Move exca folder next to the run outputs
    exca_infra_args["folder"] = str(workdir_root / "_exca")
    # make sure we have enough resources or the parallel buffer may be slow
    exca_infra_args["mem_gb"] = max(int(exca_infra_args["mem_gb"]), 100)
    exca_infra_args["cpus_per_task"] = max(int(exca_infra_args["cpus_per_task"]), 16)

    base_config = BASE_CFG.copy()
    base_config.update(
        {
            "data": {
                "name": "dmc",
                "dataset_root": "/private/home/marbaga/motivo/datasets/processed_flat",
                "domain": "walker",
                "load_n_episodes": 5_000,
                "obs_type": "pixels",
                "buffer_type": "parallel",
            },
            "work_dir": str(workdir_root),
            "use_wandb": args.use_wandb,
            "wandb_ename": args.wandb_ename,
            "wandb_pname": args.wandb_pname,
            "wandb_gname": args.wandb_gname,
            "infra": exca_infra_args,
            "env": {"name": "dmc", "domain": "walker", "task": "walk", "obs_type": "pixels", "frame_stack": 3},
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
        trial["evaluations"] = [
            {
                "name": "dmc_reward_eval",
                "env": {
                    "name": "dmc",
                    "domain": trial["env.domain"],
                    "task": ALL_TASKS[trial["env.domain"]][0],
                    "obs_type": "pixels",
                    "frame_stack": base_config["env"]["frame_stack"],
                },
                "tasks": ALL_TASKS[trial["env.domain"]],
                "num_envs": 1,
                "num_episodes": 20,
                "num_inference_samples": 10_000,
            },
        ]
    launch_trials(base_config, trials, args.local, args.dry)


if __name__ == "__main__":
    args = tyro.cli(LaunchArgs)
    main(args)
    # uv run -m scripts.pixel.launch_sptdff_dmc --wandb_gname sptdff_walker_v10 --sweep_config sweep_walker --use_wandb
    # uv run -m scripts.pixel.launch_sptdff_dmc --wandb_gname sptdff_cheetah_v10 --sweep_config sweep_cheetah --use_wandb
    # uv run -m scripts.pixel.launch_sptdff_dmc --wandb_gname sptdff_quadruped_v10 --sweep_config sweep_quadruped --use_wandb
    # uv run -m scripts.pixel.launch_sptdff_dmc --wandb_gname sptdff_pointmass_v10 --sweep_config sweep_pointmass --use_wandb
