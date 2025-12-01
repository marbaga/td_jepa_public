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
        "num_train_steps": 3_000_000,
        "eval_every_steps": 250_000,
        "checkpoint_every_steps": 250_000,
        "agent": {
            "name": "SPTDAgent",
            "compile": True,
            "model": {
                "device": "cuda",
                "actor_std": 0.2,
                "obs_normalizer": {
                    "normalizers": {
                        "state": {
                            "name": "IdentityNormalizerConfig",
                        }
                    }
                },
                "archi": {
                    "predictor": {"name": "ForwardArchi", "hidden_dim": 1024, "hidden_layers": 1},
                    "actor": {"hidden_dim": 1024, "hidden_layers": 1, "name": "simple"},
                    "encoder": {
                        "name": "BackwardArchi",
                        "hidden_dim": 256,
                        "hidden_layers": 2,
                        "norm": True,
                        "input_filter": {"name": "DictInputFilterConfig", "key": "state"},
                    },
                    "z_dim": 50,
                    "norm_z": True,
                },
            },
            "train": {
                "batch_size": 1024,
                "discount": 0.98,
                "lr_predictor": 1e-4,
                "lr_encoder": 1e-4,
                "lr_actor": 1e-4,
                "ortho_coef": 0.1,
                "train_goal_ratio": 0.5,
                "encoder_target_tau": 0.001,
                "predictor_target_tau": 0.001,
                "sptd_pessimism_penalty": 0.0,
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
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_encoder": [1e-4, 1e-5], "ortho_coef": [0.01, 0.1, 1]},
        },
    }
    return conf


def sweep_cheetah():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["cheetah"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_encoder": [1e-4, 1e-5], "ortho_coef": [0.01, 0.1, 1]},
        },
    }
    return conf


def sweep_quadruped():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["quadruped"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_encoder": [1e-4, 1e-5], "ortho_coef": [0.01, 0.1, 1]},
        },
    }
    return conf


def sweep_pointmass():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["pointmass"],
        "env.task": ["reach_top_left"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_encoder": [1e-4, 1e-5], "ortho_coef": [0.01, 0.1, 1], "discount": [0.99], "lr_actor": [1e-6]},
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
    wandb_gname: str | None = "sptd"
    wandb_pname: str | None = "replearn_dmc_paper"
    # to run sweeps
    sweep_config: str | None = None


def main(args: LaunchArgs):
    # Get default slurm arguments and location to store results
    data_paths = get_default_data_paths_for_current_cluster()
    exca_infra_args = get_default_exca_infra_args_for_current_cluster()
    workdir_root = get_workdir_root(args.wandb_pname, args.wandb_gname)
    # Move exca folder next to the run outputs
    exca_infra_args["folder"] = str(workdir_root / "_exca")

    base_config = BASE_CFG.copy()
    base_config.update(
        {
            "data": {
                "name": "dmc",
                "dataset_root": data_paths["url_data_root_path"],
                "domain": "walker",
                "load_n_episodes": 5_000,
                "obs_type": "state",
            },
            "work_dir": str(workdir_root),
            "use_wandb": args.use_wandb,
            "wandb_ename": args.wandb_ename,
            "wandb_pname": args.wandb_pname,
            "wandb_gname": args.wandb_gname,
            "infra": exca_infra_args,
            "env": {"name": "dmc", "domain": "walker", "task": "walk"},
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
    # uv run -m scripts.replearn.launch_sptd_dmc --wandb_gname sptd_walker_v10 --sweep_config sweep_walker --use_wandb
    # uv run -m scripts.replearn.launch_sptd_dmc --wandb_gname sptd_cheetah_v10 --sweep_config sweep_cheetah --use_wandb
    # uv run -m scripts.replearn.launch_sptd_dmc --wandb_gname sptd_quadruped_v10 --sweep_config sweep_quadruped --use_wandb
    # uv run -m scripts.replearn.launch_sptd_dmc --wandb_gname sptd_pointmass_v10 --sweep_config sweep_pointmass --use_wandb
