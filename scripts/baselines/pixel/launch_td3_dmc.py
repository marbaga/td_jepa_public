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
        "agent": {
            "name": "TD3Agent",
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
                    "critic": {"hidden_dim": 1024, "hidden_layers": 2},
                    "actor": {"hidden_dim": 1024, "hidden_layers": 2},
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
                },
            },
            "train": {
                "batch_size": 512,
                "discount": 0.98,
                "lr": 1e-4,
                "critic_target_tau": 0.001,
                "pessimism_penalty": 0,
            },
        },
    }
)


def sweep_walker():
    conf = {"seed": [3917, 3502, 8948, 9460, 4729], "env.domain": ["walker"], "env.task": ALL_TASKS["walker"]}
    return conf


def sweep_cheetah():
    conf = {"seed": [3917, 3502, 8948, 9460, 4729], "env.domain": ["cheetah"], "env.task": ALL_TASKS["cheetah"]}
    return conf


def sweep_quadruped():
    conf = {"seed": [3917, 3502, 8948, 9460, 4729], "env.domain": ["quadruped"], "env.task": ALL_TASKS["quadruped"]}
    return conf


def sweep_pointmass():
    conf = {"seed": [3917, 3502, 8948, 9460, 4729], "env.domain": ["pointmass"], "env.task": ALL_TASKS["pointmass"]}
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
            "relabel_dataset": True,
            "work_dir": str(workdir_root),
            "use_wandb": args.use_wandb,
            "wandb_ename": args.wandb_ename,
            "wandb_pname": args.wandb_pname,
            "wandb_gname": args.wandb_gname,
            "infra": exca_infra_args,
            "env": {"name": "dmc", "domain": "walker", "task": "walk", "obs_type": "pixels", "frame_stack": 3},
            "eval_every_steps": 250_000,
            "checkpoint_every_steps": 250_000,
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
                    "domain": trial["env"]["domain"],
                    "task": trial["env"]["task"],
                    "obs_type": "pixels",
                    "frame_stack": base_config["env"]["frame_stack"],
                },
                "tasks": [trial["env"]["task"]],
                "num_envs": 1,
                "num_episodes": 20,
                "num_inference_samples": 10_000,
            },
        ]
    launch_trials(base_config, trials, args.local, args.dry)


if __name__ == "__main__":
    args = tyro.cli(LaunchArgs)
    main(args)
    # uv run -m scripts.pixel.launch_sptd_dmc --wandb_gname td3_walker_v0 --sweep_config sweep_walker --use_wandb
    # uv run -m scripts.pixel.launch_sptd_dmc --wandb_gname td3_cheetah_v0 --sweep_config sweep_cheetah --use_wandb
    # uv run -m scripts.pixel.launch_sptd_dmc --wandb_gname td3_quadruped_v0 --sweep_config sweep_quadruped --use_wandb
    # uv run -m scripts.pixel.launch_sptd_dmc --wandb_gname td3_pointmass_v0 --sweep_config sweep_pointmass --use_wandb
