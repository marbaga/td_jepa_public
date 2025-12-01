# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import tyro
from exca.confdict import ConfDict

from entry_points.train_offline import TrainConfig
from metamotivo.envs.ogbench import ALL_TASKS
from metamotivo.misc.launcher_utils import all_combinations_of_nested_dicts_for_sweep, launch_trials

BASE_CFG = ConfDict(
    {
        "num_train_steps": 1_000_000,
        "eval_every_steps": 250_000,
        "checkpoint_every_steps": 250_000,
        "agent": {
            "name": "SPTDFFFlowBCAgent",
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
                    "fw_predictor": {"name": "ForwardArchi", "hidden_dim": 512, "hidden_layers": 2},
                    "bw_predictor": {"name": "ForwardArchi", "hidden_dim": 512, "hidden_layers": 2},
                    "actor": {"name": "noise_conditioned_actor", "hidden_dim": 512, "hidden_layers": 2},
                    "actor_vf": {"hidden_layers": 4, "hidden_dim": 512},
                    "left_encoder": {"name": "BackwardArchi", "hidden_dim": 512, "hidden_layers": 0, "norm": True},
                    "right_encoder": {"name": "BackwardArchi", "hidden_dim": 512, "hidden_layers": 4, "norm": True},
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
                "actor_encode_obs": False,
            },
            "train": {
                "batch_size": 256,
                "discount": 0.99,
                "lr_predictor": 1e-4,
                "lr_left": 1e-4,
                "lr_right": 1e-4,
                "lr_actor": 1e-4,
                # orthonormalize left emebeddings
                "left_ortho_coef": 1.0,
                "right_ortho_coef": 1.0,
                "train_goal_ratio": 0.5,
                "predictor_target_tau": 0.005,
                "encoder_target_tau": 0.005,
                "sptdff_pessimism_penalty": 0.0,
                "actor_pessimism_penalty": 0.0,
                "scale_train_goals": True,
            },
        },
    }
)


def antmaze_v40():
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
        "agent.train.right_ortho_coef": [0.1, 1],
        "agent.train.lr_right": [1.0e-4, 1.0e-5],
    }
    return conf


def cube_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": [
            "cube-single-play-v0",
            "cube-double-play-v0",
            "scene-play-v0",
            "puzzle-3x3-play-v0",
        ],
        "agent.train.bc_coeff": [3.0],
        "agent.train.right_ortho_coef": [1, 10],
        "agent.train.lr_right": [1.0e-4, 1.0e-5],
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
    wandb_pname: str | None = "replearn_ogbench_pixel_paper"
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
                "name": "ogbench",
                "dataset_root": "/private/home/marbaga/motivo/datasets/processed_flat",
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
    # the versions here do not match (v10, v20) because cube was repeated with a lower ortho range (see Table 5)
    # uv run -m scripts.pixel.launch_sptdff_ogbench --wandb_gname sptdff_antmaze_v10 --sweep_config antmaze_v40 --use_wandb
    # uv run -m scripts.pixel.launch_sptdff_ogbench --wandb_gname sptdff_cube_v20 --sweep_config cube_v40 --use_wandb
