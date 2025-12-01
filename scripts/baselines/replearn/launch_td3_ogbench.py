# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import tyro
from exca.confdict import ConfDict

from entry_points.train_offline import TrainConfig
from metamotivo.misc.launcher_utils import (
    all_combinations_of_nested_dicts_for_sweep,
    launch_trials,
)

BASE_CFG = ConfDict(
    {
        "relabel_dataset": True,
        "num_train_steps": 1_000_000,
        "eval_every_steps": 250_000,
        "checkpoint_every_steps": 250_000,
        "agent": {
            "name": "TD3Agent",
            "compile": True,
            "model": {
                "device": "cuda",
                "obs_normalizer": {
                    "normalizers": {
                        "name": "IdentityNormalizerConfig",
                    }
                },
                "archi": {
                    "critic": {
                        "hidden_dim": 512,
                        "hidden_layers": 4,
                        "layer_norm": True,
                        "input_filter": {"name": "DictInputFilterConfig", "key": "state"},
                    },
                    "actor": {
                        "hidden_dim": 512,
                        "hidden_layers": 4,
                        "input_filter": {"name": "DictInputFilterConfig", "key": "state"},
                    },
                },
            },
            "train": {
                "batch_size": 256,
                "discount": 0.99,
                "lr": 3.0e-4,
                "critic_target_tau": 0.005,
                "pessimism_penalty": 0.0,
            },
        },
    }
)


def cube_single_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["cube-single-play-v0"],
        "env.task": [f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [1.0, 3.0, 10.0],
    }
    return conf


def flowq_cube_single_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["cube-single-play-v0"],
        "env.task": [f"cube-single-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [1.0, 3.0, 10.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
        },
    }
    return conf


def cube_double_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["cube-double-play-v0"],
        "env.task": [f"cube-double-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [1.0, 3.0, 10.0],
    }
    return conf


def flowq_cube_double_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["cube-double-play-v0"],
        "env.task": [f"cube-double-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [1.0, 3.0, 10.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
        },
    }
    return conf


def scene_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["scene-play-v0"],
        "env.task": [f"scene-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [1.0, 3.0, 10.0],
    }
    return conf


def flowq_scene_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["scene-play-v0"],
        "env.task": [f"scene-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [1.0, 3.0, 10.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
        },
    }
    return conf


def puzzle_3x3_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["puzzle-3x3-play-v0"],
        "env.task": [f"puzzle-3x3-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [1.0, 3.0, 10.0],
    }
    return conf


def flowq_puzzle_3x3_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["puzzle-3x3-play-v0"],
        "env.task": [f"puzzle-3x3-play-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [1.0, 3.0, 10.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
        },
    }
    return conf


def antmaze_medium_navigate_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-medium-navigate-v0"],
        "env.task": [f"antmaze-medium-navigate-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [0.1, 0.3, 1.0],
    }
    return conf


def flowq_antmaze_medium_navigate_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-medium-navigate-v0"],
        "env.task": [f"antmaze-medium-navigate-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.1, 0.3, 1.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
        },
    }
    return conf


def antmaze_large_navigate_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-large-navigate-v0"],
        "env.task": [f"antmaze-large-navigate-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [0.1, 0.3, 1.0],
    }
    return conf


def flowq_antmaze_large_navigate_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-large-navigate-v0"],
        "env.task": [f"antmaze-large-navigate-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.1, 0.3, 1.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
        },
    }
    return conf


def antmaze_medium_stitch_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-medium-stitch-v0"],
        "env.task": [f"antmaze-medium-stitch-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [0.1, 0.3, 1.0],
    }
    return conf


def flowq_antmaze_medium_stitch_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-medium-stitch-v0"],
        "env.task": [f"antmaze-medium-stitch-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.1, 0.3, 1.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
        },
    }
    return conf


def antmaze_large_stitch_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-large-stitch-v0"],
        "env.task": [f"antmaze-large-stitch-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [0.1, 0.3, 1.0],
    }
    return conf


def flowq_antmaze_large_stitch_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-large-stitch-v0"],
        "env.task": [f"antmaze-large-stitch-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.1, 0.3, 1.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
        },
    }
    return conf


def antmaze_medium_explore_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-medium-explore-v0"],
        "env.task": [f"antmaze-medium-explore-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [0.1, 0.3, 1.0],
    }
    return conf


def flowq_antmaze_medium_explore_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-medium-explore-v0"],
        "env.task": [f"antmaze-medium-explore-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.1, 0.3, 1.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
        },
    }
    return conf


def antmaze_large_explore_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-large-explore-v0"],
        "env.task": [f"antmaze-large-explore-singletask-task{i}-v0" for i in range(1, 6)],
        "agent.train.bc_coeff": [0.1, 0.3, 1.0],
    }
    return conf


def flowq_antmaze_large_explore_v40():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729, 2226, 1744, 7742, 4501, 6341],
        "env.domain": ["antmaze-large-explore-v0"],
        "env.task": [f"antmaze-large-explore-singletask-task{i}-v0" for i in range(1, 6)],
        "agent": {
            "name": ["TD3FlowBCAgent"],
            "train.bc_coeff": [0.1, 0.3, 1.0],
            "model.archi.actor_vf": {
                "hidden_layers": [4],
                "hidden_dim": [512],
                "input_filter": {"name": ["DictInputFilterConfig"], "key": ["state"]},
            },
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
    wandb_gname: str | None = "td3"
    wandb_pname: str | None = "replearn_ogbench_paper"
    # to run sweeps
    sweep_config: str | None = None


def main(args: LaunchArgs):
    # Get default slurm arguments and location to store results
    exca_infra_args = get_default_exca_infra_args_for_current_cluster()
    workdir_root = get_workdir_root(args.wandb_pname, args.wandb_gname)
    # Move exca folder next to the run outputs
    exca_infra_args["folder"] = str(workdir_root / "_exca")

    base_config = BASE_CFG.copy()
    base_config.update(
        {
            "data": {
                "name": "ogbench",
                "dataset_root": "/private/home/marbaga/motivo/datasets/processed_flat",
                "domain": "cube-single-play-v0",
                "obs_type": "state",
            },
            "relabel_dataset": True,
            "work_dir": str(workdir_root),
            "use_wandb": args.use_wandb,
            "wandb_ename": args.wandb_ename,
            "wandb_pname": args.wandb_pname,
            "wandb_gname": args.wandb_gname,
            "infra": exca_infra_args,
            "env": {
                "name": "ogbench",
                "obs_type": "state",
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
        trial["data.domain"] = trial["env"]["domain"]
        trial["evaluations"] = [
            {
                "name": "ogbench_reward_eval",
                "env": {
                    "name": "ogbench",
                    "domain": trial["env"]["domain"],
                    "task": trial["env"]["task"],
                    "obs_type": "state",
                    "frame_stack": base_config["env"]["frame_stack"],
                },
                "tasks": [trial["env"]["task"]],
                "num_envs": 1,
                "num_episodes": 10,
                "num_inference_samples": 10_000,
            },
        ]
    launch_trials(base_config, trials, args.local, args.dry, use_current_python_env=True)  # YOLO


if __name__ == "__main__":
    args = tyro.cli(LaunchArgs)
    main(args)
    # uv run -m scripts.replearn.launch_td3_ogbench --wandb_gname td3_cube_single_v0 --sweep_config cube_single_v30 --use_wandb
