# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import Literal

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
            "name": "",  # to be updated later
            "compile": True,
            "model": {
                "device": "cuda",
                "obs_normalizer": {
                    "normalizers": {
                        "state": {
                            "name": "IdentityNormalizerConfig",
                        }
                    }
                },
                "actor_std": 0.2,
                "center_features": False,
                "archi": {
                    "successor_features": {
                        "name": "ForwardArchi",
                        "hidden_dim": 1024,
                        "hidden_layers": 1,
                    },
                    "actor": {
                        "hidden_dim": 1024,
                        "hidden_layers": 1,
                        "name": "simple",
                    },
                    "features": {
                        "name": "BackwardArchi",
                        "hidden_dim": 256,
                        "hidden_layers": 2,
                        "norm": True,
                        "input_filter": {"name": "DictInputFilterConfig", "key": "state"},
                    },
                    "left_encoder": {
                        "name": "BackwardArchi",
                        "hidden_dim": 256,
                        "hidden_layers": 0,
                        "norm": True,
                        "input_filter": {"name": "DictInputFilterConfig", "key": "state"},
                    },
                    "L_dim": 256,
                    "z_dim": 50,
                    "norm_z": True,
                },
            },
            "train": {
                "batch_size": 1024,
                "discount": 0.98,
                "lr_sf": 1e-4,
                "lr_features": 1e-4,
                "lr_actor": 1e-4,
                "train_goal_ratio": 0.5,
                "q_loss": False,
                "sf_target_tau": 0.001,
                "features_target_tau": 0.001,
                "sf_pessimism_penalty": 0,
                "actor_pessimism_penalty": 0,
            },
        },
    }
)

# extra params for the specific instance of SF agent
# must be in flat format
BASE_AGENT_CFG = {
    "sf": {"agent.name": "SFAgent"},
    "laplacian": {"agent.name": "LaplacianAgent"},
    "hilp": {
        "agent.name": "HilpAgent",
        "agent.train.expectile": 0.5,
        "agent.train.prob_random_goal": 0.375,
        "agent.model.archi.features.norm": False,
        "agent.model.center_features": True,
        "data.future": 0.99,
    },
    "spr": {
        "agent.name": "SPRAgent",
        "agent.train.lr_predictor": 1e-4,
        "agent.train.ortho_coef": 0.1,
        "agent.train.multi_step": False,
        "agent.model.archi.predictor.hidden_dim": 1024,
        "agent.model.archi.predictor.hidden_layers": 1,
    },
    "spr-multi": {
        "agent.name": "SPRAgent",
        "agent.train.lr_predictor": 1e-4,
        "agent.train.ortho_coef": 0.1,
        "agent.train.multi_step": True,
        "data.future": 0.98,
        "agent.model.archi.predictor.hidden_dim": 1024,
        "agent.model.archi.predictor.hidden_layers": 1,
    },
    "icvf": {
        "agent.name": "ICVFAgent",
        "agent.train.expectile": 0.9,
        "agent.train.prob_random_goal": 0.375,
        "agent.model.archi.features.norm": True,
        "agent.model.center_features": False,
        "data.future": 0.99,
        "agent.model.archi.L_dim": 50,
    },
}


def sweep_spr_walker():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["walker"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "ortho_coef": [0.001, 0.01, 0.1]},
        },
    }
    return conf


def sweep_spr_cheetah():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["cheetah"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "ortho_coef": [0.001, 0.01, 0.1]},
        },
    }
    return conf


def sweep_spr_quadruped():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["quadruped"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "ortho_coef": [0.001, 0.01, 0.1]},
        },
    }
    return conf


def sweep_spr_pointmass():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["pointmass"],
        "env.task": ["reach_top_left"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "ortho_coef": [0.001, 0.01, 0.1], "discount": [0.99], "lr_actor": [1e-6]},
        },
    }
    return conf


def sweep_hilp_walker():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["walker"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "prob_random_goal": [0.375, 0.2, 0.5]},
        },
    }
    return conf


def sweep_hilp_cheetah():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["cheetah"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "prob_random_goal": [0.375, 0.2, 0.5]},
        },
    }
    return conf


def sweep_hilp_quadruped():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["quadruped"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "prob_random_goal": [0.375, 0.2, 0.5]},
        },
    }
    return conf


def sweep_hilp_pointmass():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["pointmass"],
        "env.task": ["reach_top_left"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "prob_random_goal": [0.375, 0.2, 0.5], "discount": [0.99], "lr_actor": [1e-6]},
        },
    }
    return conf


def sweep_laplacian_walker():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["walker"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "ortho_coef": [0, 0.1, 1]},
        },
    }
    return conf


def sweep_laplacian_cheetah():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["cheetah"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "ortho_coef": [0, 0.1, 1]},
        },
    }
    return conf


def sweep_laplacian_quadruped():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["quadruped"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "ortho_coef": [0, 0.1, 1]},
        },
    }
    return conf


def sweep_laplacian_pointmass():
    conf = {
        "seed": [3917, 3502, 8948, 9460, 4729],
        "env.domain": ["pointmass"],
        "env.task": ["reach_top_left"],
        "agent": {
            "model": {"archi": {"z_dim": [50]}},
            "train": {"lr_features": [1e-4, 1e-5], "discount": [0.99], "lr_actor": [1e-6], "ortho_coef": [0, 0.1, 1]},
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
    wandb_gname: str | None = "sf"
    wandb_pname: str | None = "replearn_dmc_paper"
    # to run sweeps
    sweep_config: str | None = None
    # which sf agent to run
    sf_agent: str = "sf"
    # selects the depth of the left encoder
    left_encoder: Literal["none", "shallow", "deep"] = "shallow"


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
                "buffer_type": "parallel",
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

    flat = ConfDict.flat(base_config)
    flat.update(BASE_AGENT_CFG[args.sf_agent])
    base_config = ConfDict(flat)

    match args.left_encoder:
        case "none":
            del base_config["agent"]["model"]["archi"]["left_encoder"]
            base_config["agent"]["model"]["archi"]["left_encoder"] = {"name": "IdentityNNConfig"}
            base_config["agent"]["model"]["archi"]["successor_features"]["input_filter"] = {"name": "DictInputFilterConfig", "key": "state"}
            base_config["agent"]["model"]["archi"]["actor"]["input_filter"] = {"name": "DictInputFilterConfig", "key": "state"}
        case "shallow":
            pass
        case "deep":
            base_config["agent"]["model"]["archi"]["left_encoder"]["hidden_layers"] = 2
            base_config["agent"]["model"]["archi"]["L_dim"] = 50
        case _:
            raise NotImplementedError("Unknown left encoder configuration: ", args.left_encoder)

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
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname spr_walker_v10 --sweep_config sweep_spr_walker --sf_agent spr --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname spr_cheetah_v10 --sweep_config sweep_spr_cheetah --sf_agent spr --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname spr_quadruped_v10 --sweep_config sweep_spr_quadruped --sf_agent spr --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname spr_pointmass_v10 --sweep_config sweep_spr_pointmass --sf_agent spr --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname spr_multi_walker_v10 --sweep_config sweep_spr_walker --sf_agent spr-multi --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname spr_multi_cheetah_v10 --sweep_config sweep_spr_cheetah --sf_agent spr-multi --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname spr_multi_quadruped_v10 --sweep_config sweep_spr_quadruped --sf_agent spr-multi --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname spr_multi_pointmass_v10 --sweep_config sweep_spr_pointmass --sf_agent spr-multi --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname hilp_walker_v10 --sweep_config sweep_hilp_walker --sf_agent hilp --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname hilp_cheetah_v10 --sweep_config sweep_hilp_cheetah --sf_agent hilp --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname hilp_quadruped_v10 --sweep_config sweep_hilp_quadruped --sf_agent hilp --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname hilp_pointmass_v10 --sweep_config sweep_hilp_pointmass --sf_agent hilp --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname laplacian_walker_v10 --sweep_config sweep_laplacian_walker --sf_agent laplacian --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname laplacian_cheetah_v10 --sweep_config sweep_laplacian_cheetah --sf_agent laplacian --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname laplacian_quadruped_v10 --sweep_config sweep_laplacian_quadruped --sf_agent laplacian --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname laplacian_pointmass_v10 --sweep_config sweep_laplacian_pointmass --sf_agent laplacian --use_wandb

    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname icvf_walker_v10 --sweep_config sweep_hilp_walker --sf_agent icvf --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname icvf_cheetah_v10 --sweep_config sweep_hilp_cheetah --sf_agent icvf --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname icvf_quadruped_v10 --sweep_config sweep_hilp_quadruped --sf_agent icvf --use_wandb
    # uv run -m scripts.replearn.launch_sf_dmc --wandb_gname icvf_pointmass_v10 --sweep_config sweep_hilp_pointmass --sf_agent icvf --use_wandb
