# TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning
**[Meta, FAIR](https://ai.facebook.com/research/)**

# Features

- Benchmarked implementations of **zero-shot RL algorithms**:
  - [TD-JEPA (including symmetric variant)](https://arxiv.org/abs/2510.00739)
  - Existing methods, including [FB](https://arxiv.org/abs/2103.07945), [HILP](https://arxiv.org/abs/2402.15567), [ICVF*](https://arxiv.org/pdf/2304.04782), [Laplacian](https://arxiv.org/pdf/1810.04586), [RLDP](https://openreview.net/forum?id=jdL6WB5jHZ), [BYOL*](https://arxiv.org/abs/2006.07733), [BYOL-γ*](https://arxiv.org/pdf/2506.10137)
- Zero-shot training and **evaluation** across:
  - [ExORL/DMC](): `walker`, `cheetah`, `quadruped`, `point-mass-maze`
  - [OGBench](): `antmaze-{medium,large}-{navigate,stitch}`, `antmaze-medium-explore`, `cube-single`, `cube-double`, `puzzle-3x3, scene`
- Support for learning from both states and **RGB** input.
- Minimal cluster integrations.

# Overview
This repository provides a PyTorch implementation of TD-JEPA. It allows reproducing the experiments of the paper [TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning](https://arxiv.org/abs/2510.00739). It also provides a framework for training, evaluating, and comparing unsupervised zero-shot RL methods on proprioceptive- and pixel-based environments from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) and [OGBench](https://github.com/seohongpark/ogbench). The implementation of baselines is discussed in detail in the [paper](https://arxiv.org/abs/2510.00739).

This codebase is based on [Meta Motivo](https://github.com/facebookresearch/metamotivo).

# Quick Start

## Installation

We use [uv](https://github.com/astral-sh/uv) to manage dependencies. After [installation](https://docs.astral.sh/uv/getting-started/installation/), simply `cd` into the `td_jepa` directory and run
```
uv sync --all-extras
```
This will create a virtual environment in `.venv` with the dependencies required by this project. You can activate it explicitly by `source .venv/bin/activate`.

## Downloading the data

We provide scripts for downloading and processing [ExORL](https://github.com/denisyarats/exorl) and [OGBench](https://github.com/seohongpark/ogbench) datasets:
```
# OGBench
uv run -m scripts.data_processing.ogbench.extract_all --output_folder your_path_here
# ExORL
uv run -m scripts.data_processing.exorl.download --output_folder exorl_path_here
# the following command regenerates the data to match the current mujoco version
uv run -m scripts.data_processing.exorl.update_all --save_rgb --input_folder exorl_path_here --output_folder your_path_here --num_workers 8
```

## Reproducing the experiments

You can launch experiments using the scripts in `scripts/train`. Each of them trains and evaluates one of the baselines on either ExORL or OGBench data with either proprioceptive or pixel-based observations. Jointly, these scripts will reproduce the results displayed in Table 1.
At the bottom of each script you can find examples of commands to launch it. For instance, you can train and evaluate TD-JEPA on the walker domain from DMC with proprioceptive observations by running
```
uv run -m scripts.train.proprio.launch_td_jepa_dmc --use_wandb --wandb_gname td_jepa_walker_proprio --data_path datasets --workdir_root results --sweep_config sweep_walker
```
By default, this command will sequentially run through a grid of experiments. Adding a `first_only` flag will only run the first experiment in the grid. By default, jobs are executed locally.

## Running on a cluster

We provide lightweight slurm/[exca](https://github.com/facebookresearch/exca) interfaces that instead submit jobs to a cluster. They can be enabled by appending the `slurm` and `exca` flags, respectively, to each command. These interfaces are fully contained in `metamotivo/misc/launcher_utils.py`: as slurm configurations vary, the default scripts in `metamotivo/misc/launcher_utils.py` likely needs to be adapted (respectively, on line 81 and 152).

# Citation
```
@article{bagatella2025td,
  title={TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning},
  author={Bagatella, Marco and Pirotta, Matteo and Touati, Ahmed and Lazaric, Alessandro and Tirinzoni, Andrea},
  journal={arXiv preprint arXiv:2510.00739},
  year={2025}
}
```

# License

TD-JEPA is licensed under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
