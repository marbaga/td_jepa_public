# TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning
**[Meta, FAIR](https://ai.facebook.com/research/)**


# Overview
This repository provides a PyTorch implementation of TD-JEPA. It allows reproducing the experiments of the paper [TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning](https://arxiv.org/abs/2510.00739). It also provides a framework for training, evaluating, and comparing unsupervised zero-shot RL methods on proprioceptive- and pixel-based environments from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) and [OGBench](https://github.com/seohongpark/ogbench). In particular, we provide implementations of [FB](), [HILP](), [ICVF*](), [Laplacian](), [RLDP](), [BYOL*](), [BYOL-gamma*]() and [TD3](), as discussed in the paper.

This codebase is based on [Meta Motivo](https://github.com/facebookresearch/metamotivo).

# Quick Start

## Installation

We use [uv](https://github.com/astral-sh/uv) to manage dependencies. First [install it](https://docs.astral.sh/uv/getting-started/installation/), then `cd` into the td_jepa directory and run
```
uv sync --all-extras
```
This will create a virtual environment `.venv` with the dependencies required by this project. You can activate it explicitly by `source .venv/bin/activate`.

## Downloading the data

We provide scripts for downloading and processing [ExORL](https://github.com/denisyarats/exorl) and [OGBench](https://github.com/seohongpark/ogbench) datasets:
```
uv run -m scripts.data_processing.ogbench.extract_all --output_folder your_path_here
uv run -m scripts.data_processing.exorl.download --output_folder exorl_path_here
uv run -m scripts.data_processing.exorl.update_all --save_rgb --input_folder exorl_path_here --output_folder your_path_here --num_workers 8
```

## Reproducing the experiments

You can launch experiments using the scripts in `scripts/train`. Each of them trains and evaluates one of the baselines on either ExORL or OGBench data with either proprioceptive or pixel-based observations. At the bottom of each script you can find examples of commands to launch it. For instance, you can train and evaluate TD-JEPA on the walker domain from DMC with proprioceptive observations by running
```
uv run -m scripts.train.proprio.launch_td_jepa_dmc --data_path your_path_here --workdir_root your_workdir --sweep_config sweep_walker --first_only
```
Removing the `first_only` flag launches a sweep over hyperparameter grids as specified in the launcher. By default, jobs are running sequentially; we provide a lightweight slurm interface that instead submits them individually. It can be enabled by the `slurm` flag. As slurm configurations vary, the default script in `metamotivo/misc/launcher_utils.py` likely needs to be adapted.

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
