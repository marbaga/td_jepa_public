# TD-JEPA
**[Meta, FAIR](https://ai.facebook.com/research/)**


# Overview
This repository provides a PyTorch implementation of TD-JEPA. It allows reproducing the experiments of the paper [TD-JEPA: Latent-predictive Representations for Zero-Shot Reinforcement Learning](https://arxiv.org/abs/2510.00739). It also provides a framework for training, evaluating, and comparing unsupervised zero-shot RL methods on proprioceptive- and pixel-based environments from the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) and [OGBench](https://github.com/seohongpark/ogbench).

This codebase is based on [Meta Motivo](https://github.com/facebookresearch/metamotivo).

# Quick Start

## Installation

We use [uv](https://github.com/astral-sh/uv) to manage dependencies. First install it e.g. via `pip install uv`. Then `cd` into the td_jepa directory and run
```
uv sync --all-extras
```
This will create a virtual environment `.venv` with the dependencies required by this project. You can activate it by `source .venv/bin/activate`. 

## Downloading the data

Download the [ExORL](https://github.com/denisyarats/exorl) and [OGBench](https://github.com/seohongpark/ogbench) datasets following the instructions in the original repositories. Then run the following scripts to preprocess them in the format required by our code
```
uv run -m scripts.data_processing.ogbench.extract_all --save_rgb --dataset_path ogbench_path_here --output-folder your_path_here
uv run -m scripts.data_processing.exorl.update_all --save_rgb --datasets_dir exorl_path_here --new_dataset_dir your_path_here
```

## Reproducing the experiments

You can launch experiments using the scripts in `scripts/train`. Each of them trains and evaluates one of the baselines on either ExORL or OGBench data with either proprioceptive or pixel-based observations. At the bottom of each script you can find examples of commands to launch it. For instance, you can train and evaluate TD-JEPA on the walker domain from DMC with proprioceptive observations by running
```
uv run -m scripts.train.proprio.launch_td_jepa_dmc --local --sweep_config sweep_walker
```
NOTE: you first need to update the `data_path` variable in each script to point to the directory where you saved the processed datasets.


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
