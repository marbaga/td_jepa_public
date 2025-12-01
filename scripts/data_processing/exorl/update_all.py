# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import tyro

from scripts.data_processing.exorl.update_data import main as update_data_main

#####
# uv run python -m scripts.data_processing.exorl.update_all
#####
DEFAULT_DATA_DIR = Path("/large_experiments/unicorns/datasets/url_benchmark")
if not DEFAULT_DATA_DIR.exists():
    DEFAULT_DATA_DIR = Path("/fsx-unicorns/shared/datasets/url_benchmark")


def main(
    folder: Path = DEFAULT_DATA_DIR,
    num_workers: int = 0,
    save_rgb: bool = False,
):
    for domain in ["quadruped", "cheetah", "walker", "pointmass"]:
        update_data_main(
            num_workers=num_workers,
            env_name=domain,
            expl_agent="rnd",
            datasets_dir=folder / "original",
            new_dataset_dir=folder / "processed",
            save_rgb=save_rgb,
        )


if __name__ == "__main__":
    tyro.cli(main)
