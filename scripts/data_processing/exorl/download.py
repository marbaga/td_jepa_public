# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from pathlib import Path

import tyro

DEFAULT_DATA_DIR = Path("/large_experiments/unicorns/datasets/url_benchmark")
if not DEFAULT_DATA_DIR.exists():
    DEFAULT_DATA_DIR = Path("/fsx-unicorns/shared/datasets/url_benchmark")


def main(folder: Path = DEFAULT_DATA_DIR):
    folder = folder / "original"
    folder.mkdir(parents=True, exist_ok=True)
    for domain in ["quadruped", "cheetah", "walker", "pointmass"]:
        orig_domain = "point_mass_maze" if domain == "pointmass" else domain
        (folder / domain).mkdir(parents=True, exist_ok=True)
        print(domain)
        result = subprocess.run(
            ["wget", "-O", f"{str(folder / domain / 'rnd.zip')}", f"https://dl.fbaipublicfiles.com/exorl/{orig_domain}/rnd.zip"],
            capture_output=True,
        )
        assert "100%" in result.stdout.decode() or "100%" in result.stderr.decode(), f"Error downloading {domain} rnd dataset"
        print("Download done")
        result = subprocess.run(
            ["unzip", "-qq", f"{str(folder / domain / 'rnd.zip')}", "-d", f"{str(folder / domain / 'rnd')}"], capture_output=True
        )
        print(result)


if __name__ == "__main__":
    tyro.cli(main)
