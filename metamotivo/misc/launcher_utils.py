# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import typing as tp

import rich
from exca.confdict import ConfDict

from entry_points.train_offline import TrainConfig as OfflineTrainConfig
from metamotivo.base import BaseConfig


def all_combinations_of_nested_dicts_for_sweep(nested_dict):
    """
    For a nested dict, with lists, return unique dictionaries for all combinations

    This is done by flattening the dict, doing all combinations and returning back
    """
    # NOTE that this needs more careful thinking if you have "." in the name etc
    # Use confdict to flatten everything
    flat_dict = ConfDict(nested_dict).flat()
    keys = list(flat_dict.keys())
    return_list = []
    for vals in itertools.product(*[flat_dict[k] for k in keys]):
        sweep_dict = dict(zip(keys, vals))
        # Use ConfDict to get the nested dict back
        return_list.append(ConfDict(sweep_dict))
    return return_list


def launch_trials(
    base_config: BaseConfig,
    trials: list[ConfDict | dict],
    local: bool = False,
    dry: bool = False,
    # Maximum number of workers (i.e. concurrent jobs in the SLURM job array) to be launched at any time
    max_workers: int = 1024,
    # Config keys (only top level) to check for duplicates
    prevent_duplicates_of: tuple[str] = ("work_dir",),
    # Explicit flag to use current python env to run experiments
    # This is highly discouraged, as it will lead to issues if environment changes during experiments
    use_current_python_env: bool = False,
):
    """
    Launch trials, applying their changes on the base_config and using the exca job array.
    The job array will use maximum max_workers at a time.
    If local is True, run the first trial locally.
    If dry is True, print the configs instead of running them.
    """

    assert local, "Only local runs are supported currently"

    # Build full config
    first_trial = ConfDict(trials[0])
    launch_config = base_config.infra.clone_obj(first_trial)
    if dry:
        rich.print(launch_config)
    else:
        assert isinstance(launch_config, OfflineTrainConfig)
        workspace = launch_config.build()
        workspace.train()
    return


def get_all_xmls_from_location(xml_location: str) -> tp.List[str]:
    """
    Get all the xml files from a given location.
    """
    from pathlib import Path

    xml_files = []
    for path in Path(xml_location).rglob("*.xml"):
        xml_files.append(str(path))
    return xml_files
