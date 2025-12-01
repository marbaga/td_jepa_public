# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclasses.dataclass
class CSVLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def log(self, log_data: Dict[str, Any]) -> None:
        if self.fields is None:
            self.fields = sorted(list(log_data.keys()))
            if not Path(self.filename).exists():
                pd.DataFrame(columns=self.fields).to_csv(self.filename, index=False)

        data = {field: log_data.get(field, "") for field in self.fields}  # Ensure all fields are present
        islist = [isinstance(v, Iterable) and not isinstance(v, str) for k, v in data.items()]
        if all(islist):
            df = pd.DataFrame(data)
        elif not any(islist):
            df = pd.DataFrame([data])
        else:
            raise RuntimeError("Fields should all be a numbers, a string or iterable objects. We don't support mixed types.")
        df.to_csv(self.filename, mode="a", header=False, index=False)


@dataclasses.dataclass
class JSONLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def log(self, log_data: Dict[str, Any]) -> None:
        if self.fields is None:
            self.fields = sorted(list(log_data.keys()))
            if not Path(self.filename).exists():
                with open(self.filename, "w+") as f:
                    json.dump({k: [] for k in self.fields}, f)

        # not the most efficient way of logging since we cannot append
        with open(self.filename, "r+") as f:
            logz = json.load(f)
        with open(self.filename, "w+") as f:
            for field in self.fields:
                logz[field].append(log_data.get(field, ""))
            json.dump(logz, f)
