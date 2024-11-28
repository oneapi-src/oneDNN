################################################################################
# Copyright 2022-2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

from collections import defaultdict
from typing import Any, Dict, List

from . import ir


class Aggregate:
    def __init__(self):
        self.occurrences = 0
        self.time = 0.0

    def add(self, occurrence: float):
        self.occurrences += 1
        self.time += occurrence

    def __iter__(self):
        yield self.occurrences
        yield self.time


class BreakdownGenerator:
    """
    Generates an input for benchdnn from internal representation.
    """

    def __init__(self, _: Any = None):  # Maintain old interface
        pass

    def generate(self, input: Dict[int, ir.Entry], agg_keys: List[str]):
        data: Dict[str, Aggregate] = defaultdict(Aggregate)
        ofs = ","

        if not input:
            return {}

        def key2str(key, value):
            def mds2str(mds):
                return " ".join(map(str, mds))

            def aux2str(aux):
                auxfs = " "
                return auxfs.join([f"{k}:{v}" for k, v in aux.items()])

            if key == "mds":
                return mds2str(value)
            elif key == "aux":
                return aux2str(value)
            else:
                return str(value)

        # Gather occurences and aggregate time statistics
        total_time: float = 0
        for value in input.values():
            item_key = ofs.join(key2str(k, getattr(value, k)) for k in agg_keys)
            data[item_key].add(value.time)
            total_time += value.time

        # sort keys by increasing total time
        sorted_keys = sorted(data, key=lambda t: data[t].time, reverse=True)

        cum_entry: int = 0
        cum_time: float = 0
        avg_call: float = 0
        sorted_avg_call: Dict[str, float] = {}
        sorted_cum_time: Dict[str, float] = {}
        for key in sorted_keys:
            item = data[key]
            cum_entry += 1
            cum_time = cum_time + item.time
            avg_call = avg_call + (item.occurrences - avg_call) / cum_entry
            sorted_avg_call[key] = avg_call
            sorted_cum_time[key] = cum_time

        fixed_keys = [
            "ncalls",
            "time(ms)",
            "overall%",
            "agg_ncalls(avg)",
            "agg_time(ms)",
            "agg_overall%",
        ]

        output = ofs.join(agg_keys + fixed_keys)

        def str_num(s):
            return f"{s:.2f}"

        def str_pct(s):
            return f"{s * 100:.2f}"

        def safe_div(n, d):
            # Assumption: 0 <= n <= d
            # If the assumption is broken, we can still raise ZeroDivisionError
            return 1 if n == d == 0 else n / d

        for key in sorted_keys:
            item = data[key]
            avg_call = sorted_avg_call[key]
            cum_time = sorted_cum_time[key]
            fields = [
                str(key),
                str(item.occurrences),
                str_num(item.time),
                str_pct(safe_div(item.time, total_time)),
                str_num(avg_call),
                str_num(cum_time),
                str_pct(safe_div(cum_time, total_time)),
            ]
            output += "\n" + ofs.join(fields)
        return {"all": output}
