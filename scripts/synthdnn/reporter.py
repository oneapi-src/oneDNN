################################################################################
# Copyright 2024 Intel Corporation
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

import statistics as stat

from tabulate import tabulate

import metrics
from metrics import perf_data

import os
from utils import *


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


class summaryStats:
    def __init__(self, scaling, metricValue):
        self.scaling = scaling
        self.metric_value = metricValue
        self.title = f"{self.scaling.title} {self.metric_value.title}"
        self.data: dict[matmul.Kind, perf_data] = {}

        super().__init__()

    def update(self):
        # DebugPrint("summaryStats update")
        header = ["Primitive Kind", "Mean", "Std Dev", "Sample Size"]
        table = []
        for kind, data in self.data.items():
            sample_size = len(data.metrics.data)
            mean = stat.mean(data.metrics.data)
            stdev = float("nan")
            if sample_size > 1:
                stdev = stat.stdev(data.metrics.data)

            if self.scaling.format_to_percent():
                table.append(
                    [
                        kind,
                        "{:.1%}".format(mean),
                        "{:.1%}".format(stdev),
                        sample_size,
                    ]
                )
            else:
                table.append([kind, mean, stdev, sample_size])

        print(self.title)
        print(tabulate(table, header, tablefmt="rst"))

    def add(self, sample):
        # DebugPrint("summaryStats add")
        # dt = sample.primitive["dt"]
        kind = sample.kind()
        if not kind in self.data:
            self.data[kind] = perf_data(
                metrics.Metric(self.scaling, self.metric_value)
            )
        self.data[kind].add([], sample)


class summaryBounds:

    def __init__(self):
        self.title = f"Memory Bound and Compute Bound Efficiencies"
        self.data: dict[matmul.Kind, perf_data] = {}
        self.max_bandwidth = 0
        self.max_flops = {}

        super().__init__()

    def update(self):
        # DebugPrint("summaryStats update")
        header = [
            "Primitive Kind",
            "Compute Bound Mean",
            "Compute Std Dev",
            "Memory Bound Mean",
            "Memory Std Dev",
            "Sample Size",
        ]
        table = []
        for kind, data in self.data.items():
            sample_size = len(data.flops)
            if sample_size != len(data.bandwidths):
                raise RuntimeError(
                    f"Inconsistent data, the number of flops entries {len(data.flops)} is not the same as bandwidth entries {len(data.bandwidths)}"
                )

            mem_data = []
            compute_data = []
            for i in range(sample_size):
                compute_efficiency = data.flops[i] / self.max_flops[data.type]
                mem_efficiency = data.bandwidths[i] / self.max_bandwidth
                if mem_efficiency > compute_efficiency:
                    mem_data.append(mem_efficiency)
                else:
                    compute_data.append(compute_efficiency)

            compute_mean = float("nan")
            compute_stdev = float("nan")
            if len(compute_data) > 0:
                compute_mean = stat.mean(compute_data)
            if len(compute_data) > 1:
                compute_stdev = stat.stdev(compute_data)

            mem_mean = float("nan")
            mem_stdev = float("nan")
            if len(mem_data) > 0:
                mem_mean = stat.mean(mem_data)
            if len(mem_data) > 1:
                mem_stdev = stat.stdev(mem_data)

            table.append(
                [
                    kind,
                    "{:.1%} of {} GFLOPS".format(
                        compute_mean, self.max_flops[data.type]
                    ),
                    "{:.1%}".format(compute_stdev),
                    "{:.1%} of {} GB/S".format(mem_mean, self.max_bandwidth),
                    "{:.1%}".format(mem_stdev),
                    sample_size,
                ]
            )

        table.sort(key=lambda x: x[0])
        print(self.title)
        print(tabulate(table, header, tablefmt="rst"))

    def add(self, sample):
        # DebugPrint("summaryStats add")
        # dt = sample.primitive["dt"]
        kind = sample.kind()
        if not kind in self.data:
            self.data[kind] = perf_data(
                # Unused
                metrics.Metric(metrics.Absolute(), metrics.Flops())
            )
            self.data[kind].type = sample.type()

        self.data[kind].add([], sample)
        if sample.bandwidth > self.max_bandwidth:
            self.max_bandwidth = sample.bandwidth
        if (
            sample.type() not in self.max_flops
            or self.max_flops[sample.type()] < sample.flops
        ):
            self.max_flops[sample.type()] = sample.flops
