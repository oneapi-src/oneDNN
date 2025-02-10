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
    os.system('cls' if os.name == 'nt' else 'clear')

class summaryStats:
    def __init__(self, scaling, metricValue):
        self.scaling = scaling
        self.metric_value=metricValue
        self.title = f"{self.scaling.title} {self.metric_value.title}"
        self.data: dict[matmul.Kind, perf_data] = {}

        super().__init__()

    def update(self,doprint):
        #DebugPrint("summaryStats update")
        header = ["Primitive Kind", "Mean", "Std Dev", "Sample Size"]
        table = []
        for kind, data in self.data.items():
            sample_size = len(data.metrics.data)
            mean = stat.mean(data.metrics.data)
            stdev = float('nan')
            if sample_size > 1:
                #DebugPrint(f"{doprint} sample_size = {sample_size}")
                #DebugPrint(f"{doprint} data = {data.metrics.data}")
                stdev = stat.stdev(data.metrics.data)
            table.append([kind, "{:.1%}".format(mean), "{:.1%}".format(stdev), sample_size])

        if doprint:
            print(self.title)
            print(tabulate(table, header, tablefmt="rst"))
    """
    def finalize(self):
        DebugPrint("summaryStats finalize")
        header = ["Primitive Kind", "Mean-final", "Std Dev", "Sample Size"]
        table = []
        for kind, data in self.data.items():
            sample_size = len(data.metrics.data)
            mean = stat.mean(data.metrics.data)
            stdev = float('nan')
            if sample_size > 1:
                stdev = stat.stdev(data.metrics.data)
            table.append([kind, "{:.1%}".format(mean), "{:.1%}".format(stdev), sample_size])

        print(self.title)
        print(tabulate(table, header, tablefmt="rst"))
    """
    def add(self, sample):
        #DebugPrint("summaryStats add")
        #dt = sample.primitive["dt"]
        kind = sample.kind()
        if not kind in self.data:
            self.data[kind] = perf_data(metrics.Metric(self.scaling, self.metric_value))
        self.data[kind].add([], sample)
