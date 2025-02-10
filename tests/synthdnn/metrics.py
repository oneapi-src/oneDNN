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

###from generation import matmul
from matmul import sampler

from utils import *

class Scaling:
    title = ""

    def __init__(self):
        self.max_value = 0

    def rescale(self, val):
        return 1

    def initialize(self, val):
        return val


class SampleRelative(Scaling):
    title = "Sample Relative"

    def rescale(self, val):
        if self.max_value == 0:
            self.max_value = val
            return 1
        elif val <= self.max_value:
            return 1
        else:
            ret = val / self.max_value
            self.max_value = val
            return ret

    def initialize(self, val):
        return val / self.max_value


class Absolute(Scaling):
    title = "Absolute"

    def rescale(self, val):
        self.max_value = max(self.max_value, val)
        return 1

    def initialize(self, val):
        return val


class Value:
    title = ""

    def get(self, sample):
        return 0


class Bandwidth(Value):
    title = "Bandwidth"

    def get(self, sample):
        return sample.bandwidth


class Flops(Value):
    title = "FLOPS"

    def get(self, sample):
        return sample.flops


class TotalEfficiency(Value):
    title = "Total Efficiency"

    def __init__(self, max_bw, max_flops):
        self.max_bw = max_bw
        self.max_flops = max_flops

    def get(self, sample):
        return max(
            sample.bandwidth / self.max_bw, sample.flops / self.max_flops
        )


class Metric:
    def __init__(self, scaling, value):
        self.scaling: Scaling = scaling
        self.value: Value = value
        self.data: list[float] = []

    @property
    def title(self):
        return f"{self.scaling.title} {self.value.title}"

##    def add(self, sample: matmul.Sample):
    def add(self, sample: sampler):
        val = self.value.get(sample)
        rescale = self.scaling.rescale(val)
        for x in self.data:
            x = x / rescale
        self.data.append(self.scaling.initialize(val))


class perf_data :
#Static class members, to be updated via add()
    max_bandwidth = 0
    max_flops = {}

    def __init__(self, perf_metric):
        self.xs = []
        self.ys = []
        self.zs = []
        self.metrics: metrics.Metric = perf_metric

        self.flops = []
        self.bandwidths = []

    def add(self, dims, sample):
        #DebugPrint("perf_data update")
        if len(dims) > 0:
            self.xs.append(dims[0])
        if len(dims) > 1:
            self.ys.append(dims[1])
        if len(dims) > 2:
            self.zs.append(dims[2])

        self.metrics.add(sample)
