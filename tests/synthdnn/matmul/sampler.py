################################################################################
# Copyright 2025 Intel Corporation
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

import itertools
import random
import math

from matmul.primitive import *

class Region:
    def __init__(self, line):
        restrictions = []
        for x in line.split(':'):
            if len(x) <= 0 or x[0] != '(' or x[-1] != ')':
               raise RuntimeError(f"Unable to parse restrictions: {x} in {line}")
            restrictions.append(x[1:-1])
        if len(restrictions) != 3:
               raise RuntimeError(f"Invalid number of restrictions in {line}")

        self.min = [int(x) for x in restrictions[0].split(',')]
        self.max = [int(x) for x in restrictions[1].split(',')]
        self.alignment = [int(x) for x in restrictions[2].split(',')]

        if len(self.min) != len(self.max) or len(self.min) != len(self.alignment):
               raise RuntimeError(f"Inconsistent number of dimensions between restrictions in {line}")

        self.ndims = len(self.min)

    def __str__(self):
        str_min = ",".join([str(x) for x in self.min])
        str_max = ",".join([str(x) for x in self.max])
        str_alignment = ",".join([str(x) for x in self.alignment])
        return f"({str_min}):({str_max}):({str_alignment})"


class Sampler:
    def __init__(self, samples, mode, types, layouts, region):
        self.layouts = layouts
        self.mode = mode
        self.types = types
        self.region = region
        self.samples = samples

        random.seed("oneDNN Matmul")
        self.kinds = [Kind(x,y) for x,y in itertools.product(layouts, types)]
        random.shuffle(self.kinds)
        self.dim_sampler = self.DimSampler(region)

    def __str__(self):
        return f"-s {self.samples} -m {self.mode} -l {self.layouts} -r {self.region} -t {self.types}"

    def __iter__(self):
        if self.mode == "zip":
            return self.ZipIter(self.samples, self.kinds, self.dim_sampler)
        elif self.mode == "product":
            return self.ProductIter(self.samples, self.kinds, self.dim_sampler)
        else:
            raise RuntimeError(f"Unknown iteration mode {self.mode}")


    # Itertools.product seems to break on an infinite sampler
    class ProductIter:
       def __init__(self, samples, kinds, dim_sampler):
           self.dim_sampler = dim_sampler
           self.kinds = kinds
           self.kinds_iter = iter(self.kinds)
           self.rem_samples = samples

       def __next__(self):
           if(self.rem_samples == 0):
               raise StopIteration

           try:
               self.k = next(self.kinds_iter);
               self.s = next(self.dim_sampler)
           except StopIteration:
               self.kinds_iter = iter(self.kinds)
               self.k = next(self.kinds_iter)
               self.s = next(self.dim_sampler)
               self.rem_samples = self.rem_samples - 1

           return Primitive(self.k, self.s)

    class ZipIter:
       def __init__(self, samples, kinds, dim_sampler):
           self.dim_sampler = dim_sampler
           self.kinds_iter = itertools.cycle(kinds)
           self.rem_samples = samples

       def __next__(self):
           if(self.rem_samples == 0):
               raise StopIteration

           self.rem_samples = self.rem_samples - 1
           k = next(self.kinds_iter);
           s = next(self.dim_sampler)

           return Primitive(k, s)

    class DimSampler:
        def __init__(self, region):
            self.region = region
            self.seen = set()
            if region.ndims < 3:
               raise RuntimeError(f"Insufficient dimensions for matmul operation, expected at least 3, but got {region.ndims}")

        def __next__(self, min_size=pow(2, 10)):

            # Sample from a power distribution as most problem features occur
            # when some dimension is small. In addition, small problems often
            # require less time to run enabling faster data collection
            def get_sample(minval, maxval, align):
                assert minval <= maxval, "Sample bounds are out of order"
                if minval == maxval:
                    return minval
                x = round(pow(2, random.uniform(math.log2(minval), math.log2(maxval))))
                return (x // align) * align

            for _ in range(1000):
                dims = [0] * self.region.ndims
                for i in range(self.region.ndims):
                    dims[i] = get_sample(
                        self.region.min[i],
                        self.region.max[i],
                        self.region.alignment[i],
                        )
                dims_tuple = tuple(dims)
                if dims_tuple not in self.seen and math.prod(dims) >= min_size:
                    self.seen.add(dims_tuple)
                    return Dims(dims[:-3], dims[-3], dims[-2], dims[-1])
            raise RuntimeError(
                "Can not effectively sample problems larger than: "
                + str(min_size)
            )
