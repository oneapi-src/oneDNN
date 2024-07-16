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


class Dims:
    def __init__(self, b, m, k, n):
        # b is a list due to variable size
        self.b = b
        self.m = m
        self.k = k
        self.n = n

    def __str__(self):
        a_dims = self.b + [self.m, self.k]
        b_dims = self.b + [self.k, self.n]
        a_str = "x".join([str(x) for x in a_dims])
        b_str = "x".join([str(x) for x in b_dims])
        return f"{a_str}:{b_str}"

    def __eq__(self, other):
        return (self.b, self.m, self.k, self.n) == (other.b, other.m, other.k, other.n)

    def __hash__(self):
        return hash((self.b, self.m, self.k, self.n))


class Layouts:
    class Layout:
        def __init__(self, layout):
            self.A, self.B, self.C = layout.split(":")

        def benchdnn_str(self):
            return f"--stag={self.A} --wtag={self.B} --dtag={self.C}"

        def __str__(self):
            return f"{self.A}:{self.B}:{self.C}"

        def __eq__(self, other):
            return (self.A, self.B, self.C) == (other.A, other.B, other.C)

    def __init__(self, layouts, ndims):
        if(layouts == "all"):
            self.values = self.supported(ndims)
        else:
            self.values = [self.Layout(x) for x in layouts.split(",")]

    def __str__(self):
        return ",".join([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    @staticmethod
    def supported(ndims):
        dim_str = None
        if ndims == 2:
            dim_str="ab"
        if ndims == 3:
            dim_str = "abc"
        if ndims == 4:
            dim_str = "abc"
        if dim_str == None:
            raise RuntimeError(f"No support for ndims={ndims}")

        perms = [''.join(p) for p in itertools.permutations("abc")]

        perms.insert(0, "any")


        return [Layouts.Layout(f"{a}:{b}:{c}") for a,b,c in itertools.product(perms, perms, perms)]


class Types:
    class Type:
        def __init__(self, type_str):
            s = type_str.split("(")
            self.A, self.B, self.C = s[0].split(":")
            if len(s) < 2:
                self.mode = None
            else:
                self.mode = s[1].strip(")")

        def __str__(self):
            mode_str = ""
            if self.mode:
                mode_str=f"({self.mode})"
            return f"{self.A}:{self.B}:{self.C}{mode_str}"

        def benchdnn_str(self):
            mode_str = ""
            if not self.mode is None:
                mode_str = f"--attr-fpmath={self.mode}"
            return f"--dt={self.A}:{self.B}:{self.C} {mode_str}"

        def __eq__(self, other):
            return (self.A, self.B, self.C, self.mode) == (
                other.A,
                other.B,
                other.C,
                other.mode,
            )

    def __init__(self, types):
        if(types == "all"):
            self.values = self.supported()
        else:
            self.values = [self.Type(x) for x in types.split(",")]

    def __str__(self):
        return ",".join([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    @staticmethod
    def supported():
        return [
            Types.Type("f64:f64:f64"),
            Types.Type("f32:f32:f32"),
            Types.Type("f32:f32:f32(tf32)"),
            Types.Type("f32:f32:f32(f16)"),
            Types.Type("f32:f32:f32(bf16)"),
            Types.Type("f32:u8:f32"),
            Types.Type("f32:u4:f32"),
            Types.Type("bf16:bf16:bf16"),
            Types.Type("bf16:u8:bf16(bf16:true)"),
            Types.Type("bf16:u4:bf16(bf16:true)"),
            Types.Type("f16:f16:f16"),
            Types.Type("f16:u8:f16(f16:true)"),
            Types.Type("f16:u4:f16(f16:true)"),
            Types.Type("u8:u8:u8"),
            Types.Type("u8:s8:s32"),
            Types.Type("s8:u8:f32"),
            Types.Type("s8:s8:f16"),
            Types.Type("u8:u8:bf16")
        ]


# Kind represents problem parameters that do not make sense to consider
# in aggregate for optimization purposes as these features require significant
# changes within generated implementations or the implementation dispatching.
class Kind:
    def __init__(self, layout, type):
        self.layout = layout
        self.type = type

    def __str__(self):
        return f"{self.layout.benchdnn_str()} {self.type.benchdnn_str()}"

    def __eq__(self, other):
        return (self.layouts, self.type) == (other.layout, other.type)

    def __hash__(self):
        return hash((self.layouts, self.type))

class Primitive:
    def __init__(self, kind, dims):
        self.kind: Kind = kind
        self.dims = dims

    @staticmethod
    def from_repro(repro_line):
        t = self.Type("f32:f32:f32")
        l = Layouts.Layout("any:any:any")
        dims = Dims([], 0, 0, 0)

        for arg in repro_line.split(" "):
            if arg.startswith("--dt="):
                t = self.Type(arg.split("=")[1])
            elif arg.startswith("--attr-fpmath="):
                t.mode = arg.split("=")[1]
            elif arg.startswith("--stag="):
                l.A = arg.split("=")[1]
            elif arg.startswith("--wtag="):
                l.B = arg.split("=")[1]
            elif arg.startswith("--dtag="):
                l.C = arg.split("=")[1]
            elif not arg.startswith("--"):
                argsA, argsB = [a.split("x") for a in arg.split(":")];
                if len(argsA) == len(argsB) and len(argsA) >= 2:
                    raise RuntimeError(f"Invalid Matrix Dimension {arg}")
                dims.b = [int(x) for x in argsA[:-2]]
                if dims.b != [int(x) for x in argsB[:-2]]:
                    raise RuntimeError(f"A and B batch dimensions {argsA[:-2]} and {argsB[:-2]} do not match")
                dims.m = int(argsA[-2])
                dims.k = int(args[-1])
                if dims.k != int(argsB[-2]):
                    raise RuntimeError(f"A and B k dimensions {argsA[-1]} and {argsB[-2]} do not match")
                dims.n = int(argsB[-1])

        return Primitive(Kind(l, t), dims)

    def __str__(self):
        return f"{self.kind} {self.dims}"

    def __eq__(self, other):
        return (self.kind, self.dims) == (other.kind, other.dims)

    def __hash__(self):
        return hash((self.kind, self.dims))

    def __getitem__(self, key):
        if key == "m":
            return int(self.dims.m)
        if key == "k":
            return int(self.dims.k)
        if key == "n":
            return int(self.dims.n)
        if key == "dt":
            return self.kind.types
        raise RuntimeError("Unknown primitive key: " + key)



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
