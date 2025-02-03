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

import itertools
import random
import re
import math


class Dims:
    def __init__(self, m, k, n):
        self.m = m
        self.k = k
        self.n = n

    def __str__(self):
        return f"{self.m}x{self.k}:{self.k}x{self.n}"

    def __eq__(self, other):
        return (self.m, self.k, self.n) == (other.m, other.k, other.n)

    def __hash__(self):
        return hash((self.m, self.k, self.n))


class Layouts:
    def __init__(self, layout):
        self.A, self.B, self.C = layout.split(":")

    def __str__(self):
        return f"--stag={self.A} --wtag={self.B} --dtag={self.C}"

    def __eq__(self, other):
        return (self.A, self.B, self.C) == (other.A, other.B, other.C)

    def __hash__(self):
        return hash((self.A, self.B, self.C))

    @staticmethod
    def supported():
        return {
            Layouts("ab", "ab", "ab"),
            Layouts("ab", "ba", "ab"),
            Layouts("ba", "ab", "ab"),
            Layouts("any", "any", "any"),
        }


class Types:
    def __init__(self, type_str):
        s = type_str.split("(")
        self.A, self.B, self.C = s[0].split(":")
        if len(s) < 2:
            self.mode = None
        else:
            self.mode = s[0].strip(")")

    def __str__(self):
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

    def __hash__(self):
        return hash((self.A, self.B, self.C, self.mode))

    @staticmethod
    def supported():
        return {
            # Types("f32:f32:f32", None),
            # Types("f32:f32:f32", "tf32"),
            # Types("f32:f32:f32", "f16"),
            # Types("f32:f32:f32", "bf16"),
            # Types("bf16:bf16:bf16", None),
            # Types("f16:f16:f16", None),
            Types("f16:u4:f16", "f16:true"),
            # Types("s8:s8:s8", None),
        }


# Kind represents problem parameters that do not make sense to consider
# in aggregate for optimization purposes as these features require significant
# changes within generated implementations or the implementation dispatching.
class Kind:
    def __init__(self, layouts, types):
        self.layouts = layouts
        self.types = types

    def __str__(self):
        return f"{self.layouts} {self.types}"

    def __eq__(self, other):
        return (self.layouts, self.types) == (other.layouts, other.types)

    def __hash__(self):
        return hash((self.layouts, self.types))

    @staticmethod
    def supported():
        return [
            Kind(x[0], x[1])
            for x in itertools.product(Layouts.supported(), Types.supported())
        ]


class Primitive:
    def __init__(self, kind, dims):
        self.kind: Kind = kind
        self.dims = dims

    @staticmethod
    def from_repro(repro_line):
        types = Types("f32:f32:f32", None)
        layouts = Layouts("any", "any", "any")
        dims = Dims(0, 0, 0)

        for arg in repro_line.split(" "):
            if arg.startswith("--dt="):
                types = Types(arg.split("=")[1], types.mode)
            elif arg.startswith("--attr-fpmath="):
                types.mode = arg.split("=")[1]
            elif arg.startswith("--stag="):
                layouts.A = arg.split("=")[1]
            elif arg.startswith("--wtag="):
                layouts.B = arg.split("=")[1]
            elif arg.startswith("--dtag="):
                layouts.C = arg.split("=")[1]
            elif not arg.startswith("--"):
                dims.m, dims.k, dims.n = arg.split("x")
                dims.k = dims.k.split(":")[0]

        return Primitive(Kind(layouts, types), dims)

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


class Sample:
    def __init__(self, sample_line):
        _, name, prb, flops, bandwidth = sample_line.strip().split(",")
        self.name = name
        self.primitive = Primitive.from_repro(prb)
        self.flops = float(flops)
        self.bandwidth = float(bandwidth)
    def kind(self):
        if(self.name != ""):
            return self.name + ": " + str(self.primitive.kind)
        else:
            return str(self.primitive.kind)

class Region:
    def __init__(self, line):
        p = re.compile(
            r"\((\d+),(\d+),(\d+)\)-\((\d+),(\d+),(\d+)\)/\((\d+),(\d+),(\d+)\)"
        )
        match = p.match(line)
        if not match:
            raise RuntimeError(f"Unable to parse Region: {line}")
        self.min_M = int(match.group(1))
        self.min_N = int(match.group(2))
        self.min_K = int(match.group(3))

        self.max_M = int(match.group(4))
        self.max_N = int(match.group(5))
        self.max_K = int(match.group(6))

        self.align_M = int(match.group(7))
        self.align_N = int(match.group(8))
        self.align_K = int(match.group(9))

    def __str__(self):
        return f"({self.min_M},{self.min_N},{self.min_K}-{self.max_M},{self.max_N},{self.max_K}/{self.align_M},{self.align_N},{self.align_K})"


class Sampler:
    def __init__(self, samples, kinds, region):
        self.sampler = self.DimSampler(samples, region)
        self.kinds = kinds

    def __iter__(self):
        self.iterator = itertools.product(self.kinds, self.sampler)
        return self

    def __next__(self):
        k, s = self.iterator.__next__()
        return Primitive(k, s)

    class DimSampler:
        def __init__(self, samples, region):
            self.samples = int(samples)
            self.region = region

        def __iter__(self):
            self.previous = set()
            return self

        def __next__(self, min_size=pow(2, 20)):

            # Sample from a power distribution as most problem features occur
            # when some dimension is small. In addition, small problems often
            # require less time to run enabling faster data collection
            def get_sample(minval, maxval, align):
                assert minval <= maxval, "Sample bounds are out of order"
                if minval == maxval:
                    return minval
                x = round(pow(2, random.uniform(math.log2(minval), math.log2(maxval))))
                return (x // align) * align

            if len(self.previous) < self.samples:
                for _ in range(1000):
                    m = get_sample(
                        self.region.min_M,
                        self.region.max_M,
                        self.region.align_M,
                    )
                    k = get_sample(
                        self.region.min_K,
                        self.region.max_K,
                        self.region.align_K,
                    )
                    n = get_sample(
                        self.region.min_N,
                        self.region.max_N,
                        self.region.align_N,
                    )
                    pt = (m, k, n)
                    if pt not in self.previous and m * k * n >= min_size:
                        self.previous.add(pt)
                        return Dims(m, k, n)
                raise RuntimeError(
                    "Can not effectively sample problems larger than: "
                    + str(min_size)
                )
            else:
                raise StopIteration


def get_llm_2nd_token_sampler(samples):
    # TODO: Alignment should not be hard-coded, it is type dependent
    region = Region("(1,1024,1024)-(1,32768,32768)/(1,32,32)")
    # TODO: Int8/int4 need dequantization
    types = [
        Types("f32:f32:f32"),
        Types("bf16:bf16:bf16"),
        Types("f16:f16:f16"),
        Types("f16:u8:f16(f16:true)"),
        Types("f16:u4:f16(f16:true)"),
        Types("bf16:u8:bf16(bf16:true)"),
        Types("bf16:u4:bf16(bf16:true)"),
        Types("f32:u8:f32"),
        Types("f32:u4:f32"),
    ]
    layouts = Layouts.supported()
    kinds = [Kind(l,t) for l in layouts for t in types]
    return Sampler(samples, kinds, region)

def default_matmul_sampler(args):
    types = [Types(d) for d in args.types.split(',')]
    layouts = [Layouts(l) for l in args.layouts.split(',')]
    kinds = [Kind(l,t) for l in layouts for t in types]
    return Sampler(args.samples, kinds, Region(args.region))

def setup_parser(parser):
    parser.add_argument("-r", "--region",
                        default="(1,1024,1024)-(1,32768,32768)/(1,32,32)",
                        help="(m_min,n_min,k_min)-(m_max,n_max,k_max)/(m_align,n_align,k_align)")
    parser.add_argument("-t", "--types",
                        default="f32:f32:f32",
                        help="dt:dt:dt(optional fpmath-mode), comma separated list of type configurations")
    parser.add_argument("-l", "--layouts",
                        default="any:any:any",
                        help="stag:wtag:dtag, comma separated list of layouts")
    parser.add_argument("--sampler", default=default_matmul_sampler, help="function used to generate the problem sampler")
    parser.add_argument("benchdnn", help="path to benchdnn executable")
    return parser;
