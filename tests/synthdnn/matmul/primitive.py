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

from utils import *

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

    def __init__(self, layouts, ndims):
        if(layouts == "all"):
            self.values = self.supported(ndims-1)
        else:
            self.values = [self.Layout(x) for x in layouts.split(",")]

    def __iter__(self):
        return iter(self.values)

    @staticmethod
    def supported(ndims):
        if ndims < 2 or ndims > 6:
            raise RuntimeError(f"No support for ndims={ndims}")
        dims_base="abcdef"
        perms = [''.join(p) for p in itertools.permutations(dims_base[:ndims])]
        #DebugPrint(f"ndims = {ndims}")
        #DebugPrint(f"perms = {perms}")
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

class Primitive:
    def __init__(self, kind, dims):
        self.kind: Kind = kind
        self.dims = dims

    def __str__(self):
        return f"{self.kind} {self.dims}"


