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


class Dims:
    def __init__(self, b, m, n, k):
        # b is a list due to variable size
        self.b = b
        self.m = m
        self.n = n
        self.k = k

    def __str__(self):
        a_dims = self.b + [self.m, self.k]
        b_dims = self.b + [self.k, self.n]
        a_str = "x".join([str(x) for x in a_dims])
        b_str = "x".join([str(x) for x in b_dims])
        return f"{a_str}:{b_str}"

    def __eq__(self, other):
        return (self.b, self.m, self.n, self.k) == (
            other.b,
            other.m,
            other.n,
            other.k,
        )

    def __hash__(self):
        return hash((self.b, self.m, self.n, self.k))


class Layouts:
    class Layout:
        def __init__(self, layout):
            self.A, self.B, self.C = layout.split(":")

        def benchdnn_str(self):
            return f"--stag={self.A} --wtag={self.B} --dtag={self.C}"

    def __init__(self, layouts, ndims):
        if layouts == "all":
            self.values = self.supported(ndims)
        else:
            self.values = [self.Layout(x) for x in layouts.split(",")]

    def __iter__(self):
        return iter(self.values)

    @staticmethod
    def supported(ndims):
        if ndims < 2 or ndims > 6:
            raise RuntimeError(f"No support for ndims={ndims}")
        dims_base = "abcdef"
        gemm_kn = dims_base[ndims - 1]
        gemm_mk = dims_base[ndims - 2]
        perms = [
            "".join(p)
            for p in itertools.permutations(dims_base[:ndims])
            if p[-1] == gemm_kn or p[-1] == gemm_mk
        ]
        perms.insert(0, "any")
        return [
            Layouts.Layout(f"{a}:{b}:{c}")
            for a, b, c in itertools.product(perms, perms, perms)
            if c == "any" or c[-1] == gemm_kn
        ]


class Types:
    class Type:
        def __init__(self, type_str):
            s = type_str.split("(")
            self.A, self.B, self.C = s[0].split(":")
            self.A, self.B, self.C = self.wildcard_match(self.A, self.B, self.C)
            if len(s) < 2:
                self.mode = None
            else:
                self.mode = s[1].strip(")")

        @staticmethod
        def wildcard_match(A, B, C):
            wildcard_match = A
            B = B.replace("*", wildcard_match)
            C = C.replace("*", wildcard_match)
            return [A, B, C]

        def __str__(self):
            mode_str = ""
            if self.mode:
                mode_str = f"({self.mode})"
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
        if types == "all":
            self.values = self.supported()
        else:
            self.values = [self.Type(x) for x in types.split(",")]

    def __str__(self):
        return ",".join([str(x) for x in self.values])

    def __iter__(self):
        return iter(self.values)

    @staticmethod
    def supported():
        support_matrix = [
            [["f64"], ["f64"], ["f64"]],
            [["f32"], ["f32"], ["f32"]],
            [["f32"], ["u8", "s8"], ["f32", "f16", "bf16"]],
            [
                ["f16", "bf16"],
                ["*", "u8", "s8", "u4", "s4"],
                ["f32", "*", "u8", "s8"],
            ],
            [["u8", "s8"], ["u8"], ["f32", "bf16", "f16", "s32", "u8", "s8"]],
            [
                ["f8_e5m2", "f8_e4m3"],
                ["f8_e5m2", "f8_e4m3"],
                ["f32", "bf16", "f16", "f8_e5m2", "f8_e4m3"],
            ],
        ]

        def is_int_type(t):
            return t in ["u4", "s4", "u8", "s8", "s32"]

        def get_accumulator(wei):
            if is_int_type(wei):
                return "s32"
            if wei == "f64":
                return "f64"
            return "f32"

        def get_fpmath_modes(src, wei, dst):
            src, wei, dst = Types.Type.wildcard_match(src, wei, dst)
            if get_accumulator(wei) == "f32":
                ret = [""]
                if "f32" in [src, wei]:
                    ret.append("(tf32)")
                if "f32" in [src, wei] and not "f16" in [src, wei]:
                    ret.append("(bf16)")
                if "f32" in [src, wei] and not "bf16" in [src, wei]:
                    ret.append("(f16)")
                return ret
            if (
                get_accumulator(wei) == "s32"
                and not is_int_type(dst)
                and not is_int_type(src)
            ):
                ret = []
                if "f32" in [src, wei]:
                    ret.append("(strict:true)")
                    ret.append("(tf32:true)")
                if "f16" not in [src, wei]:
                    ret.append("(bf16:true)")
                if "bf16" not in [src, wei]:
                    ret.append("(f16:true)")
                return ret
            return [""]

        out = []
        for c in support_matrix:
            for src, wei, dst in itertools.product(c[0], c[1], c[2]):
                for math in get_fpmath_modes(src, wei, dst):
                    out.append(Types.Type(f"{src}:{wei}:{dst}{math}"))
        return out


# Kind represents problem parameters that do not make sense to consider
# in aggregate for optimization purposes as these features require significant
# changes within generated implementations or the implementation dispatching.
class Kind:
    def __init__(self, layout, type):
        self.layout = layout
        self.type = type

    def benchdnn_str(self):
        return f"{self.layout.benchdnn_str()} {self.type.benchdnn_str()}"


class Primitive:
    def __init__(self, kind, dims):
        self.kind: Kind = kind
        self.dims = dims

    def benchdnn_str(self):
        return f"{self.kind.benchdnn_str()} {self.dims}"

    ##### from older version #####
    @staticmethod
    def from_repro(repro_line):
        t = Types.Type("f32:f32:f32")
        l = Layouts.Layout("any:any:any")
        dims = Dims([], 0, 0, 0)
        for arg in repro_line.split(" "):
            if arg.startswith("--dt="):
                t = Types.Type(arg.split("=")[1])
            elif arg.startswith("--attr-fpmath="):
                t.mode = arg.split("=")[1]
            elif arg.startswith("--stag="):
                l.A = arg.split("=")[1]
            elif arg.startswith("--wtag="):
                l.B = arg.split("=")[1]
            elif arg.startswith("--dtag="):
                l.C = arg.split("=")[1]
            elif not arg.startswith("--"):
                """ ???? dims parsing with correctness check. Buggy ????
                argsA, argsB = [a.split("x") for a in arg.split(":")];
                DebugPrint(f"argsA = {argsA} argsB = {argsB}")
                if len(argsA) == len(argsB) and len(argsA) >= 2:
                    raise RuntimeError(f"Invalid Matrix Dimension {arg}")
                dims.b = [int(x) for x in argsA[:-2]]
                if dims.b != [int(x) for x in argsB[:-2]]:
                    raise RuntimeError(f"A and B batch dimensions {argsA[:-2]} and {argsB[:-2]} do not match")
                dims.m = int(argsA[-2])
                dims.k = int(arg[-1])
                if dims.k != int(argsB[-2]):
                    raise RuntimeError(f"A and B k dimensions {argsA[-1]} and {argsB[-2]} do not match")
                dims.n = int(argsB[-1])
                """
                # w/o check
                dims.m, dims.k, dims.n = arg.split("x")
                dims.k = dims.k.split(":")[0]
                """
                """
        return Primitive(Kind(l, t), dims)


