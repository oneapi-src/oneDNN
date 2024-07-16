#! /bin/python3
from random import randrange
from tempfile import NamedTemporaryFile
import os
import argparse

class Matmul:
    class Dims:
        def __init__(self, m, k, n):
            self.m = m
            self.k = k
            self.n = n
        def __str__(self):
            return f"{self.m}x{self.k}:{self.k}x{self.n}"

    class Layout:
        def __init__(self, A, B, C):
            self.A = A
            self.B = B
            self.C = C

        def __str__(self):
            return f"--stag={self.A} --wtag={self.B} --dtag={self.C}"

    class DataTypes:
        class Mode:
            def __init__(self, value):
                self.value = value
            def __str__(self):
                if self.value is None:
                    return ""
                else:
                    return f"--attr-fpmath={self.value}"

        def __init__(self, A, B, C, mode):
            self.A = A
            self.B = B
            self.C = C
            self.mode = self.Mode(mode)
        def __str__(self):
            return f"--dt={self.A}:{self.B}:{self.C} {self.mode}"


    def __init__(self, layout, data_types, dims):
        self.dims = dims
        self.layout = layout
        self.data_types = data_types
    def __str__(self):
        return f"{self.layout} {self.data_types} {self.dims}"

class sample:
    layouts = { # Matmul.Layout("ab", "ab", "ab"),
                Matmul.Layout("ab", "ba", "ab"),
                # Matmul.Layout("ba", "ab", "ab"),
                # Matmul.Layout("any", "any", "any")
               }
    types = { # Matmul.DataTypes("f32", "f32", "f32", None),
              # Matmul.DataTypes("f32", "f32", "f32", "tf32"),
              # Matmul.DataTypes("f32", "f32", "f32", "f16"),
              # Matmul.DataTypes("f32", "f32", "f32", "bf16"),
              # Matmul.DataTypes("bf16", "bf16", "bf16", None),
              # Matmul.DataTypes("f16", "f16", "f16", None),
              Matmul.DataTypes("f16", "u4", "f16", "f16:true"),
              Matmul.DataTypes("s8", "s8", "s8", None),
             }

parser = argparse.ArgumentParser()
parser.add_argument("benchdnn", help="path to benchdnn executable")
parser.add_argument("-s", "--samples", default=10000, help="number of samples to collect")
args = parser.parse_args()

benchdnn = args.benchdnn
samples = int(args.samples)
batchFile = NamedTemporaryFile('w+t')
for i in range(samples):
    m = 1;
    n = randrange(1, 1024) * 16
    k = randrange(256, 1024) * 16
    for l in sample.layouts:
        for t in sample.types:
            m1 = Matmul(l, t, Matmul.Dims(m,k,n))
            s = "--reset " + str(m1) + "\n"
            batchFile.write(s)

batchFile.flush()
cmd = benchdnn + " --engine=gpu --matmul --mode=F --perf-template=perf,%0Gbw%,%prb% --batch=" + batchFile.name
bench_out = os.popen(cmd)

print("plot, k, n, Bandwidth, plot_id")
while True:
    line = bench_out.readline()
    if not line:
        break;
    if not line.startswith("perf"):
        continue
    _, bandwidth, prb = line.strip().split(",")
    plot_id = ' '.join(prb.split(' ')[:-2])
    k, n = prb.split(":")[-1].split("x")
    print("plot, " + k + ", " + n + ", " + bandwidth  + ", " + plot_id)
