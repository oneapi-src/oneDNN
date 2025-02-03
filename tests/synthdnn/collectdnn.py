#! /bin/python3
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

from tempfile import NamedTemporaryFile
import os
import argparse
from generation import matmul
import itertools
import time


def log(output):
    print("collectdnn: " + output)


log("started")
parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--samples", default=10000, help="number of samples to collect"
)
parser.add_argument("-n", "--name", default="", help="sample name")
subparsers = parser.add_subparsers(help='primitive targeted for data collection')
parse_matmul = matmul.setup_parser(subparsers.add_parser('matmul', help='call with -h for information'))

args = parser.parse_args()
benchdnn = args.benchdnn
samples = args.sampler(args)
name = args.name
if name.find(',') != -1:
    print("Error: sample name " + name + " contains an invalid character: ,")
    exit(1)

batchFile = NamedTemporaryFile("w+t")
start_time = time.monotonic()
for s in samples:
    batchFile.write("--reset " + str(s) + "\n")
log(
    "batchfile "
    + batchFile.name
    + " generated in "
    + str(time.monotonic() - start_time)
    + " s"
)

batchFile.flush()
cmd = (
    benchdnn
    + " --engine=gpu --matmul -v2 --mode=F --cold-cache=all --perf-template=sample," + name + ",%prb%,%0Gflops%,%0Gbw% --memory-kind=usm_device --attr-scratchpad=user --batch="
    + batchFile.name
)

log("executing " + cmd)

start_time = time.monotonic()
bench_out = os.system(cmd)

log(
    "command execution completed in "
    + str(time.monotonic() - start_time)
    + " s"
)
