#! /bin/python3
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

import argparse
import os
from tempfile import NamedTemporaryFile

from matmul import sampler as matmul_sampler
from matmul import primitive as matmul

from utils import *

##### ???? - to utils ????
def log(output):
    print("synthdnn: " + output)

def error(output):
    print("synthdnn: error: " + output)
    exit(1)

def setup_matmul_subparser(subparsers):
    matmul_parser = subparsers.add_parser('matmul', help='call with -h for information')
    matmul_parser.add_argument("--subprogram_main", default=matmul_main, help=argparse.SUPPRESS)

    matmul_parser.add_argument("-b", "--batch-file", default=None, help="batch file used for storing the sample")

    # Interface with benchdnn
    matmul_parser.add_argument("benchdnn", nargs='?', help="path to benchdnn executable")
    matmul_parser.add_argument("--engine", default="cpu", help="engine used for benchdnn execution")
    matmul_parser.add_argument("--collect", default="corr", help="benchdnn collection type, can be one of [corr, perf, dry-run]")
    matmul_parser.add_argument("-n", "--name", default="", help="sample name")

    # Sampler Arguments
    matmul_parser.add_argument("-l", "--layouts", default="any:any:any",
                               help="stag:wtag:dtag, comma separated list of layouts or \"all\" for every supported layout")
    matmul_parser.add_argument("-m", "--iter-mode", default="zip", help ="iteration mode, must be one of zip or product")
    matmul_parser.add_argument("-r", "--region", default="(1,1024,1024):(1,32768,32768):(1,32,32)",
                               help="(m_min,n_min,k_min)-(m_max,n_max,k_max)/(m_align,n_align,k_align)")
    matmul_parser.add_argument("-s", "--samples", default=1000, help="number of samples to collect")
    matmul_parser.add_argument("-t", "--types", default="f32:f32:f32",
                               help="dt:dt:dt(optional fpmath-mode), comma separated list of type configurations or \"all\" for every supported type")
def matmul_main(args):
    batchFile = open(args.batch_file, "w+t") if args.batch_file != None else None
    if args.benchdnn != None and batchFile == None:
        batchFile = NamedTemporaryFile("w+t")

    if batchFile:
        log(f"generating batch file: {batchFile.name}")
        region = matmul_sampler.Region(args.region)
        #DebugPrint(f"region = {region}")
        types = matmul.Types(args.types)
        layouts = matmul.Layouts(args.layouts, region.ndims)
        samples= matmul_sampler.Sampler(int(args.samples), args.iter_mode,
                                        types, layouts, region)
        for s in samples:
            batchFile.write("--reset " + str(s) + "\n")
        batchFile.flush()
        log(f"generation complete")

    if args.benchdnn:
        if not os.path.exists(args.benchdnn):
            error(f"cannot execute {args.benchdnn}, no such file exists")

        if(args.collect == "corr"):
            benchdnn_args = f"--engine={args.engine} --matmul --mode-modifier=P"
        elif(args.collect == "perf"):
            benchdnn_args = f"--engine={args.engine} --matmul --mode=F --cold-cache=all --perf-template=sample,{args.name},%prb%,%0Gflops%,%0Gbw% --memory-kind=usm_device --attr-scratchpad=user"
            if args.name.find(',') != -1:
                error(f"sample name {args.name} contains invalid character: ,")
        elif(args.collect == "dry-run"):
            benchdnn_args = f"--engine={args.engine} --matmul --mode=L"
        else:
            error(f"unknown collection method {args.collect}")

        cmd = f"{args.benchdnn} {benchdnn_args} --batch={batchFile.name}"
        log(f"executing: {cmd}")
        ret = os.system(cmd)
        log("execution complete")
        if ret != 0:
            error(f"execution of {cmd} failed with return code {ret}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #DebugPrint("parser")
    subparsers = parser.add_subparsers(help='primitive targeted for data collection')
    #DebugPrint("subparser")
    setup_matmul_subparser(subparsers)
    args = parser.parse_args()
    #DebugPrint(f"args: {args}")
    args.subprogram_main(args)
