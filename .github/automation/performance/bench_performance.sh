#! /bin/bash

# *******************************************************************************
# Copyright 2025 Arm Limited and affiliates.
# SPDX-License-Identifier: Apache-2.0
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
# *******************************************************************************

# Usage: bash bench_performance.sh {baseline_benchdnn_executable} {benchdnn_executable} {baseline_results_file} {new_results_file}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

for i in {1..5}
do
    $1 --matmul --mode=P --perf-template=%prb%,%-time% --batch=${SCRIPT_DIR}/inputs/matmul >> $3
    $2 --matmul --mode=P --perf-template=%prb%,%-time% --batch=${SCRIPT_DIR}/inputs/matmul >> $4
    $1 --conv --mode=P --perf-template=%prb%,%-time% --batch=${SCRIPT_DIR}/inputs/conv >> $3
    $2 --conv --mode=P --perf-template=%prb%,%-time% --batch=${SCRIPT_DIR}/inputs/conv >> $4
done
