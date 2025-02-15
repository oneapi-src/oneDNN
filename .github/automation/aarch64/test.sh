#! /bin/bash

# *******************************************************************************
# Copyright 2024-2025 Arm Limited and affiliates.
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

# Test oneDNN for aarch64.

set -o errexit -o pipefail -o noclobber

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Defines MP, CC, CXX and OS.
source ${SCRIPT_DIR}/common.sh

# AArch64 does not officially support graph for now.
SKIPPED_GRAPH_TEST_FAILURES="test_graph_unit_dnnl_sdp_decomp_cpu"
SKIPPED_GRAPH_TEST_FAILURES+="|test_graph_unit_dnnl_mqa_decomp_cpu"

# described in issue: https://github.com/oneapi-src/oneDNN/issues/2175
SKIPPED_TEST_FAILURES="test_benchdnn_modeC_matmul_multidims_cpu"

#  We currently have some OS and config specific test failures.
if [[ "$OS" == "Linux" ]]; then
    if [[ "$CMAKE_BUILD_TYPE" == "Debug" ]]; then
        # as test_matmul is time consuming , we only run it in release mode to save time.
        SKIPPED_TEST_FAILURES+="|test_matmul"
    fi
    SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_binary_ci_cpu"
    SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_binary_different_dt_ci_cpu"

    SKIPPED_GRAPH_TEST_FAILURES+="|test_benchdnn_modeC_graph_ci_cpu"
    SKIPPED_GRAPH_TEST_FAILURES+="|cpu-graph-gqa-cpp"
    SKIPPED_GRAPH_TEST_FAILURES+="|cpu-graph-mqa-cpp"
    SKIPPED_GRAPH_TEST_FAILURES+="|cpu-graph-sdpa-cpp"
    SKIPPED_GRAPH_TEST_FAILURES+="|cpu-graph-sdpa-stacked-qkv-cpp"
    SKIPPED_GRAPH_TEST_FAILURES+="|test_graph_unit_dnnl_large_partition_cpu"
fi

# Nightly failures
SKIPPED_NIGHTLY_TEST_FAILURES="test_benchdnn_modeC_bnorm_all_blocked_cpu"
SKIPPED_NIGHTLY_TEST_FAILURES+="|test_benchdnn_modeC_bnorm_regressions_cpu"
SKIPPED_NIGHTLY_TEST_FAILURES+="|test_benchdnn_modeC_conv_int8_cpu"
SKIPPED_NIGHTLY_TEST_FAILURES+="|test_benchdnn_modeC_graph_fusions_cpu"
SKIPPED_NIGHTLY_TEST_FAILURES+="|test_benchdnn_modeC_matmul_sparse_gpu_cpu"
SKIPPED_NIGHTLY_TEST_FAILURES+="|test_benchdnn_modeC_reorder_all_cpu"

# * c7g failures. TODO: scope these to c7g only. Better yet, fix them.
SKIPPED_NIGHTLY_TEST_FAILURES+="|test_benchdnn_modeC_binary_all_cpu"
SKIPPED_NIGHTLY_TEST_FAILURES+="|test_benchdnn_modeC_graph_int8_cpu"

SKIPPED_TEST_FAILURES+="|${SKIPPED_GRAPH_TEST_FAILURES}|${SKIPPED_NIGHTLY_TEST_FAILURES}"

# Sequential (probably macOS) builds should use num proc parallelism.
if [[ "$ONEDNN_THREADING" == "SEQ" ]]; then
    export CTEST_PARALLEL_LEVEL=""
fi

set -x
ctest --no-tests=error --output-on-failure -E "$SKIPPED_TEST_FAILURES"
set +x
