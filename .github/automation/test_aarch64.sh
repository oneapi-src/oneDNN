#! /bin/bash

# *******************************************************************************
# Copyright 2024 Arm Limited and affiliates.
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
source ${SCRIPT_DIR}/common_aarch64.sh

# Skip tests for certain config to preserve resources, while maintaining 
# coverage. Skip: 
# (SEQ,CLANG)
# (OMP,CLANG,DEBUG)
SKIP_TESTS=0
if [[ "$OS" == "Linux" ]]; then
    if [[ "$ONEDNN_THREADING" == "SEQ" ]]; then
        if [[ "$BUILD_TOOLSET" == "clang" ]]; then
            SKIP_TESTS=1
        fi
    elif [[ "$ONEDNN_THREADING" == "OMP" ]]; then 
        if [[ "$BUILD_TOOLSET" == "clang" ]]; then
            if [[ "$CMAKE_BUILD_TYPE" == "Debug" ]]; then
                SKIP_TESTS=1
            fi
        fi
    fi
fi

if [[ $SKIP_TESTS == 1 ]]; then
    echo "Skipping tests for this configuration: $OS $ONEDNN_THREADING $BUILD_TOOLSET".
    exit 0
fi

#  We currently have some OS and config specific test failures.
if [[ "$OS" == "Linux" ]]; then
    if [[ "$CMAKE_BUILD_TYPE" == "Debug" ]]; then
        SKIPPED_TEST_FAILURES="cpu-primitives-deconvolution-cpp"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_lnorm_smoke_cpu"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_brgemm_smoke_cpu"
        SKIPPED_TEST_FAILURES+="|cpu-primitives-matmul-cpp"
        SKIPPED_TEST_FAILURES+="|test_convolution_backward_weights_f32"
        SKIPPED_TEST_FAILURES+="|test_matmul"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_conv_smoke_cpu"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_deconv_smoke_cpu"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_matmul_smoke_cpu"
    elif [[ "$CMAKE_BUILD_TYPE" == "Release" ]]; then
        SKIPPED_TEST_FAILURES="cpu-primitives-deconvolution-cpp"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_lnorm_smoke_cpu"
    fi
elif [[ "$OS" == "Darwin" ]]; then
    if [[ "$CMAKE_BUILD_TYPE" == "Debug" ]]; then
        SKIPPED_TEST_FAILURES="cpu-primitives-deconvolution-cpp"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_lnorm_smoke_cpu"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_brgemm_smoke_cpu"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_brgemm_ci_cpu"
    elif [[ "$CMAKE_BUILD_TYPE" == "Release" ]]; then
        SKIPPED_TEST_FAILURES="cpu-primitives-deconvolution-cpp"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_lnorm_smoke_cpu"
        SKIPPED_TEST_FAILURES+="|test_benchdnn_modeC_lnorm_ci_cpu"
    fi
fi

if [[ "$OS" == "Darwin" ]]; then
    # Since macos does not build with OMP, we can use multiple ctest threads.
    CTEST_MP=$MP
elif [[ "$OS" == "Linux" ]]; then
    if [[ "$ONEDNN_THREADING" == "OMP" ]]; then
        # OMP is already multi-threaded. Let's not oversubscribe.
        CTEST_MP=-j2
    elif [[ "$ONEDNN_THREADING" == "SEQ" ]]; then
        CTEST_MP=$MP
    fi
fi

set -x
ctest $CTEST_MP --no-tests=error --verbose --output-on-failure -E "$SKIPPED_TEST_FAILURES"
set +x
