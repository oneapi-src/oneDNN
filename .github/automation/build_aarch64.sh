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

# Build oneDNN for aarch64.

set -o errexit -o pipefail -o noclobber

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Defines MP, CC, CXX and OS.
source ${SCRIPT_DIR}/common_aarch64.sh

export ACL_ROOT_DIR=${ACL_ROOT_DIR:-"${PWD}/ComputeLibrary"}

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-"Release"}
ONEDNN_TEST_SET=SMOKE

# ACL is not built with OMP on macOS.
if [[ "$OS" == "Darwin" ]]; then
    ONEDNN_THREADING=SEQ
fi

set -x
cmake \
    -Bbuild -S. \
    -DDNNL_AARCH64_USE_ACL=ON \
    -DONEDNN_BUILD_GRAPH=0 \
    -DDNNL_CPU_RUNTIME=$ONEDNN_THREADING \
    -DONEDNN_WERROR=ON \
    -DDNNL_BUILD_FOR_CI=ON \
    -DONEDNN_TEST_SET=$ONEDNN_TEST_SET \
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE

cmake --build build $MP
set +x
