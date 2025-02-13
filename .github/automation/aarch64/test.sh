#!/usr/bin/env bash

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

export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-"Release"}

# Defines MP, CC, CXX and OS.
source ${SCRIPT_DIR}/common.sh

# Sequential (probably macOS) builds should use num proc parallelism.
if [[ "$ONEDNN_THREADING" == "SEQ" ]]; then
    export CTEST_PARALLEL_LEVEL=""
fi

set -x
ctest --no-tests=error --output-on-failure -E $("${SCRIPT_DIR}"/skipped-tests.sh)
set +x
