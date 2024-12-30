#! /bin/bash

# *******************************************************************************
# Copyright 2020-2024 Arm Limited and affiliates.
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

# Build ACL from github.

set -o errexit -o pipefail -o noclobber

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Defines MP, CC, CXX and OS.
source ${SCRIPT_DIR}/common_aarch64.sh

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-"Release"}
ACL_ROOT_DIR=${ACL_ROOT_DIR:-"${PWD}/ComputeLibrary"}
ACL_REPO="https://github.com/ARM-software/ComputeLibrary.git"

if [[ "$ACL_THREADING" == "OMP" ]]; then
    ACL_OPENMP=1
elif [[ "$ACL_THREADING" == "SEQ" ]]; then
    ACL_OPENMP=0
fi

if [[ "$ACL_ACTION" == "clone" ]]; then
    set -x
    git clone --branch $ACL_VERSION --depth 1 $ACL_REPO $ACL_ROOT_DIR
    set +x
elif [[ "$ACL_ACTION" == "configure" ]]; then
    set -x
    cmake \
    -S$ACL_ROOT_DIR -B$ACL_ROOT_DIR/build \
	-DARM_COMPUTE_OPENMP=$ACL_OPENMP \
	-DARM_COMPUTE_CPPTHREADS=0 \
	-DARM_COMPUTE_WERROR=0 \
	-DARM_COMPUTE_BUILD_EXAMPLES=1 \
	-DARM_COMPUTE_BUILD_TESTING=1 \
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
    set +x
elif [[ "$ACL_ACTION" == "build" ]]; then
    set -x
    cmake --build $ACL_ROOT_DIR/build 
    set +x
else
    echo "Unknown action: $ACL_ACTION"
    exit 1
fi
