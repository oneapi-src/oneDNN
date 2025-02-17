#! /bin/bash

# *******************************************************************************
# Copyright 2020-2025 Arm Limited and affiliates.
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
source ${SCRIPT_DIR}/common.sh

ACL_BUILD_TYPE=${ACL_BUILD_TYPE:-"Release"}
ACL_ROOT_DIR=${ACL_ROOT_DIR:-"${PWD}/ComputeLibrary"}
ACL_REPO="https://github.com/ARM-software/ComputeLibrary.git"

if [[ "$ACL_THREADING" == "OMP" ]]; then
    ACL_OPENMP=1
elif [[ "$ACL_THREADING" == "SEQ" ]]; then
    ACL_OPENMP=0
fi

if [[ "$OS" == "Linux" ]]; then
    ACL_MULTI_ISA_SUPPORT=1
    if [[ "$ACL_THREADING" == "OMP" ]]; then
        ACL_OPENMP=1
    elif [[ "$ACL_THREADING" == "SEQ" ]]; then
        ACL_OPENMP=0
    fi
    ACL_OS="linux"
elif [[ "$OS" == "Darwin" ]]; then
    ACL_MULTI_ISA_SUPPORT=0
    ACL_OPENMP=0
    ACL_OS="macos"
else
    echo "Unknown OS: $OS"
    exit 1
fi

if [[ "$ACL_BUILD_TYPE" == "Release" ]]; then
    ACL_DEBUG=0
elif [[ "$ACL_BUILD_TYPE" == "Debug" ]]; then
    ACL_DEBUG=1
else
    echo "Unknown build config: $ACL_BUILD_TYPE"
    exit 1
fi

if [[ "$ACL_ACTION" == "clone" ]]; then
    set -x
    git clone --branch $ACL_VERSION --depth 1 $ACL_REPO $ACL_ROOT_DIR
    set +x
elif [[ "$ACL_ACTION" == "build" ]]; then
    set -x
    cd $ACL_ROOT_DIR
    set -x
    scons $MP Werror=0 debug=$ACL_DEBUG neon=1 opencl=0 embed_kernels=0 \
        os=$ACL_OS arch=armv8.2-a build=native multi_isa=$ACL_MULTI_ISA_SUPPORT \
        fixed_format_kernels=1 cppthreads=0 openmp=$ACL_OPENMP examples=0 \
        validation_tests=0
    set +x
else
    echo "Unknown action: $ACL_ACTION"
    exit 1
fi
