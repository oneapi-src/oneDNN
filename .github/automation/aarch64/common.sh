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

# Common variables for aarch64 ci. Exports: 
# CC, CXX, OS

set -o errexit -o pipefail -o noclobber

export OS=$(uname)

# Num threads on system.
if [[ "$OS" == "Darwin" ]]; then
    export MP="-j$(sysctl -n hw.ncpu)"
elif [[ "$OS" == "Linux" ]]; then
    export MP="-j$(nproc)"
fi

if [[ "$BUILD_TOOLSET" == "gcc" ]]; then
    export CC=gcc-${GCC_VERSION}
    export CXX=g++-${GCC_VERSION}
elif [[ "$BUILD_TOOLSET" == "clang" ]]; then
    export CC=clang
    export CXX=clang++
fi

# Print every exported variable.
echo "OS: $OS"
echo "Toolset: $BUILD_TOOLSET"
echo "CC: $CC"
echo "CXX: $CXX"
