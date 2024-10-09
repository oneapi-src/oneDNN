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

# Function to clone and install Eigen
install_eigen() {
  local eigen_repo="https://gitlab.com/libeigen/eigen.git"
  local eigen_commit="aa6964bf3a34fd607837dd8123bc42465185c4f8"
  
  echo "Cloning Eigen repository..."
  if ! git clone "$eigen_repo" --depth 1; then
    echo "Error: Failed to clone Eigen repository."
    return 1
  fi

  cd eigen 

  if ! git fetch --depth 1 origin "$eigen_commit" || ! git checkout "$eigen_commit"; then
    echo "Error: Failed to fetch or checkout commit."
    return 1
  fi

  cd ..

  mkdir -p eigen-build && cd eigen-build 
  
  echo "EIGEN_INSTALL_PATH: ${EIGEN_INSTALL_PATH}"
  echo "EIGEN_PATH: $EIGEN_PATH"
  if ! cmake -DCMAKE_INSTALL_PREFIX="${EIGEN_INSTALL_PATH}" "$EIGEN_PATH"; then
    echo "Error: CMake configuration failed."
    return 1
  fi

  # Build and install Eigen
  echo "Building and installing Eigen..."
  if ! make -s -j install; then
    echo "Error: Build or installation failed."
    return 1
  fi

  echo "Eigen installed successfully!"
}

# Defines MP, CC, CXX and OS.
source ${SCRIPT_DIR}/common_aarch64.sh

export ACL_ROOT_DIR=${ACL_ROOT_DIR:-"${PWD}/ComputeLibrary"}

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-"Release"}
ONEDNN_THREADING=${ONEDNN_THREADING:-"SEQ"}

ONEDNN_TEST_SET=SMOKE

# ACL is not built with OMP on macOS.
if [[ "$OS" == "Darwin" ]]; then
    ONEDNN_THREADING=SEQ
fi

CMAKE_OPTIONS="-Bbuild -S. \
    -DDNNL_AARCH64_USE_ACL=ON \
    -DONEDNN_BUILD_GRAPH=0 \
    -DONEDNN_WERROR=ON \
    -DDNNL_BUILD_FOR_CI=ON \
    -DONEDNN_TEST_SET=${ONEDNN_TEST_SET} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
"

if [[ "$ONEDNN_THREADING" == 'TP' ]]; then
  # install eigen
  export EIGEN_INSTALL_PATH="$WORKSPACE/eigen-build"
  export EIGEN_PATH="${WORKSPACE}/eigen"
  install_eigen

  CMAKE_OPTIONS="${CMAKE_OPTIONS} -DDNNL_CPU_RUNTIME=THREADPOOL \
      -D_DNNL_TEST_THREADPOOL_IMPL=EIGEN -DEigen3_DIR=${EIGEN_INSTALL_PATH}/share/eigen3/cmake"
elif [[ "$ONEDNN_THREADING" == "OMP" || "$ONEDNN_THREADING" == "SEQ" ]]; then
  CMAKE_OPTIONS="${CMAKE_OPTIONS} -DDNNL_CPU_RUNTIME=${ONEDNN_THREADING}"
else
  echo "Only OMP, TP, SEQ schedulers supported, $ONEDNN_THREADING requested"
  exit 1
fi

echo "eigen tasks completed."
echo "compile oneDNN......"
set -x
cd $WORKSPACE
cmake ${CMAKE_OPTIONS}
cmake --build build $MP
set +x
