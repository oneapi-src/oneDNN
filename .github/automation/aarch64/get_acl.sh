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

set -o errexit -o pipefail -o noclobber

WORKSPACE=${GITHUB_WORKSPACE:-$(pwd)}
echo "github workspace $GITHUB_WORKSPACE"

os_type=$(uname)

ACL_WITH_ASSERTS=${ACL_WITH_ASSERTS:-0}
ACL_VERSION=${ACL_VERSION:-v24.08.1}

if [[ "$os_type" == "Linux" ]]; then
  echo "This machine is running Linux"
  ARCHIVE="arm_compute-${ACL_VERSION}-linux-aarch64-cpu-bin.tar.gz"
elif [[ "$os_type" == "Darwin" ]]; then
  echo "This machine is running macOS"
  ARCHIVE="arm_compute-${ACL_VERSION}-macos-aarch64-cpu-bin.tar.gz"
else
  echo "Unknown OS: $os_type"
  exit 1
fi

# Set version and root directory
export ACL_ROOT_DIR="${WORKSPACE}/ComputeLibrary"

echo "ACL_VERSION: ${ACL_VERSION}"
echo "ACL_DIR_NAME: ${ACL_DIR_NAME}"
echo "ACL_ROOT_DIR: ${ACL_ROOT_DIR}"
echo "ACL_WITH_ASSERTS: ${ACL_WITH_ASSERTS}"

# Download the specified Compute Library version
if [[ ! -f $ARCHIVE ]]; then
  ACL_URL="https://github.com/ARM-software/ComputeLibrary/releases/download/${ACL_VERSION}/${ARCHIVE}"
  echo "Downloading ACL from ${ACL_URL}"
  wget ${ACL_URL}
else
  echo "$ARCHIVE already exists, skipping download."
fi

# Function to find the appropriate lib directory
find_acl_lib_dir() {
  local dirs=("$ACL_ROOT_DIR"/lib/*/)
  local selected_dir=""

  # Select directory based on build type
  for dir in "${dirs[@]}"; do
    if [[ $ACL_WITH_ASSERTS == 1 ]]; then
      [[ "$dir" == *"-asserts/" ]] && selected_dir="$dir" && break
    else
      [[ "$dir" != *"-asserts/" ]] && selected_dir="$dir" && break
    fi
  done

  # Return result or exit if not found
  if [[ -z "$selected_dir" ]]; then
    echo "No matching ACL lib directory found."
    exit 1
  else
    echo "$selected_dir"
  fi
}

# Extract the tarball if not already extracted
if [[ ! -d $ACL_ROOT_DIR ]]; then
  mkdir -p $ACL_ROOT_DIR
  tar -xzvf "${ARCHIVE}" -C $ACL_ROOT_DIR --strip-components=1 >/dev/null 2>&1
else
  echo "$ACL_ROOT_DIR directory already exists, skipping extraction."
fi

# Find the ACL library directory
ACL_LIB_DIR=$(find_acl_lib_dir)
echo "Using ACL lib from ${ACL_LIB_DIR}"
echo "cp contents from ${ACL_LIB_DIR} to ${ACL_ROOT_DIR}/lib"
cp -rf "$ACL_LIB_DIR"* "$ACL_ROOT_DIR/lib/"

echo "${ACL_VERSION}" >"${ACL_ROOT_DIR}/arm_compute/arm_compute_version.embed"
