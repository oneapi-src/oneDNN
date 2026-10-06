#!/bin/bash
#===============================================================================
# Copyright 2026 Intel Corporation
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
#===============================================================================


usage() {
    echo "Upstream all oneDNN commits to gemmstone since root_commit"
    echo "  Usage:    upstream_patch.sh <dnnl_dir> <root_commit>"
    echo "     or:    upstream_patch.sh --continue"
    echo "Options:"
    echo "    --continue    merge remaining patch files after resolving conflict"
    exit $1
}

error() {
    echo $1
    exit 1
}

gemmstone_root=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/..

if [ "$1" = "--continue" ]; then
   patch_files=$(ls ${gemmstone_root}/*.patch)
elif [ "$#" -eq 2 ]; then
    gemmstone_src=$1/src/gpu/intel/gemm/jit
    if [ ! -d ${gemmstone_src} ]; then
        echo "Error: Directory ${gemmstone_src} does not exist"
        usage 1
    fi
    commit_root=$2
    upstream_dirs="include generator generator_dsl dsl"
    patch_files=$(git -C "${gemmstone_src}" format-patch -o ${gemmstone_root} $commit_root -- ${upstream_dirs})
else
    echo "Error: Incorrect arguments supplied"
    usage 1
fi

for patch_file in ${patch_files}; do
    echo "Applying ${patch_file}"
    sed -i "s/src\/gpu\/intel\/gemm\/jit\///g" ${patch_file}
    sed -i "s/intel\/gemm\/jit\///g" ${patch_file}
    sed -i "s/gemm\/jit\///g" ${patch_file}
    sed -i "s/jit\///g" ${patch_file}
    git -C "${gemmstone_root}" am ${patch_file} || error "Commit ${commit_id} failed to apply, consider using git apply --reject and then manually merge .rej files"
    msg=$(git log -1 --pretty=format:"%s" | sed "s/.*gemmstone: //g")
    msg="$(echo ${msg} | sed 's/xe: //g')"
    msg="$(echo ${msg} | sed 's/gemm: //g')"
    msg="$(echo ${msg} | sed 's/jit: //g')"
    git commit --amend -m "${msg}" || error "Failed to update ${commit_id} message"
    rm ${patch_file}
done;
