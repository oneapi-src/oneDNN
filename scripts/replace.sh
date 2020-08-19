#!/bin/bash
#===============================================================================
# Copyright 2019-2020 FUJITSU LIMITED
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

# This script modify cmake output files under cross compile environment.
#
# If you want to build dnnl_aarch64 and run on x64 linux,
# do the following commands.
#
# export CC=/usr/bin/aarch64-linux-gnu-gcc
# export CXX=/usr/bin/aarch64-linux-gnu-g++
# export LD_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib
# cd dnnl_aarch64/mkl-dnn
# mkdir build
# cd build
# cmake -DCMAKE_BUILD_TYPE=Debug ..
# ../../tools/replace.sh
# make -j28
# cd tests/gtests
# qemu-aarch64 ./test_reorder


list=`find . \( -name "flags.make" -o -name "link.txt" \)`

for i in ${list} ; do
    cat ${i} | sed -e "s/\-mcpu\=native/\-march\=armv8-a/" | sed -e "s/\-mcpu\=native/\-march\=armv8-a/" | sed -e "s/\-mtune\=native//" | sed -e "s/\-msse4\.1//" > hogefugafuga
#    cat ${i} | sed -e "s/\-march\=native/\-march\=armv8-a/" | sed -e "s/\-mcpu\=native/\-march\=armv8-a/" | sed -e "s/\-mtune\=native//" > hogefugafuga
#    cat ${i} | sed -e "s/\-march\=native//" | sed -e "s/\-mcpu\=native//" | sed -e "s/\-mtune\=native//" > hogefugafuga
    mv hogefugafuga ${i}
done
