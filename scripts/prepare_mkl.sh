#!/bin/bash
#===============================================================================
# Copyright 2016-2017 Intel Corporation
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

OS=$(uname -s)
case $OS in
  Linux) platform=lnx;;
  Darwin) platform=mac;;
  Windows) platform=win;;
esac

MKL_VERSION="2018.0.1.20171007"
MKL_NAME=mklml
MKLURL="https://github.com/01org/mkl-dnn/releases/download/v0.11/${MKL_NAME}_${platform}_${MKL_VERSION}.tgz"

# MacOS readlink doesn't support -f option
DST=`dirname $0`/../
pushd $DST > /dev/null
DST=$(pwd)/external
popd > /dev/null

mkdir -p $DST

wget -P $DST $MKLURL
tar -xzf "$DST/${MKL_NAME}_${platform}_${MKL_VERSION}.tgz" -C $DST

echo "Downloaded and unpacked Intel(R) MKL small libraries to $DST"
