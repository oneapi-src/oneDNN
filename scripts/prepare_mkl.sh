#!/bin/sh
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

MKLURLROOT="https://github.com/01org/mkl-dnn/releases/download/v0.11/"
MKLVERSION="2018.0.1.20171007"

os=`uname`
if [ "$os" == "Linux" ]; then
  MKLPACKAGE="mklml_lnx_${MKLVERSION}.tgz"
elif [ "$os" == "Darwin" ]; then
  MKLPACKAGE="mklml_mac_${MKLVERSION}.tgz"
else
  echo "Cannot identify operating system. Try downloading package manually."
  exit 1
fi

MKLURL=${MKLURLROOT}${MKLPACKAGE}
DST=`dirname $0`/../external
mkdir -p $DST
DST=`cd $DST;pwd`

curl -L -o "${DST}/${MKLPACKAGE}" "$MKLURL"
if [ \! $? ]; then
  echo "Download from $MKLURL to $DST failed"
  exit 1
fi

tar -xzf "$DST/${MKLPACKAGE}" -C $DST
echo "Downloaded and unpacked Intel(R) MKL small libraries to $DST"
