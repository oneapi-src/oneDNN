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

MKLURL="https://github.com/01org/mkl-dnn/releases/download/v0.7/mklml_lnx_2017.0.3.20170424.tgz"

DST=`dirname $0`/../external
DST=`readlink -f $DST`
mkdir -p $DST
wget --no-check-certificate -P $DST $MKLURL
tar -xzf $DST/mklml_lnx*.tgz -C $DST

echo "Downloaded and unpacked Intel(R) MKL libraries for machine learning to $DST"
