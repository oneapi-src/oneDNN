#!/bin/sh

MKLURL="https://github.com/intelcaffe/caffe/releases/download/self_contained_BU1/mklml_lnx_2017.0.b1.20160513.1.tgz"

DST=`dirname $0`/../external
DST=`readlink -f $DST`
mkdir -p $DST
wget --no-check-certificate -P $DST $MKLURL
tar -xzf $DST/mklml_lnx*.tgz -C $DST

echo "Downloaded and unpacked MKL libraries for machine learning to $DST"

