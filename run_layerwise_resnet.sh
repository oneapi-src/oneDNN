#!/bin/bash

set -x
ninja benchdnn
echo "################ BINARY ######################"
./tests/benchdnn/benchdnn --engine=gpu --binary --mode=p --batch=../tests/benchdnn/inputs/binary/shapes_resnet_50
echo "################ BATCH NORM ######################"
./tests/benchdnn/benchdnn --engine=gpu --bnorm --mode=p --batch=../tests/benchdnn/inputs/bnorm/shapes_resnet_50
echo "################ CONVOLUTION ######################"
./tests/benchdnn/benchdnn --engine=gpu --conv --mode=p --batch=../tests/benchdnn/inputs/conv/shapes_resnet_50
echo "################ ELTWISE ######################"
./tests/benchdnn/benchdnn --engine=gpu --eltwise --mode=p --batch=../tests/benchdnn/inputs/eltwise/shapes_resnet_50
echo "################ INNER PRODUCT ######################"
./tests/benchdnn/benchdnn --engine=gpu --ip --mode=p --batch=../tests/benchdnn/inputs/ip/shapes_resnet_50
echo "################ POOLING ######################"
./tests/benchdnn/benchdnn --engine=gpu --pool --mode=p --batch=../tests/benchdnn/inputs/pool/shapes_resnet_50
echo "################ SOFTMAX ######################"
./tests/benchdnn/benchdnn --engine=gpu --softmax --mode=p --batch=../tests/benchdnn/inputs/softmax/shapes_resnet_50
