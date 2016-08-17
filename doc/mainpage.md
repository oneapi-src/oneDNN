<-- @mainpage -->
# Developer Manual
Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) is an open source performance library for Deep Learning (DL) applications intended for acceleration of DL frameworks on Intel® architecture. Intel MKL-DNN includes highly vectorized and threaded building blocks for implementation of convolutional neural networks (CNN) with C and C++ interfaces. We created this project to help DL community innovate on Intel® processors.

The library supports the most commonly used primitives necessary to accelerate bleeding edge image recognition topologies, including AlexNet, VGG, GoogleNet and ResNet. The primitives include convolution, inner product, pooling, normalization and activation primitives with support for forward (scoring or inference) operations. Current release includes the following clasess of functions:
* Convolution: direct batched convolution
* Inner Product
* Pooling: maximum, minimum, average
* Normalization: local response normalization across channels (LRN), batch normalization
* Activation: rectified linear neuron activation (ReLU)
* Data manipulation: multi-dimensional transposition (conversion), split, concat, sum and scale.

Intel MKL DNN primitives implement a plain C application programming interface (API) that can be used in the existing C/C++ DNN frameworks, as well as in custom DNN applications.

## Programming Model


@subpage legal_information.md "Legal Information"