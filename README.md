# Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN)[]({#mainpage})
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![Technical Preview](https://img.shields.io/badge/version-technical_preview-orange.svg)

Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) is an open source performance library for Deep Learning (DL) applications intended for acceleration of DL frameworks on Intel® Architecture. Intel MKL-DNN includes highly vectorized and threaded building blocks for implementation of convolutional neural networks (CNN). We created this project to help DL community innovate on Intel processors.

Intel MKL-DNN functionality shares implementation with [Intel® Math Kernel Library (Intel® MKL)](https://software.intel.com/en-us/intel-mkl), but is not API compatible with Intel MKL 2017. We will be looking into ways to converge API in future releases of Intel MKL.

This release is a technical preview with functionality limited to AlexNet and VGG topologies forward path. While this library is in technical preview phase it's API may change without considerations of backward compatibility.

## License
Intel MKL-DNN is licensed under [Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Support
Please report issues and suggestions via [GitHub issues](https://github.com/01org/mkl-dnn/issues) or start a topic on [Intel MKL forum](https://software.intel.com/en-us/forums/intel-math-kernel-library).

## How to Contribute
We welcome community contributions to Intel MKL-DNN. If you have an idea how to improve the product:
* Let us know about your proposal via [GitHub issues](https://github.com/01org/mkl-dnn/issues).
* Make sure you can build the product and run all the examples with your patch
* In case of a larger feature, create a test
* Submit a [pull request](https://github.com/01org/mkl-dnn/pulls)

We will review your contribution and, if any additional fixes or modifications are necessary, may give some feedback to guide you. When accepted, your pull request will be merged into our internal and GitHub repositories.

## System Requirements
Intel MKL-DNN supports Intel® 64 architecture processors and is optimized for
* Intel® Xeon® processor E5-xxxx v3 (codename Haswell)
* Intel® Xeon® processor E5-xxxx v4 (codename Broadwell)

Other processors and IA-32 code will run unoptimized reference implementation.

The software was validated on RedHat* Enterprise Linux 7 with
* GNU* Compiler Collection 4.8
* Intel® C/C++ Compiler for Linux 16.0
The implementation relies on OpenMP SIMD extensions and we recommend using Intel compiler for the best performance results.

## Installation
<placeholder>
