# Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![v0.9 beta](https://img.shields.io/badge/v0.9-beta-orange.svg)

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an
open source performance library for Deep Learning (DL) applications intended
for acceleration of DL frameworks on Intel(R) architecture. Intel(R) MKL-DNN
includes highly vectorized and threaded building blocks for implementation of
convolutional neural networks (CNN) with C and C++ interfaces. We created this
project to enable the DL community to innovate on Intel(R) processors.

Intel MKL-DNN includes functionality similar to [Intel(R) Math Kernel
Library (Intel(R) MKL) 2017](https://software.intel.com/en-us/intel-mkl), but is not
API compatible. We are investigating how to unify the APIs in future Intel MKL releases.

This release is a technical preview with functionality necessary to accelerate
bleeding edge image recognition topologies, including Cifar\*, AlexNet\*, VGG\*, 
GoogleNet\* and ResNet\*. 

## License
Intel MKL-DNN is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Documentation
The latest Intel MKL-DNN documentation is at [GitHub pages](http://01org.github.io/mkl-dnn/).

## Support
Please report issues and suggestions via
[GitHub issues](https://github.com/01org/mkl-dnn/issues) or start a topic on
[Intel MKL forum](https://software.intel.com/en-us/forums/intel-math-kernel-library).

## How to Contribute
We welcome community contributions to Intel MKL-DNN. If you have an idea how to improve the library:

* Share your proposal via
 [GitHub issues](https://github.com/01org/mkl-dnn/issues).

* Ensure you can build the product and run all the examples with your patch

* In the case of a larger feature, create a test

* Submit a [pull request](https://github.com/01org/mkl-dnn/pulls)

We will review your contribution and, if any additional fixes or modifications
are necessary, may provide feedback to guide you. When accepted, your pull
request will be merged into our internal and GitHub repositories.

## System Requirements
Intel MKL-DNN supports Intel(R) 64 architecture processors and is optimized for
* Intel Atom(R) processor with Intel(R) SSE4.1 support
* 4th, 5th, 6th and 7th generation Intel(R) Core processor
* Intel(R) Xeon(R) processor E5 v3 family (code named Haswell)
* Intel(R) Xeon(R) processor E5 v4 family (code named Broadwell)
* Intel(R) Xeon(R) Platinum processor family (code name Skylake)
* Intel(R) Xeon Phi(TM) product family x200 (code named Knights Landing)
* Future Intel(R) Xeon Phi(TM) processor (code named Knights Mill)

The software dependencies are:
* [Cmake](https://cmake.org/download/) 2.8.0 or later
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html#srcbin) 1.8.5 or later
* C++ compiler with C++11 standard support

The software was validated on RedHat\* Enterprise Linux 7 with
* GNU\* Compiler Collection 4.8
* GNU\* Compiler Collection 6.1
* Clang\* 3.8.0
* [Intel(R) C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  16.0 or later

The implementation uses OpenMP\* 4.0 SIMD extensions. We recommend using
Intel(R) Compiler for the best performance results.

## Installation
Download [Intel MKL-DNN source code](https://github.com/01org/mkl-dnn/archive/master.zip)
or clone the repository to your system

```
	git clone https://github.com/01org/mkl-dnn.git
```

Satisfy all hardware and software dependencies and ensure that the versions
are correct before installing. Intel MKL-DNN can take advantage of optimized
matrix-matrix multiplication (GEMM) function from Intel MKL. The dynamic
library with this functionality is included in the repository. If you choose 
to build Intel MKL-DNN with binary dependency download Intel MKL small
libraries first using provided script

```
	cd scripts && ./prepare_mkl.sh && cd ..
```

or download manually and unpack it to the `external` directory in the 
repository root.

Intel MKL-DNN uses a CMake-based build system

```
	mkdir -p build && cd build && cmake .. && make
```

Intel MKL-DNN includes unit tests implemented using the googletest framework. To validate your build, run:

```
	make test
```

Documentation is provided inline and can be generated in HTML format with Doxygen:

```
	make doc
```

Documentation will reside in `build/reference/html` folder.

Finally,
```
	make install
```
will place the  header files, libraries and documentation in `/usr/local`. To change
the installation path, use the option `-DCMAKE_INSTALL_PREFIX=<prefix>` when invoking CMake.
