# Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![v0.12 beta](https://img.shields.io/badge/v0.12-beta-orange.svg)

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an
open source performance library for Deep Learning (DL) applications intended
for acceleration of DL frameworks on Intel(R) architecture. Intel(R) MKL-DNN
includes highly vectorized and threaded building blocks for implementation of
convolutional neural networks (CNN) with C and C++ interfaces. We created this
project to enable the DL community to innovate on Intel(R) processors.

Intel MKL-DNN includes functionality similar to [Intel(R) Math Kernel
Library (Intel(R) MKL) 2017](https://software.intel.com/en-us/intel-mkl), 
but is not API compatible. We are investigating how to unify the APIs in 
future Intel MKL releases.

This release contains a range of performance critical functions used in modern
image recognition (AlexNet, VGG, GoogleNet\*, ResNet), semantic
segmentation (FCNs, SegNet), and object detection topologies (SSD,
Fast/Faster R-CNN) optimized for wide range of Intel processors.

**WARNING** Functionality related to `s16` data type is experimental 
and might change without prior notification in future releases.

## License
Intel MKL-DNN is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Documentation
The latest version of Intel MKL-DNN reference manual is available 
[GitHub pages](http://01org.github.io/mkl-dnn/). Basic concepts are also
explained in the tutorial
* [Intel MKL-DNN: Part 1--Overview and Installation](https://software.intel.com/en-us/articles/intel-mkl-dnn-part-1-library-overview-and-installation)
* [Intel MKL-DNN: Part 2--Code Build and Walkthrough](https://software.intel.com/en-us/articles/intel-mkl-dnn-part-2-sample-code-build-and-walkthrough)

## Support
Please submit your questions, feature requests and bug reports on
[GitHub issues](https://github.com/01org/mkl-dnn/issues) page.

## How to Contribute
We welcome community contributions to Intel MKL-DNN. If you have an idea how to improve the library:

* Share your proposal via
 [GitHub issues](https://github.com/01org/mkl-dnn/issues).

* Ensure you can build the product and run all the examples with your patch

* In the case of a larger feature, create a test

* Submit a [pull request](https://github.com/01org/mkl-dnn/pulls)

We will review your contribution and, if any additional fixes or modifications
are necessary, may provide feedback to guide you. When accepted, your pull
request will be merged the repository.

## System Requirements
Intel MKL-DNN supports Intel(R) 64 architecture and compatible architectures.
The library is optimized for the systems based on
* Intel Atom(R) processor with Intel(R) SSE4.1 support
* 4th, 5th, 6th and 7th generation Intel(R) Core processor
* Intel(R) Xeon(R) processor E5 v3 family (formerly Haswell)
* Intel Xeon processor E5 v4 family (formerly Broadwell)
* Intel Xeon Platinum processor family (formerly Skylake)
* Intel(R) Xeon Phi(TM) processor x200 product family (formerly Knights Landing)
* Intel Xeon Phi processor x205 product family (formerly Knights Mill)
and compatible processors.

The software dependencies are:
* [Cmake](https://cmake.org/download/) 2.8.0 or later
* [Doxygen](http://www.stack.nl/~dimitri/doxygen/download.html#srcbin) 1.8.5 or later
* C++ compiler with C++11 standard support

The software was validated on RedHat\* Enterprise Linux 7 with
* GNU\* Compiler Collection 4.8, 5.2, 6.1 and 7.2
* Clang\* 3.8.0
* [Intel(R) C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0 and 18.0

on Windows Server\* 2012 R2 with
* Microsoft\* Visual C++ 14.0 (Visual Studio 2015)
* [Intel(R) C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0 and 18.0

on macOS\* 10.13 (High Sierra) with
* Apple LLVM version 9.0.0 (XCode 9.0.0)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  18.0 (XCode 8.3.2)

The implementation uses OpenMP\* 4.0 SIMD extensions. We recommend using
Intel(R) Compiler for the best performance results.

## Installation
Download [Intel MKL-DNN source code](https://github.com/01org/mkl-dnn/archive/master.zip)
or clone the repository to your system

```
	git clone https://github.com/01org/mkl-dnn.git
```

Ensure that all software dependencies are in place and have at least minimal
supported version.

Intel MKL-DNN can take advantage of optimized
matrix-matrix multiplication (GEMM) function from Intel MKL. The dynamic
library with this functionality is included in the repository. If you choose 
to build Intel MKL-DNN with the binary dependency download Intel MKL small
libraries using provided script

```
	cd scripts && ./prepare_mkl.sh && cd ..
```

or manually from [GitHub release section](https://github.com/01org/mkl-dnn/releases)
and unpack it to the `external` directory in the repository root. 

You can choose to build Intel MKL-DNN without binary dependency. The resulting
version will be fully functional, however performance of certain convolution
shapes and sizes and inner product relying on SGEMM function may be suboptimal.

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

## Linking your application
Intel MKL-DNN include several header files providing C and C++ APIs for 
the functionality and several dynamic libraries depending on how Intel MKL-DNN
was built. Intel OpenMP runtime and Intel MKL small libraries are not installed
for standalone Intel MKL-DNN build.

|File                   | Description
|:---                   |:---
|lib/libmkldnn.so       | Intel MKL-DNN dynamic library
|lib/libiomp5.so        | Intel OpenMP* runtime library
|lib/libmklml_gnu.so    | Intel MKL small library for GNU* OpenMP runtime
|lib/libmklml_intel.so  | Intel MKL small library for Intel(R) OpenMP runtime
|include/mkldnn.h       | C header
|include/mkldnn.hpp     | C++ header
|include/mkldnn_types.h | auxillary C header

Intel MKL-DNN uses OpenMP* for parallelism and requires an OpenMP runtime 
library to work. As different OpenMP runtimes may not be binary compatible
it's important to ensure that only one OpenMP runtime is used throughout the
application. Having more than one OpenMP runtime initialized may lead to
undefined behavior resulting in incorrect results or crashes.

Intel MKL-DNN library built with binary dependency will link against Intel OpenMP
runtime included with Intel MKL small libraries package. Intel OpenMP runtime
is binary compatible with GNU OpenMP and CLANG OpenMP runtimes and is 
recommended for the best performance results. Here are example linklines for 
GNU C++ compiler and Intel C++ compiler.
```
	g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn -lmklml_intel -liomp5
```
```
	icpc -std=c++11 -qopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn -lmklml_intel
```
Using GNU compiler with `-fopenmp` and `-liomp5` options will link the 
application with both Intel and GNU OpenMP runtime libraries. This will lead
to undefined behavior of the application.

Intel MKL-DNN library built standalone will use OpenMP runtime supplied by
the compiler, so as long as both the library and the application use the
same compiler correct OpenMP runtime will be used. 
```
	g++ -std=c++11 -fopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn
```
```
	icpc -std=c++11 -qopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn
```

--------

[Legal Information](doc/legal_information.md)