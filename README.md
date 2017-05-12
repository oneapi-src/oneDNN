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
The latest version of Intel MKL-DNN reference manual is available 
[GitHub pages](http://01org.github.io/mkl-dnn/). Basic concepts are also
explained in the tutorial
* [Intel MKL-DNN: Part 1--Overview and Installation](https://software.intel.com/en-us/articles/intel-mkl-dnn-part-1-library-overview-and-installation)
* [Intel MKL-DNN: Part 2--Code Build and Walkthrough](https://software.intel.com/en-us/articles/intel-mkl-dnn-part-2-sample-code-build-and-walkthrough)

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
was built.
```
	lib/
	    libmkldnn.so     # Intel MKL-DNN dynamic library
# The following libraries are included only if Intel MKL-DNN is built with binary dependency.
	    libiomp5.so      # Intel OpenMP* runtime library
	    libmkl_gnu.so    # Intel MKL small library for GNU* OpenMP runtime
	    libmkl_intel.so  # Intel MKL small library for Intel(R) OpenMP runtime
	include/
	    mkldnn.h         # C header
	    mkldnn.hpp       # C++ header
	    mkldnn_types.h   # auxillary C header
```

Intel MKL-DNN uses OpenMP* for parallelism and requires an OpenMP runtime 
library to work. As different OpenMP runtimes may not be binary compatible
it's important to ensure that only one OpenMP runtime is used throughout the
application. Having more than one OpenMP runtime initialized may lead to
undefined behavior resulting in incorrect results or crashes.

Intel MKL-DNN library built with binary dependency will link against Intel OpenMP
runtime included with Intel MKL small libraries package. Intel OpenMP runtime
is binary compatible with GNU OpenMP and CLANG OpenMP runtimes and should
be used in the final application. Here are example linklines for GNU C++ compiler
and Intel C++ compiler.
```
	g++ -std=c++11 -fopenmp -Wl,--as-needed -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn -lmklml_intel -liomp5
```
```
	icpc -std=c++11 -qopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn -lmklml_intel
```
In `g++` example option `-Wl,--as-needed` forces linker to resolve OpenMP symbols
in Intel OpenMP runtime library.

Intel MKL-DNN library built standalone will use OpenMP runtime supplied by
the compiler, so as long as both the library and the application use the
same compiler correct OpenMP runtime will be used. 
```
	g++ -std=c++11 -fopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn
```
```
	icpc -std=c++11 -qopenmp -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn
```
