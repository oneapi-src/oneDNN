Deep Neural Network Library (DNNL)
==================================

> **Note**
>
> Starting with version 1.1 the library is renamed to DNNL. Please read
> [Intel MKL-DNN to DNNL Transition Guide](https://intel.github.io/mkl-dnn/dev_guide_transition_to_dnnl.html).

> **Note**
>
> Version 1.0 brings incompatible changes to the 0.20 version. Please read
> [Version 1.0 Transition Guide](https://intel.github.io/mkl-dnn/dev_guide_transition_to_v1.html).

Deep Neural Network Library (DNNL) is an
open-source performance library for deep learning applications. The library
includes basic building blocks for neural networks optimized
for Intel Architecture Processors and Intel Processor Graphics.

DNNL is intended for deep learning applications and framework
developers interested in improving application performance
on Intel CPUs and GPUs. Deep learning practitioners should use one of the
applications enabled with DNNL:
* [Apache\* MXNet](https://mxnet.apache.org)
* [BigDL](https://github.com/intel-analytics/BigDL)
* [Caffe\* Optimized for Intel Architecture](https://github.com/intel/caffe)
* [Chainer\*](https://chainer.org)
* [DeepLearning4J\*](https://deeplearning4j.org)
* [Intel Nervana Graph](https://github.com/NervanaSystems/ngraph)
* [MATLAB\* Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/)
* [Menoh\*](https://github.com/pfnet-research/menoh)
* [Microsoft\* Cognitive Toolkit (CNTK)](https://docs.microsoft.com/en-us/cognitive-toolkit)
* [ONNX Runtime](https://github.com/microsoft/onnxruntime)
* [OpenVINO(TM) toolkit](https://01.org/openvinotoolkit)
* [PaddlePaddle\*](http://www.paddlepaddle.org)
* [PyTorch\*](https://pytorch.org/)
* [Tensorflow\*](https://www.tensorflow.org)

# Documentation

* [Developer guide](https://intel.github.io/mkl-dnn) explains programming
model, supported functionality, details of primitives implementations and
includes annotated examples.
* [API reference](https://intel.github.io/mkl-dnn/modules.html) provides
comprehensive reference of the library API.

# Installation

Pre-built binaries for Linux\*, Windows\*, and macOS\* are available for download
in the [releases section](https://github.com/intel/mkl-dnn/releases). Package
names use the following convention:

| OS      | Package name
| :------ | :-----------
| Linux   | `dnnl_lnx_<version>_cpu_<cpu runtime>[_gpu_<gpu runtime>].tgz`
| Windows | `dnnl_win_<version>_cpu_<cpu runtime>[_gpu_<gpu runtime>].zip`
| macOS   | `dnnl_mac_<version>_cpu_<cpu runtime>.tgz`

Several packages are available for each operating system to ensure
interoperability with CPU or GPU runtime libraries used by the application.

| Configuration | Dependency
| :-------------| :---------
| `cpu_iomp`    | Intel OpenMP runtime
| `cpu_gomp`    | GNU\* OpenMP runtime
| `cpu_vcomp`   | Microsoft Visual C OpenMP runtime
| `cpu_tbb`     | Threading Building Blocks

The packages do not include library dependencies and these need to be resolved
in the application at build time. See the
[System Requirements](#system-requirements) section below and the
[Build Options](http://intel.github.io/mkl-dnn/dev_guide_build_options.html)
section in the [developer guide](http://intel.github.io/mkl-dnn) for more
details on CPU and GPU runtimes.

If the configuration you need is not available, you can
[build the library from source](http://intel.github.io/mkl-dnn/dev_guide_build.html).

# System Requirements

DNNL supports systems based on Intel 64 architecture or
compatible processors.

The library is optimized for the following CPUs:
* Intel Atom processor with Intel SSE4.1 support
* 4th, 5th, 6th, 7th, and 8th generation Intel Core(TM) processor
* Intel Xeon(R) processor E3, E5, and E7 family (formerly Sandy Bridge,
  Ivy Bridge, Haswell, and Broadwell)
* Intel Xeon Phi(TM) processor (formerly Knights Landing and Knights Mill)
* Intel Xeon Scalable processor (formerly Skylake and Cascade Lake)
* future Intel Xeon Scalable processor (code name Cooper Lake)

DNNL detects instruction set architecture (ISA) in the runtime and uses
just-in-time (JIT) code generation to deploy the code optimized
for the latest supported ISA.

> **WARNING**
>
> On macOS, applications that use DNNL may need to request special
> entitlements if they use the hardened runtime. See the
> [linking guide](https://intel.github.io/mkl-dnn/dev_guide_link.html)
> for more details.

The library is optimized for the following GPUs:
* Intel HD Graphics
* Intel UHD Graphics
* Intel Iris Plus Graphics

## Requirements for Building from Source

DNNL supports systems meeting the following requirements:
* Operating system with Intel 64 architecture support
* C++ compiler with C++11 standard support
* [CMake](https://cmake.org/download/) 2.8.11 or later
* [Doxygen](http://www.doxygen.nl/download.html#srcbin) 1.8.5 or later to build
documentation

Configurations of CPU and GPU engines may introduce additional build time
dependencies.

### CPU Engine

Intel Architecture Processors and compatible devices are supported by the
DNNL CPU engine. The CPU engine is built by default and cannot
be disabled at build time. The engine can be configured to use the OpenMP or
TBB threading runtime. The following additional requirements apply:
* OpenMP runtime requires C++ compiler with OpenMP 2.0 or later standard support
* TBB runtime requires
[Threading Building Blocks (TBB)](https://www.threadingbuildingblocks.org/)
2017 or later.

Some implementations rely on OpenMP 4.0 SIMD extensions, and we recommend using
the Intel C++ Compiler for the best performance results.

### GPU Engine

Intel Processor Graphics is supported by the DNNL GPU engine. The GPU
engine is disabled in the default build configuration. The following
additional requirements apply when GPU engine is enabled:
* OpenCL\* runtime library (OpenCL version 1.2 or later)
* OpenCL driver (with kernel language support for OpenCL C 2.0 or later)
  with Intel subgroups extension support

### Runtime Dependencies

When DNNL is built from source, the library runtime dependencies
and specific versions are defined by the build environment.

#### Linux

Common dependencies:
* System C/C++ runtime (libc.so, libstdc++.so)
* Dynamic Linking Library (libdl.so)
* C Math Library (libm.so)
* POSIX Threads Library (libpthread.so)

Runtime specific dependencies:

| Runtime configuration  | Compiler                      | Dependency
| :--------------------- | :---------------------------- | :---------
| `DNNL_CPU_RUNTIME=OMP` | GCC                           | GNU OpenMP runtime (libgomp.so)
| `DNNL_CPU_RUNTIME=OMP` | Intel C/C++ Compiler          | Intel OpenMP runtime (libiomp5.so)
| `DNNL_CPU_RUNTIME=OMP` | Clang                         | Intel OpenMP runtime (libiomp5.so)
| `DNNL_CPU_RUNTIME=TBB` | any                           | Threading Building Blocks (libtbb.so)
| `DNNL_GPU_RUNTIME=OCL` | any                           | OpenCL runtime (libOpenCL.so)

#### Windows

Common dependencies:
* Microsoft Visual C++ Redistributable (msvcrt.dll)

Runtime specific dependencies:

| Runtime configuration  | Compiler                      | Dependency
| :--------------------- | :---------------------------- | :---------
| `DNNL_CPU_RUNTIME=OMP` | Microsoft Visual C++ Compiler | No additional requirements
| `DNNL_CPU_RUNTIME=OMP` | Intel C/C++ Compiler          | Intel OpenMP runtime (iomp5.dll)
| `DNNL_CPU_RUNTIME=TBB` | any                           | Threading Building Blocks (tbb.dll)
| `DNNL_GPU_RUNTIME=OCL` | any                           | OpenCL runtime (OpenCL.dll)

#### macOS

Common dependencies:
* System C/C++ runtime (libc++.dylib, libSystem.dylib)

Runtime specific dependencies:

| Runtime configuration  | Compiler                      | Dependency
| :--------------------- | :---------------------------- | :---------
| `DNNL_CPU_RUNTIME=OMP` | Intel C/C++ Compiler          | Intel OpenMP runtime (libiomp5.dylib)
| `DNNL_CPU_RUNTIME=TBB` | any                           | Threading Building Blocks (libtbb.dylib)

### Validated Configurations

CPU engine was validated on RedHat\* Enterprise Linux 7 with
* GNU Compiler Collection 4.8, 5.4, 6.1, 7.2, and 8.1
* Clang\* 3.8.0
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0, 18.0, and 19.0

on Windows Server\* 2012 R2 with
* Microsoft Visual C++ 14.0 (Visual Studio 2015 Update 3)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0 and 19.0

on macOS 10.13 (High Sierra) with
* Apple LLVM version 9.2 (XCode 9.2)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  18.0 and 19.0

GPU engine was validated on Ubuntu\* 18.04 with
* GNU Compiler Collection 6.1 and 8.1
* Clang 3.8.1
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  19.0
* [Intel SDK for OpenCL applications](https://software.intel.com/en-us/intel-opencl) 2019 Update 3
* [Intel Graphics Compute Runtime for OpenCL](https://github.com/intel/compute-runtime/releases) 19.37.14191

on Windows Server 2019 with
* Microsoft Visual C++ 14.0 (Visual Studio 2015 Update 3)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  19.0
* [Intel SDK for OpenCL applications](https://software.intel.com/en-us/intel-opencl) 2019 Update 3
* [Intel Graphics - Windows 10 DCH Drivers](https://downloadcenter.intel.com/download/28783/Intel-Graphics-Windows-10-DCH-Drivers) 26.20.100.6709

## Requirements for Pre-built Binaries

See README included into corresponding binary package.

# Support

Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/intel/mkl-dnn/issues) page.

You may reach out to project maintainers privately at dnnl.maintainers@intel.com.

> **WARNING**
>
> The following functionality has preview status and might be changed without
> prior notification in future releases:
> * [Primitive cache](https://intel.github.io/mkl-dnn/dev_guide_primitive_cache.html)

# Contributing

We welcome community contributions to DNNL. If you have an idea on how
to improve the library:

* For changes impacting the public API, submit
  an [RFC pull request](CONTRIBUTING.md#RFC_pull_requests).
* Ensure that the changes are consistent with the
 [code contribution guidelines](CONTRIBUTING.md#code_contribution_guidelines)
 and [coding style](CONTRIBUTING.md#coding_style).
* Ensure that you can build the product and run all the examples with your
  patch.
* Submit a [pull request](https://github.com/intel/mkl-dnn/pulls).

For additional details, see [contribution guidelines](CONTRIBUTING.md).

This project is intended to be a safe, welcoming space for collaboration, and
contributors are expected to adhere to the
[Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

# License

DNNL is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).  This
software includes components with separate copyright notices and license
terms. Your use of the source code for these components is subject to the terms
and conditions of the following licenses.

3-clause BSD license:
* [Xbyak](https://github.com/herumi/xbyak)
* [gtest](https://github.com/google/googletest)
* [ittnotify](https://github.com/intel/IntelSEAPI)
* [CMake](https://github.com/Kitware/CMake)

Apache License Version 2.0:
* [MathJax](https://github.com/mathjax/MathJax)

Boost Software License, Version 1.0:
* [Boost C++ Libraries](https://www.boost.org/)

See accompanying [LICENSE](LICENSE) file for full license text and copyright notices.

--------

[Legal Information](doc/legal_information.md)
