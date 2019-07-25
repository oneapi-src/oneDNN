Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)
========================================================================

> **Note**
>
> Version 1.0 brings incompatible changes to the 0.20 version. Please read
> [Version 1.0 Transition Guide](https://intel.github.io/mkl-dnn/dev_guide_transition_to_v1.html).

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an
open-source performance library for deep learning applications. The library
includes basic building blocks for neural networks optimized
for Intel Architecture Processors and Intel Processor Graphics.

> **Note**
> Intel MKL-DNN is distinct from Intel MKL, which is general math
> performance library.

Intel MKL-DNN is intended for deep learning applications and framework
developers interested in improving application performance
on Intel CPUs and GPUs. Deep learning practitioners should use one of the
applications enabled with Intel MKL-DNN:
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

## License
Intel MKL-DNN is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). This
software includes the following third-party components:
* [Xbyak](https://github.com/herumi/xbyak) distributed under
[3-clause BSD licence](src/cpu/xbyak/COPYRIGHT)
* [gtest](https://github.com/google/googletest) distributed under
[3-clause BSD license](tests/gtests/gtest/LICENSE)
* [ittnotify](https://github.com/intel/IntelSEAPI) distributed under
[3-clause BSD license](src/cpu/jit_utils/jitprofiling/LICENSE.BSD)

## Documentation
* [Developer guide](https://intel.github.io/mkl-dnn) explains programming
model, supported functionality, details of primitives implementations and
includes annotated examples.
* [API reference](https://intel.github.io/mkl-dnn/modules.html) provides
comprehensive reference of the library API.

## Support
Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/intel/mkl-dnn/issues) page.

> **WARNING**
> The following functionality has preview status and might change without prior
> notification in future releases.

* Threading Building Blocks (TBB) support

## How to Contribute
We welcome community contributions to Intel MKL-DNN. If you have an idea on how
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

## System Requirements
Intel MKL-DNN supports systems meeting the following requirements:
* Intel 64 architecture or compatible
* C++ compiler with C++11 standard support
* [CMake](https://cmake.org/download/) 2.8.11 or later
* [Doxygen](http://www.doxygen.nl/download.html#srcbin) 1.8.5 or later

Configurations of CPU and GPU engines may introduce additional build time
dependencies.

### CPU Support
Intel Architecture Processors and compatible devices are supported by
Intel MKL-DNN CPU engine. The CPU engine is built by default and cannot
be disabled at build time. The engine can be configured to use OpenMP or
TBB threading runtime. The following additional requirements apply:
* OpenMP runtime requires C++ compiler with OpenMP 2.0 or later standard support
* TBB runtime requires
[Threading Building Blocks (TBB)](https://www.threadingbuildingblocks.org/)
2017 or later.

The library is optimized for systems based on
* Intel Atom processor with Intel SSE4.1 support
* 4th, 5th, 6th, 7th, and 8th generation Intel Core(TM) processor
* Intel Xeon(R) processor E3, E5, and E7 family (formerly Sandy Bridge,
  Ivy Bridge, Haswell, and Broadwell)
* Intel Xeon Phi(TM) processor (formerly Knights Landing and Knights Mill)
* Intel Xeon Scalable processor (formerly Skylake and Cascade Lake)
* future Intel Xeon Scalable processor (code name Cooper Lake)

and compatible processors.

Intel MKL-DNN detects instruction set architecture (ISA) in the runtime and uses
just-in-time (JIT) code generation to deploy the code optimized
for the latest supported ISA. Some implementations rely on OpenMP 4.0 SIMD
extensions and we recommend using the Intel C++ Compiler for the best
performance results.

> **Warning**
> In the default build configuration, Intel MKL-DNN targets build system ISA as
> the minimal supported ISA for the build. To make sure that the build is
> portable to older systems, you might need to override
> [MKLDNN_ARCH_OPT_FLAGS](http://intel.github.io/mkl-dnn/dev_guide_build_options.html).

CPU engine was validated on RedHat\* Enterprise Linux 7 with
* GNU Compiler Collection 4.8, 5.4, 6.1, 7.2, and 8.1
* Clang\* 3.8.0
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0, 18.0, and 19.0

on Windows Server\* 2012 R2 with
* Microsoft Visual C++ 14.0 (Visual Studio 2015 Update 3)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0 and 19.0

on macOS\* 10.13 (High Sierra) with
* Apple LLVM version 9.2 (XCode 9.2)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  18.0 and 19.0

### GPU Support
Intel Processor Graphics is supported by Intel MKL-DNNs GPU engine. GPU engine
is disabled in the default build configuration. The following
additional requirements apply when GPU engine is enabled:
* OpenCL\* runtime library (OpenCL\* version 1.2 or later)
* OpenCL\* driver (with kernel language support for OpenCL\* C 2.0 or later)
  with Intel subgroups extension support

The library is optimized for systems based on
* Intel HD Graphics
* Intel UHD Graphics
* Intel Iris Plus Graphics

GPU engine was validated on Ubuntu\* 18.04 with
* GNU Compiler Collection 5.4 and 8.1
* Clang\* 3.8.1
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  19.0
* [Intel SDK for OpenCL\* applications](https://software.intel.com/en-us/intel-opencl) 2019 Update 3
* [Intel Graphics Compute Runtime for OpenCL\*](https://github.com/intel/compute-runtime/releases) 19.15.12831

on Windows Server\* 2019 with
* Microsoft Visual C++ 14.0 (Visual Studio 2015 Update 3)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  19.0
* [Intel SDK for OpenCL\* applications](https://software.intel.com/en-us/intel-opencl) 2019 Update 3
* [Intel Graphics - Windows\* 10 DCH Drivers](https://downloadcenter.intel.com/download/28783/Intel-Graphics-Windows-10-DCH-Drivers) 26.20.100.6709

--------

[Legal Information](doc/legal_information.md)
