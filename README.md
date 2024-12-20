[![UXL Foundation Logo](https://github.com/uxlfoundation/artwork/blob/main/foundation/uxl-foundation-logo-horizontal-color.png)][UXL Foundation]

oneAPI Deep Neural Network Library (oneDNN)
===========================================

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8762/badge)](https://www.bestpractices.dev/projects/8762)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/oneapi-src/oneDNN/badge)](https://securityscorecards.dev/viewer/?uri=github.com/oneapi-src/oneDNN)

oneAPI Deep Neural Network Library (oneDNN) is an open-source cross-platform
performance library of basic building blocks for deep learning applications.
oneDNN project is part of the [UXL Foundation] and is an implementation
of the [oneAPI specification] for oneDNN component.

The library is optimized for Intel(R) Architecture Processors, Intel Graphics,
and Arm(R) 64-bit Architecture (AArch64)-based processors. oneDNN has
experimental support for the following architectures: NVIDIA\* GPU,
AMD\* GPU, OpenPOWER\* Power ISA (PPC64), IBMz\* (s390x), and RISC-V.

oneDNN is intended for deep learning applications and framework
developers interested in improving application performance on CPUs and GPUs.
Deep learning practitioners should use one of the
[applications enabled with oneDNN](#applications-enabled-with-onednn).

[UXL Foundation]: http://www.uxlfoundation.org
[oneAPI specification]: https://spec.oneapi.io

# Table of Contents

- [Documentation](#documentation)
- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Applications Enabled with oneDNN](#applications-enabled-with-onednn)
- [Governance](#governance)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)
- [Security](#security)
- [Trademark Information](#trademark-information)

# Documentation

* [Developer Guide] explains the programming model, supported functionality,
  and implementation details, and includes annotated examples.
* [API Reference] provides a comprehensive reference of the library API.

[Developer Guide]: https://oneapi-src.github.io/oneDNN
[API Reference]: https://oneapi-src.github.io/oneDNN/group_dnnl_api.html

# Installation

Binary distribution of this software is available in:
* [Anaconda]
* [Intel oneAPI]

The packages do not include library dependencies and these need to be resolved
in the application at build time. See the [System Requirements] section below
and the [Build Options] section in the [Developer Guide] for more details on
CPU and GPU runtimes.

If the configuration you need is not available, you can
[build the library from source][Build from Source].

[Anaconda]: https://anaconda.org/conda-forge/onednn
[Intel oneAPI]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html
[System Requirements]: #system-requirements
[Build Options]: https://oneapi-src.github.io/oneDNN/dev_guide_build_options.html
[Build from Source]: https://oneapi-src.github.io/oneDNN/dev_guide_build.html

# System Requirements

oneDNN supports platforms based on the following architectures:
- [Intel 64 or AMD64](https://en.wikipedia.org/wiki/X86-64),
- [Arm 64-bit Architecture (AArch64)](https://developer.arm.com/architectures/cpu-architecture/a-profile).
- [OpenPOWER](https://openpowerfoundation.org/) / [IBM Power ISA](https://en.wikipedia.org/wiki/Power_ISA).
- [IBMz z/Architecture (s390x)](https://en.wikipedia.org/wiki/Z/Architecture).
- [RISC-V 64-bit (RV64)](https://en.wikipedia.org/wiki/RISC-V).

> **WARNING**
>
> Power ISA (PPC64), IBMz (s390x), and RISC-V (RV64) support is
> **experimental** with limited testing validation.

The library is optimized for the following CPUs:
* Intel 64/AMD64 architecture
  * Intel Atom(R) processor (at least Intel SSE4.1 support is required)
  * Intel Core(TM) processor (at least Intel SSE4.1 support is required)
  * Intel Xeon(R) processor E3, E5, and E7 family (formerly Sandy Bridge,
    Ivy Bridge, Haswell, and Broadwell)
  * Intel Xeon Scalable processor (formerly Skylake, Cascade Lake, Cooper
    Lake, Ice Lake, Sapphire Rapids, and Emerald Rapids)
  * Intel Xeon CPU Max Series (formerly Sapphire Rapids HBM)
  * Intel Core Ultra processors (formerly Meteor Lake, Arrow Lake,
    and Lunar Lake)
  * Intel Xeon 6 processors (formerly Sierra Forest and Granite Rapids)
* AArch64 architecture
  * Arm Neoverse(TM) N1 and V1 processors

On a CPU based on Intel 64 or on AMD64 architecture, oneDNN detects
the instruction set architecture (ISA) at runtime and uses just-in-time (JIT)
code generation to deploy the code optimized for the latest supported ISA.
Future ISAs may have initial support in the library disabled by default and
require the use of run-time controls to enable them. See
[CPU dispatcher control] for more details.


> **WARNING**
>
> On macOS, applications that use oneDNN may need to request special
> entitlements if they use the hardened runtime. See the
> [Linking Guide] for more details.

The library is optimized for the following GPUs:
* Intel Graphics for 11th-14th Generation Intel Core Processors
* Intel Iris Xe MAX Graphics (formerly DG1)
* Intel Arc(TM) graphics (formerly Alchemist)
* Intel Data Center GPU Flex Series (formerly Arctic Sound)
* Intel Data Center GPU Max Series (formerly Ponte Vecchio)
* Intel Graphics and Intel Arc graphics for Intel Core Ultra processors
 (formerly Meteor Lake, Arrow Lake and Lunar Lake)
* future Intel Arc graphics (code name Battlemage)

[CPU dispatcher control]: https://oneapi-src.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html
[Linking Guide]: https://oneapi-src.github.io/oneDNN/dev_guide_link.html

## Requirements for Building from Source

oneDNN supports systems meeting the following requirements:
* Operating system with Intel 64 / Arm 64 / Power / IBMz architecture support
* C++ compiler with C++11 standard support
* [CMake] 3.13 or later

The following tools are required to build oneDNN documentation:
* [Doxygen] 1.8.5 or later
* [Doxyrest] 2.1.2 or later
* [Sphinx] 4.0.2 or later
* [sphinx-book-theme] 0.0.41 or later

Configurations of CPU and GPU engines may introduce additional build time
dependencies.

[CMake]: https://cmake.org/download/
[Doxygen]: http://www.doxygen.nl/download.html#srcbin
[Doxyrest]: https://github.com/vovkos/doxyrest
[Sphinx]: https://www.sphinx-doc.org/en/master/usage/installation.html
[sphinx-book-theme]: https://sphinx-book-theme.readthedocs.io/en/latest

### CPU Engine

oneDNN CPU engine is used to execute primitives on Intel Architecture
Processors, 64-bit Arm Architecture (AArch64) processors,
64-bit Power ISA (PPC64) processors, IBMz (s390x), and compatible devices.

The CPU engine is built by default but can be disabled at build time by setting
`DNNL_CPU_RUNTIME` to `NONE`. In this case, GPU engine must be enabled.
The CPU engine can be configured to use the OpenMP, TBB or SYCL runtime.
The following additional requirements apply:
* OpenMP runtime requires C++ compiler with OpenMP 2.0 or later
  standard support
* TBB runtime requires [Threading Building Blocks (TBB)] 2017 or later.
* SYCL runtime requires
  * [Intel oneAPI DPC++/C++ Compiler]
  * [Threading Building Blocks (TBB)]

Some implementations rely on OpenMP 4.0 SIMD extensions. For the best
performance results on Intel Architecture Processors we recommend using the
Intel C++ Compiler.

[Threading Building Blocks (TBB)]: https://www.threadingbuildingblocks.org/
[Intel oneAPI DPC++/C++ Compiler]: https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html

On a CPU based on Arm AArch64 architecture, oneDNN CPU engine can be built with
[Arm Compute Library (ACL)] integration. ACL is an open-source library for
machine learning applications and provides AArch64 optimized implementations
of core functions. This functionality currently requires that ACL is downloaded
and built separately. See [Build from Source] section of the Developer Guide for
details. oneDNN only supports Compute Library versions 24.11.1 or later.

[Arm Compute Library (ACL)]: https://github.com/arm-software/ComputeLibrary

### GPU Engine

Intel Processor Graphics and Xe Architecture graphics are supported by
the oneDNN GPU engine. The GPU engine is disabled in the default build
configuration. The following additional requirements apply when GPU engine
is enabled:
* OpenCL runtime requires
    * OpenCL\* runtime library (OpenCL version 1.2 or later)
    * OpenCL driver (with kernel language support for OpenCL C 2.0 or later)
      with Intel subgroups and USM extensions support
* SYCL runtime requires
    * [Intel oneAPI DPC++/C++ Compiler]
    * OpenCL runtime library (OpenCL version 3.0 or later)
    * [oneAPI Level Zero]
* SYCL runtime with NVIDIA GPU support requires
    * [oneAPI DPC++ Compiler with support for CUDA] or [oneAPI for NVIDIA GPUs]
    * NVIDIA CUDA\* driver
    * cuBLAS 10.1 or later
    * cuDNN 7.6 or later
* SYCL runtime with AMD GPU support requires
    * [oneAPI DPC++ Compiler with support for HIP AMD] or [oneAPI for AMD GPUs]
    * [AMD ROCm] version 5.3 or later
    * [MIOpen] version 2.18 or later (optional if AMD ROCm includes the required
    version of MIOpen)
    * [rocBLAS] version 2.45.0 or later (optional if AMD ROCm includes
    the required version of rocBLAS)
* SYCL runtime with a generic GPU support requires
    * oneAPI DPC++/C++ Compiler that supports the target GPU. Refer to the
    [generic GPU vendor readme](src/gpu/generic/sycl/README.md) for more information.

> **WARNING**
>
> Linux will reset GPU when kernel runtime exceeds several seconds. The user
> can prevent this behavior by [disabling hangcheck] for Intel GPU driver.
> Windows has built-in [timeout detection and recovery] mechanism that results
> in similar behavior. The user can prevent this behavior by increasing the
> [TdrDelay] value.

> **WARNING**
>
> NVIDIA GPU support is experimental. General information, build instructions,
> and implementation limitations are available in the
> [NVIDIA backend readme](src/gpu/nvidia/README.md).

> **WARNING**
>
> AMD GPU support is experimental. General information, build instructions,
> and implementation limitations are available in the
> [AMD backend readme](src/gpu/amd/README.md).

[oneAPI Level Zero]: https://github.com/oneapi-src/level-zero
[oneAPI DPC++ Compiler with support for CUDA]: https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-nvidia-cuda
[oneAPI for NVIDIA GPUs]: https://developer.codeplay.com/products/oneapi/nvidia/home
[oneAPI DPC++ Compiler with support for HIP AMD]: https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md#build-dpc-toolchain-with-support-for-hip-amd
[oneAPI for AMD GPUs]: https://developer.codeplay.com/products/oneapi/amd/home/
[AMD ROCm]: https://github.com/RadeonOpenCompute/ROCm
[MIOpen]: https://github.com/ROCmSoftwarePlatform/MIOpen
[rocBLAS]: https://github.com/ROCmSoftwarePlatform/rocBLAS
[disabling hangcheck]: https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2023-0/gpu-disable-hangcheck.html
[timeout detection and recovery]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery
[TdrDelay]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/tdr-registry-keys#tdrdelay

### Runtime Dependencies

When oneDNN is built from source, the library runtime dependencies and specific
versions are defined by the build environment.

#### Linux

Common dependencies:
* GNU C Library (`libc.so`)
* GNU Standard C++ Library v3 (`libstdc++.so`)
* Dynamic Linking Library (`libdl.so`)
* C Math Library (`libm.so`)
* POSIX Threads Library (`libpthread.so`)

Runtime-specific dependencies:

| Runtime configuration    | Compiler                      | Dependency
| :----------------------- | :---------------------------- | :---------
| `DNNL_CPU_RUNTIME=OMP`   | GCC                           | GNU OpenMP runtime (`libgomp.so`)
| `DNNL_CPU_RUNTIME=OMP`   | Intel C/C++ Compiler          | Intel OpenMP runtime (`libiomp5.so`)
| `DNNL_CPU_RUNTIME=OMP`   | Clang                         | Intel OpenMP runtime (`libiomp5.so`)
| `DNNL_CPU_RUNTIME=TBB`   | any                           | TBB (`libtbb.so`)
| `DNNL_CPU_RUNTIME=SYCL`  | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`libsycl.so`), TBB (`libtbb.so`), OpenCL loader (`libOpenCL.so`)
| `DNNL_GPU_RUNTIME=OCL`   | any                           | OpenCL loader (`libOpenCL.so`)
| `DNNL_GPU_RUNTIME=SYCL`  | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`libsycl.so`), OpenCL loader (`libOpenCL.so`), oneAPI Level Zero loader (`libze_loader.so`)

#### Windows

Common dependencies:
* Microsoft Visual C++ Redistributable (`msvcrt.dll`)

Runtime-specific dependencies:

| Runtime configuration    | Compiler                      | Dependency
| :----------------------- | :---------------------------- | :---------
| `DNNL_CPU_RUNTIME=OMP`   | Microsoft Visual C++ Compiler | No additional requirements
| `DNNL_CPU_RUNTIME=OMP`   | Intel C/C++ Compiler          | Intel OpenMP runtime (`iomp5.dll`)
| `DNNL_CPU_RUNTIME=TBB`   | any                           | TBB (`tbb.dll`)
| `DNNL_CPU_RUNTIME=SYCL`  | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`sycl.dll`), TBB (`tbb.dll`), OpenCL loader (`OpenCL.dll`)
| `DNNL_GPU_RUNTIME=OCL`   | any                           | OpenCL loader (`OpenCL.dll`)
| `DNNL_GPU_RUNTIME=SYCL`  | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`sycl.dll`), OpenCL loader (`OpenCL.dll`), oneAPI Level Zero loader (`ze_loader.dll`)

#### macOS

Common dependencies:
* System C/C++ runtime (`libc++.dylib`, `libSystem.dylib`)

Runtime-specific dependencies:

| Runtime configuration  | Compiler                      | Dependency
| :--------------------- | :---------------------------- | :---------
| `DNNL_CPU_RUNTIME=OMP` | Intel C/C++ Compiler          | Intel OpenMP runtime (`libiomp5.dylib`)
| `DNNL_CPU_RUNTIME=TBB` | any                           | TBB (`libtbb.dylib`)

### Validated Configurations

x86-64 CPU engine was validated on RedHat\* Enterprise Linux 8 with
* GNU Compiler Collection 8.5, 9.5, 11.1, 11.3
* Clang\* 11.0, 14.0.6
* [Intel oneAPI DPC++/C++ Compiler] 2024.0

on Windows Server\* 2019 with
* Microsoft Visual Studio 2022
* [Intel oneAPI DPC++/C++ Compiler] 2024.0

on macOS 11 (Big Sur) with
* Apple LLVM version 13.0

AArch64 CPU engine was validated on Ubuntu 22.04 with
* GNU Compiler Collection 10.0, 13.0
* Clang\* 17.0
* [Arm Compiler for Linux] 24.04
* [Arm Compute Library (ACL)] built for armv8-a arch, latest stable version
available at the time of release

on macOS 14 (Sonoma) with
* Apple LLVM version 15.0

GPU engine was validated on Ubuntu\* 22.04 with
* GNU Compiler Collection 8.5, and 9.5
* Clang 11.0
* [Intel oneAPI DPC++/C++ Compiler] 2024.0
* [Intel Software for General Purpose GPU capabilities] latest stable version
available at the time of release

on Windows Server 2019 with
* Microsoft Visual Studio 2022
* [Intel oneAPI DPC++/C++ Compiler] 2024.0
* [Intel Arc & Iris Xe Graphics Driver] latest stable version available at the
time of release

[Intel Software for General Purpose GPU capabilities]: https://dgpu-docs.intel.com/index.html
[Intel Arc & Iris Xe Graphics Driver]: https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html
[Arm Compiler for Linux]: https://developer.arm.com/Tools%20and%20Software/Arm%20Compiler%20for%20Linux

# Applications Enabled with oneDNN

* [Apache\* MXNet](https://mxnet.apache.org)
* [Apache SINGA](https://singa.apache.org)
* [DeepLearning4J\*](https://deeplearning4j.konduit.ai)
* [Flashlight\*](https://github.com/flashlight/flashlight)
* [Korali](https://github.com/cselab/korali)
* [MATLAB\* Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning)
* [ONNX Runtime](https://onnxruntime.ai)
* [OpenVINO(TM) toolkit](https://github.com/openvinotoolkit/openvino)
* [PaddlePaddle\*](http://www.paddlepaddle.org)
* [PyTorch\*](https://pytorch.org). Intel GPU support and additional
optimizations are available with [Intel Extension for PyTorch].
* [Tensorflow\*](https://www.tensorflow.org). Intel GPU support and additional
optimizations are available with [Intel Extension for Tensorflow].

[Intel Extension for PyTorch]: https://github.com/intel/intel-extension-for-pytorch
[Intel Extension for Tensorflow]: https://github.com/intel/intel-extension-for-tensorflow

# Support

Submit questions, feature requests, and bug reports on the
[GitHub issues] page.

You can also contact oneDNN developers via [UXL Foundation Slack] using
[#onednn] channel.

[Github issues]: https://github.com/oneapi-src/oneDNN/issues
[UXL Foundation Slack]: https://slack-invite.uxlfoundation.org/
[#onednn]: https://uxlfoundation.slack.com/channels/onednn

# Governance

oneDNN project is governed by the [UXL Foundation] and you can get involved in
this project in multiple ways. It is possible to join the [AI Special Interest
Group (SIG)] meetings where the groups discuss and demonstrate work using this
project. Members can also join the Open Source and Specification Working Group
meetings.

You can also join the [mailing lists for the UXL Foundation] to be informed
of when meetings are happening and receive the latest information and
discussions.

[AI Special Interest Group (SIG)]: https://github.com/uxlfoundation/foundation
[mailing lists for the UXL Foundation]: https://lists.uxlfoundation.org/g/main/subgroups

# Contributing

We welcome community contributions to oneDNN. You can find the oneDNN release
schedule and work already in progress towards future milestones in Github's
[Milestones] section. If you are looking for a specific task to start,
consider selecting from issues that are marked with the [help wanted] label.

If you have an idea on how to improve the library:
* For changes impacting the public API or library overall, such as adding new
primitives or changes to the architecture, submit an [RFC pull request].
* Ensure that the changes are consistent with the [code contribution guidelines]
and [coding standards].
* Ensure that you can build the product and run all the examples with your
patch.
* Submit a [pull request].

For additional details, see [contribution guidelines](CONTRIBUTING.md). You can
also contact oneDNN developers and maintainers via [UXL Foundation Slack] using
[#onednn] channel.

This project is intended to be a safe, welcoming space for collaboration, and
contributors are expected to adhere to the
[Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

[RFC pull request]: https://github.com/oneapi-src/oneDNN/tree/rfcs
[code contribution guidelines]: CONTRIBUTING.md#code-contribution-guidelines
[coding standards]: CONTRIBUTING.md#coding-standards
[pull request]: https://github.com/oneapi-src/oneDNN/pulls
[Milestones]: https://github.com/oneapi-src/oneDNN/milestones
[help wanted]: https://github.com/oneapi-src/oneDNN/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22

# License

oneDNN is licensed under [Apache License Version 2.0](LICENSE). Refer to the
"[LICENSE](LICENSE)" file for the full license text and copyright notice.

This distribution includes third party software governed by separate license
terms.

3-clause BSD license:
* [Xbyak](https://github.com/herumi/xbyak)
* [gtest](https://github.com/google/googletest)
* [Instrumentation and Tracing Technology API
(ITT API)](https://github.com/intel/ittapi)
* [CMake](https://github.com/Kitware/CMake)

2-clause BSD license:
* [Sphinx](https://www.sphinx-doc.org/)

Apache License Version 2.0:
* [Xbyak_aarch64](https://github.com/fujitsu/xbyak_aarch64)
* [LLVM](https://llvm.org)

Boost Software License, Version 1.0:
* [Boost C++ Libraries](https://www.boost.org/)

MIT License:
* [Intel Graphics Compute Runtime for oneAPI Level Zero
and OpenCL Driver](https://github.com/intel/compute-runtime)
* [Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler)
* [oneAPI Level Zero](https://github.com/oneapi-src/level-zero)
* [Doxyrest](https://github.com/vovkos/doxyrest)
* [Intel Metrics Discovery Application Programming
Interface](https://github.com/intel/metrics-discovery)
* [spdlog](https://github.com/gabime/spdlog)

This third party software, even if included with the distribution of
the Intel software, may be governed by separate license terms, including
without limitation, third party license terms, other Intel software license
terms, and open source software license terms. These separate license terms
govern your use of the third party programs as set forth in the
"[THIRD-PARTY-PROGRAMS](THIRD-PARTY-PROGRAMS)" file.

# Security

[Security Policy](SECURITY.md) outlines our guidelines and procedures
for ensuring the highest level of Security and trust for our users
who consume oneDNN.

# Trademark Information

Intel, the Intel logo, Arc, Intel Atom, Intel Core, Iris,
OpenVINO, the OpenVINO logo, Pentium, VTune, and Xeon are trademarks
of Intel Corporation or its subsidiaries.

Arm and Neoverse are trademarks, or registered trademarks of Arm Ltd.

\* Other names and brands may be claimed as the property of others.

Microsoft, Windows, and the Windows logo are trademarks, or registered
trademarks of Microsoft Corporation in the United States and/or other
countries.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission
by Khronos.

(C) Intel Corporation
