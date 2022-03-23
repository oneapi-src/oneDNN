oneAPI Deep Neural Network Library (oneDNN) Graph API
===========================================

This branch is to propose a preview for the graph API extension to oneDNN.
oneDNN Graph API extends oneDNN with a unified high-level graph API for multiple
AI hardware classes (CPU, GPU, accelerators). With a flexible graph interface,
it maximizes the optimization opportunity for generating efficient code across a
variety of Intel and non-Intel hardwares, and can be closely integrated with
ecosystem frameworks and inference engines.

The goal of the preview branch is to:

* Collect feedback on the API design, not the implementation.
* Demonstrate the programming model of oneDNN Graph API.
* Show ease of use of the API for framework integration.
* Provide product-level quality support to early adoption before merging to
  oneDNN master branch.

The current API version aims to work with the framework graph and identify graph
partitions to offload. The partition will be further compiled and executed as a
fused operation in the framework graph. To support aggressive operation fusion
and achieve the best performance, oneDNN Graph implementation includes a
low-level graph compiler, also known as oneDNN Graph compiler.

Enabling oneDNN Graph compiler doesn't result in any change to its programming
model, Graph API users can choose to turn on or off graph compiler during build
or installation of oneDNN Graph.

## Documentation

[Public specification](https://spec.oneapi.com/onednn-graph/latest/index.html)
on oneAPI SPEC website explains the design, programming model, and operation set
of oneDNN Graph API.

In the doc folder of this branch, [an overview introduction](doc/README.md) to
oneDNN Graph API is provided along with two tutorials on how to use the API:
[CPU version](doc/programming_model/cpu_programming.md) and [SYCL
version](doc/programming_model/sycl_get_started.md).

Developer guide and API reference of this branch can be generated from source
code. Public webpages are still under construction.

## System Requirements

oneDNN Graph supports platforms based on the following architectures:

* [Intel 64 or AMD64](https://en.wikipedia.org/wiki/X86-64)

## Requirements for Building from Source

oneDNN Graph supports systems meeting the following requirements:

* Operating system with Intel 64 and Arm 64 architecture support
* C++ compiler with C++11 standard support
* [CMake](https://cmake.org/download/) 2.8.12 or later
* [Doxygen](http://www.doxygen.nl/download.html#srcbin) 1.8.5 or later to build
  the documentation

Configurations of CPU and GPU engines may introduce additional build time
dependencies.

### Validated Configurations

CPU engine was validated on RedHat* Enterprise Linux 7 with

* GNU Compiler Collection 4.8, 6.3, 8.2, 9.3, 10.2

on Ubuntu* 18.04 with

* GNU Compiler Collection 7.5
* Clang\* 3.8.1, 10.0
* [Intel C++ Compiler Classic](https://software.intel.com/content/www/us/en/develop/tools/oneapi/hpc-toolkit.html)

on macOS* 11.2 (BigSur) with

* Apple LLVM version 12.0 (XCode 12.0)
* [Intel C++ Compiler Classic](https://software.intel.com/content/www/us/en/develop/tools/oneapi/hpc-toolkit.html)

on Windows* with

* Microsoft Visual C++ 16.0 (Visual Studio 2019)
* Microsoft Visual C++ 15.0 (Visual Studio 2017)

GPU engine was validated on Ubuntu* 18.04 with

* [Intel Graphics Compute Runtime for OpenCL](https://github.com/intel/compute-runtime/releases)
* [Intel oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler)
  Beta

### Typical Build Steps

To build the library from source code:

```bash
git clone https://github.com/oneapi-src/oneDNN.git --branch dev-graph-alpha --recursive
cd oneDNN
mkdir build && cd build
cmake .. -DDNNL_GRAPH_BUILD_TESTS=1 -DDNNL_GRAPH_BUILD_EXAMPLES=1
make -j
```

To validate the library with tests and examples:

```bash
cd build
ctest -V
```

To install the built library, you need to have the write privilege of the target
directory with sudo or specifying the target directory via
`-DCMAKE_INSTALL_PREFIX` in the cmake command line.

```bash
make install
```

To build on Windows, see [Build from Source](./doc/build/build.md#Windows).

## Support

Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/oneapi-src/oneDNN/issues) page.

You may reach out to project maintainers privately
at dnnl.maintainers@intel.com.

> **WARNING**
>
> This is pre-production software and functionality may change without prior
> notice.

## License

oneDNN Graph Library is licensed under [Apache License Version 2.0](LICENSE).
Refer to the "[LICENSE](LICENSE)" file for the full license text and copyright
notice.

This distribution includes third party software governed by separate license
terms. This third party software, even if included with the distribution of the
Intel software, may be governed by separate license terms, including without
limitation, third party license terms, other Intel software license terms, and
open source software license terms. These separate license terms govern your use
of the third party programs as set forth in the
"[THIRD-PARTY-PROGRAMS](THIRD-PARTY-PROGRAMS)" file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

## Trademark Information

Intel, the Intel logo, Intel Atom, Intel Core, Intel Xeon Phi, Iris, OpenVINO,
the OpenVINO logo, Pentium, VTune, and Xeon are trademarks of Intel Corporation
or its subsidiaries.

\* Other names and brands may be claimed as the property of others.

Microsoft, Windows, and the Windows logo are trademarks, or registered
trademarks of Microsoft Corporation in the United States and/or other
countries.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by
Khronos.

(C) Intel Corporation
