oneAPI Deep Neural Network Library (oneDNN) Graph API
===========================================

This branch is to propose a preview for the graph API extension to oneDNN. 
oneDNN Graph API extends oneDNN with a unified high-level graph API for 
multiple AI hardware classes (CPU, GPU, accelerators). With a flexible graph 
interface, it maximizes the optimization opportunity for generating efficient 
code across a variety of Intel and non-Intel hardwares, and can be closely 
integrated with ecosystem frameworks and inference engines.

The goal of the preview branch is to:
* Collect feedbacks on the API design, not the implementation.
* Demonstrate the programming model of oneDNN Graph API.
* Show ease of use of the API for framework integration.

The current API version has limited support for direct programming model, as 
it assumes users maintain their own graphs and use oneDNN Graph API to identify 
the partitions which could be offloaded to oneDNN Graph. Currently, it aims to 
work with the framework graph and identify graph partitions to offload. The 
partition will be further compiled and executed as a fused operation in the 
framework graph.

## Documentation

[Public specification](https://spec.oneapi.com/onednn-graph/latest/index.html) 
on oneAPI SPEC website explains the design, programming model, and operation 
set of oneDNN Graph API.

Developer guide and API reference of this branch can be generated from source 
code. Public webpages are still under construction.

## System Requirements

oneDNN Graph Library will support systems based on
[Intel 64 or AMD64 architecture](https://en.wikipedia.org/wiki/X86-64).

## Requirements for Building from Source

oneDNN Graph supports systems meeting the following requirements:
* Operating system with Intel 64 architecture support
* C++ compiler with C++11 standard support
* [CMake](https://cmake.org/download/) 3.9 or later
* [Doxygen](http://www.doxygen.nl/download.html#srcbin) 1.8.5 or later
  to build the documentation

Configurations of CPU and GPU engines may introduce additional build time
dependencies.

### Validated Configurations

CPU engine was validated on RedHat* Enterprise Linux 7 with
* GNU Compiler Collection 4.8
* GNU Compiler Collection 8.2
* GNU Compiler Collection 9.3

on Ubuntu* 18.04 with
* GNU Compiler Collection 7.5

on macOS* 10.15 (Catalina) with
* Apple LLVM version 10.0 (XCode 11.2)

GPU engine was validated on Ubuntu* 18.04 with
* [Intel Graphics Compute Runtime for OpenCL](https://github.com/intel/compute-runtime/releases)
* [Intel oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler)
  Beta

### Typical Build Steps

To build the library from source code:

```bash
git clone https://github.com/oneapi-src/oneDNN.git --branch dev-graph --recursive
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

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

## Trademark Information

Intel, the Intel logo, Intel Atom, Intel Core, Intel Xeon Phi, Iris, OpenVINO,
the OpenVINO logo, Pentium, VTune, and Xeon are trademarks of Intel Corporation
 or its subsidiaries.

\* Other names and brands may be claimed as the property of others.
