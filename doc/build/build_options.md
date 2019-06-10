Build Options {#dev_guide_build_options}
====================================

## Configuring the Build

Intel MKL-DNN supports the following build-time options.

| Option                      | Supported values (defaults in bold)  | Description
| :---                        | :---                                 | :---
| MKLDNN_LIBRARY_TYPE         | **SHARED**, STATIC                   | Defines the resulting library type
| MKLDNN_THREADING            | **OMP**, TBB                         | Defines the threading type
| MKLDNN_CPU_BACKEND          | **NONE**, SYCL                       | Defines the backend for CPU engine
| MKLDNN_GPU_BACKEND          | **NONE**, OPENCL, SYCL               | Defines the backend for GPU engine
| MKLDNN_BUILD_EXAMPLES       | **ON**, OFF                          | Controls building the examples
| MKLDNN_BUILD_TESTS          | **ON**, OFF                          | Controls building the tests
| MKLDNN_ARCH_OPT_FLAGS       | *compiler flags*                     | Specifies compiler optimization flags (see warning note below)
| MKLDNN_ENABLE_JIT_PROFILING | **ON**, OFF                          | Enables integration with Intel(R) VTune(TM) Amplifier

All other building options that can be found in CMake files are dedicated for
the development/debug purposes and are subject to change without any notice.
Please avoid using them.

## Targeting Specific Architecture

Intel MKL-DNN uses JIT code generation to implement most of its functionality
and will choose the best code based on detected processor features. However,
some Intel MKL-DNN functionality will still benefit from targeting a specific
processor architecture at build time. You can use `MKLDNN_ARCH_OPT_FLAGS` CMake
option for this.

For Intel(R) C++ Compilers, the default option is `-xHOST`, which instructs
the compiler to generate the code for the architecture of the processor where
the build is occurring.  This option would not allow you to run the library on
older processor architectures.

For GNU\* Compiler 5.0 and newer, the default options are `-march=native
-mtune=native`.

@warning
While use of `MKLDNN_ARCH_OPT_FLAGS` option gives better performance, the
resulting library can be run only on systems that have instruction set
compatible with the target instruction set. Therefore, `ARCH_OPT_FLAGS`
should be set to an empty string (`""`) if the resulting library needs to be
portable.

## GPU support (experimental)

To enable GPU support in Intel MKL-DNN you need to specify the GPU backend
to use. Currently the only supported one is implemented using OpenCL\* and
requires Intel(R) SDK for OpenCL\* applications. You can explicitly specify
the pass to the SDK using `-DOPENCLROOT` cmake option.

~~~sh
cmake -DMKLDNN_GPU_BACKEND=OPENCL -DOPENCLROOT=/path/to/opencl/sdk ..
~~~

## SYCL\* support (experimental)

To enable SYCL support in Intel MKL-DNN you need to specify the SYCL backend for
CPU engine (and for GPU engine to enable GPU support).

SYCL support requires a SYCL compiler with SYCL 1.2.1 standard support.

You need to set C and C++ compilers to point to SYCL compilers. Also you can
explicitly specify the path to the SYCL installation using `-DSYCLROOT` cmake option.

~~~sh
export CC=path/to/c/compiler
export CXX=path/to/cpp/sycl/compiler

cmake -DMKLDNN_CPU_BACKEND=SYCL -DMKLDNN_GPU_BACKEND=SYCL -DSYCLROOT=/path/to/sycl ..
~~~

## Threading

Intel MKL-DNN can use the OpenMP or TBB threading runtimes. OpenMP threading
is the default build mode and is recommended for the best performance. TBB
support is experimental. This behavior is controlled by the `MKLDNN_THREADING`
CMake option.

### OpenMP

Intel MKL-DNN uses OpenMP runtime library provided by the compiler.

@warning
Because different OpenMP runtimes may not be binary-compatible, it's important
to ensure that only one OpenMP runtime is used throughout the application.
Having more than one OpenMP runtime linked to an executable may lead to
undefined behavior including incorrect results or crashes. However as long as
both the library and the application use the same or compatible compilers there
would be no conflicts.

### TBB

TBB support is experimental.

To build Intel MKL-DNN with TBB support, set the `TBBROOT` environmental
variable to point to the TBB installation path or pass the path directly to
cmake:

~~~sh
$ cmake -DTBBROOT=/opt/intel/path/tbb ..
~~~

Intel MKL-DNN has limited optimizations for Intel TBB and has some functional
limitations if built with Intel TBB.

Functional limitations:
* Winograd convolution algorithm is not supported.

The following primitives have lower performance compared to OpenMP (mostly due
to limited parallelism):
* Batch normalization,
* Convolution backward by weights,
* Inner product,
* `mkldnn_*gemm()`.
