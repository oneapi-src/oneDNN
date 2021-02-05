Build Options {#dev_guide_build_options}
====================================

oneDNN supports the following build-time options.

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| DNNL_LIBRARY_TYPE           | **SHARED**, STATIC                  | Defines the resulting library type
| DNNL_CPU_RUNTIME            | **OMP**, TBB, SEQ, THREADPOOL, DPCPP| Defines the threading runtime for CPU engines
| DNNL_GPU_RUNTIME            | **NONE**, OCL, DPCPP                | Defines the offload runtime for GPU engines
| DNNL_BUILD_EXAMPLES         | **ON**, OFF                         | Controls building the examples
| DNNL_BUILD_TESTS            | **ON**, OFF                         | Controls building the tests
| DNNL_ARCH_OPT_FLAGS         | *compiler flags*                    | Specifies compiler optimization flags (see warning note below)
| DNNL_ENABLE_CONCURRENT_EXEC | ON, **OFF**                         | Disables sharing a common scratchpad between primitives in #dnnl::scratchpad_mode::library mode
| DNNL_ENABLE_JIT_PROFILING   | **ON**, OFF                         | Enables [integration with performance profilers](@ref dev_guide_profilers)
| DNNL_ENABLE_PRIMITIVE_CACHE | **ON**, OFF                         | Enables [primitive cache](@ref dev_guide_primitive_cache)
| DNNL_ENABLE_MAX_CPU_ISA     | **ON**, OFF                         | Enables [CPU dispatcher controls](@ref dev_guide_cpu_dispatcher_control)
| DNNL_ENABLE_CPU_ISA_HINTS   | **ON**, OFF                         | Enables [CPU ISA hints](@ref dev_guide_cpu_isa_hints)
| DNNL_VERBOSE                | **ON**, OFF                         | Enables [verbose mode](@ref dev_guide_verbose)
| DNNL_AARCH64_USE_ACL        | ON, **OFF**                         | Enables integration with Arm Compute Library for AArch64 builds
| DNNL_BLAS_VENDOR            | **NONE**, ARMPL                     | Defines an external BLAS library to link to for GEMM-like operations
| DNNL_GPU_VENDOR             | **INTEL**, NVIDIA                   | Defines GPU vendor for GPU engines

All other building options or values that can be found in CMake files are intended for
development/debug purposes and are subject to change without notice.
Please avoid using them.

## Common options

## CPU Options
Intel Architecture Processors and compatible devices are supported by
oneDNN CPU engine. The CPU engine is built by default and cannot
be disabled at build time.

### Targeting Specific Architecture
oneDNN uses JIT code generation to implement most of its functionality
and will choose the best code based on detected processor features. However,
some oneDNN functionality will still benefit from targeting a specific
processor architecture at build time. You can use `DNNL_ARCH_OPT_FLAGS` CMake
option for this.

For Intel(R) C++ Compilers, the default option is `-xSSE4.1`, which instructs
the compiler to generate the code for the processors that support SSE4.1
instructions. This option would not allow you to run the library on
older processor architectures.

For GNU\* Compilers and Clang, the default option is `-msse4.1`.

@warning
While use of `DNNL_ARCH_OPT_FLAGS` option gives better performance, the
resulting library can be run only on systems that have instruction set
compatible with the target instruction set. Therefore, `ARCH_OPT_FLAGS`
should be set to an empty string (`""`) if the resulting library needs to be
portable.

### Runtime CPU dispatcher control
oneDNN JIT relies on ISA features obtained from the processor it is being run
on.  There are situations when it is necessary to control this behavior at
run-time to, for example, test SSE4.1 code on an AVX2-capable processor. The
`DNNL_ENABLE_MAX_CPU_ISA` build option controls the availability of this
feature. See @ref dev_guide_cpu_dispatcher_control for more information.

### Runtime CPU ISA hints
For performance reasons, sometimes oneDNN JIT needs to be provided with extra
hints so as to prefer or avoid particular CPU ISA feature. For example, one
might want to disable Zmm registers usage in order to take advantage of higher
clock speed. The `DNNL_ENABLE_CPU_ISA_HINTS` build option makes this feature
available at runtime. See @ref dev_guide_cpu_isa_hints for more information.

### Runtimes
CPU engine can use OpenMP, Threading Building Blocks (TBB) or sequential
threading runtimes. OpenMP threading is the default build mode. This behavior
is controlled by the `DNNL_CPU_RUNTIME` CMake option.

#### OpenMP
oneDNN uses OpenMP runtime library provided by the compiler.

@warning
Because different OpenMP runtimes may not be binary-compatible, it's important
to ensure that only one OpenMP runtime is used throughout the application.
Having more than one OpenMP runtime linked to an executable may lead to
undefined behavior including incorrect results or crashes. However as long as
both the library and the application use the same or compatible compilers there
would be no conflicts.

#### Threading Building Blocks (TBB)
To build oneDNN with TBB support, set `DNNL_CPU_RUNTIME` to `TBB`:

~~~sh
$ cmake -DDNNL_CPU_RUNTIME=TBB ..
~~~

Optionally, set the `TBBROOT` environmental variable to point to the TBB
installation path or pass the path directly to CMake:

~~~sh
$ cmake -DDNNL_CPU_RUNTIME=TBB -DTBBROOT=/opt/intel/path/tbb ..
~~~

oneDNN has functional limitations if built with TBB:
* Winograd convolution algorithm is not supported for fp32 backward
  by data and backward by weights propagation.

#### Threadpool
To build oneDNN with support for threadpool threading, set `DNNL_CPU_RUNTIME` to
`THREADPOOL`

~~~sh
$ cmake -DDNNL_CPU_RUNTIME=THREADPOOL ..
~~~

The `_DNNL_TEST_THREADPOOL_IMPL` CMake variable controls which of the three
threadpool implementations would be used for testing: `STANDALONE`, `TBB`, or
`EIGEN`. The latter two require also passing `TBBROOT` or `Eigen3_DIR` paths
to CMake. For example:

~~~sh
$ cmake -DDNNL_CPU_RUNTIME=THREADPOOL -D_DNNL_TEST_THREADPOOL_IMPL=EIGEN -DEigen3_DIR=/path/to/eigen/share/eigen3/cmake ..
~~~

Threadpool threading support is experimental and has the same limitations as
TBB plus more:
* As threadpools are attached to streams which are only passed during
  primitive execution, work decomposition is performed statically at the
  primitive creation time. At the primitive execution time, the threadpool is
  responsible for balancing the static decomposition from the previous item
  across available worker threads.

### AArch64 Options

oneDNN includes experimental support for Arm 64-bit Architecture (AArch64).
By default, AArch64 builds will use the reference implementations throughout.
The following options enable the use of AArch64 optimised implementations
for a limited number of operations, provided by AArch64 libraries.

| AArch64 build configuration           | CMake Option              | Environment variables                         | Dependencies
| :---                                  | :---                      | :---                                          | :---
| Arm Compute Library based primitives  | DNNL_AARCH64_USE_ACL=ON   | ACL_ROOT_DIR=*Arm Compute Library location*   | [Arm Compute Library](https://github.com/ARM-software/ComputeLibrary)
| Vendor BLAS library support           | DNNL_BLAS_VENDOR=ARMPL    | None                                          | [Arm Performance Libraries](https://developer.arm.com/tools-and-software/server-and-hpc/downloads/arm-performance-libraries)

#### Arm Compute Library
Arm Compute Library is an open-source library for machine learning applications.
The development repository is available from
[mlplatform.org](https://review.mlplatform.org/#/admin/projects/ml/ComputeLibrary),
and releases are also available on [GitHub](https://github.com/ARM-software/ComputeLibrary).
The `DNNL_AARCH64_USE_ACL` CMake option is used to enable Compute Library integration:

~~~sh
$ cmake -DDNNL_AARCH64_USE_ACL=ON ..
~~~

This assumes that the environment variable `ACL_ROOT_DIR` is
set to the location of Arm Compute Library, which must be downloaded and built
independently of oneDNN.

@warning
For a debug build of oneDNN it is advisable to specify a Compute Library build
which has also been built with debug enabled.

@warning
oneDNN is only compatible with Compute Library builds v20.11 or later.

#### Vendor BLAS libraries
oneDNN can use a standard BLAS library for GEMM operations.
The `DNNL_BLAS_VENDOR` build option controls BLAS library selection, and
defaults to `NONE`. For AArch64 builds with GCC, use the
[Arm Performance Libraries](https://developer.arm.com/tools-and-software/server-and-hpc/downloads/arm-performance-libraries):

~~~sh
$ cmake -DDNNL_BLAS_VENDOR=ARMPL ..
~~~

Additional options available for development/debug purposes. These options are
subject to change without notice, see
[`cmake/options.cmake`](https://github.com/oneapi-src/oneDNN/blob/master/cmake/options.cmake)
for details.

## GPU Options
Intel Processor Graphics is supported by oneDNN GPU engine. GPU engine
is disabled in the default build configuration.

### Runtimes
To enable GPU support you need to specify the GPU runtime by setting
`DNNL_GPU_RUNTIME` CMake option. The default value is `"NONE"` which
corresponds to no GPU support in the library.

#### OpenCL\*
OpenCL runtime requires Intel(R) SDK for OpenCL\* applications. You can
explicitly specify the path to the SDK using `-DOPENCLROOT` CMake option.

~~~sh
$ cmake -DDNNL_GPU_RUNTIME=OCL -DOPENCLROOT=/path/to/opencl/sdk ..
~~~
