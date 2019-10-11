Build Options {#dev_guide_build_options}
====================================

DNNL supports the following build-time options.

| Option                       | Supported values (defaults in bold)  | Description
| :---                         | :---                                 | :---
| DNNL_LIBRARY_TYPE            | **SHARED**, STATIC                   | Defines the resulting library type
| DNNL_CPU_RUNTIME             | **OMP**, TBB                         | Defines the threading runtime for CPU engines
| DNNL_GPU_RUNTIME             | **NONE**, OCL                        | Defines the offload runtime for GPU engines
| DNNL_BUILD_EXAMPLES          | **ON**, OFF                          | Controls building the examples
| DNNL_BUILD_TESTS             | **ON**, OFF                          | Controls building the tests
| DNNL_ARCH_OPT_FLAGS          | *compiler flags*                     | Specifies compiler optimization flags (see warning note below)
| DNNL_ENABLE_CONCURRENT_EXEC  | ON, **OFF**                          | Disables sharing a common scratchpad between primitives in #dnnl::scratchpad_mode::library mode
| DNNL_ENABLE_JIT_PROFILING    | **ON**, OFF                          | Enables integration with Intel(R) VTune(TM) Amplifier
| DNNL_ENABLE_PRIMITIVE_CACHE  | ON, **OFF**                          | Enables primitive cache

All other building options that can be found in CMake files are dedicated for
the development/debug purposes and are subject to change without any notice.
Please avoid using them.

## Common options

### Primitive cache
Primitive cache is disabled in the default build configuration.

To enable the primitive cache you can use `DNNL_ENABLE_PRIMITIVE_CACHE` CMake option.
The default value is `"OFF"`.

## CPU Options
Intel Architecture Processors and compatible devices are supported by
DNNL CPU engine. The CPU engine is built by default and cannot
be disabled at build time.

### Targeting Specific Architecture
DNNL uses JIT code generation to implement most of its functionality
and will choose the best code based on detected processor features. However,
some DNNL functionality will still benefit from targeting a specific
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

### Runtimes
CPU engine can use OpenMP or TBB threading runtime. OpenMP threading
is the default build mode. This behavior is controlled by the `DNNL_CPU_RUNTIME`
CMake option.

#### OpenMP
DNNL uses OpenMP runtime library provided by the compiler.

@warning
Because different OpenMP runtimes may not be binary-compatible, it's important
to ensure that only one OpenMP runtime is used throughout the application.
Having more than one OpenMP runtime linked to an executable may lead to
undefined behavior including incorrect results or crashes. However as long as
both the library and the application use the same or compatible compilers there
would be no conflicts.

#### TBB
To build DNNL with TBB support, set the `TBBROOT` environmental
variable to point to the TBB installation path or pass the path directly to
cmake:

~~~sh
$ cmake -DDNNL_CPU_RUNTIME=TBB -DTBBROOT=/opt/intel/path/tbb ..
~~~

DNNL has limited optimizations for Intel TBB and has some functional
limitations if built with Intel TBB.

Functional limitations:
* Winograd convolution algorithm is not supported for fp32 backward
by data and backward by weights propogation.

## GPU Options
Intel Processor Graphics is supported by DNNLs GPU engine. GPU engine
is disabled in the default build configuration. 

### Runtimes
To enable GPU support you need to specify the GPU runtime by setting
`DNNL_GPU_RUNTIME` CMake option. The default value is `"NONE"` which
corresponds to no GPU support in the library.

#### OpenCL\*
OpenCL runtime requires Intel(R) SDK for OpenCL\* applications. You can
explicitly specify the path to the SDK using `-DOPENCLROOT` CMake option.

~~~sh
cmake -DDNNL_GPU_RUNTIME=OCL -DOPENCLROOT=/path/to/opencl/sdk ..
~~~
