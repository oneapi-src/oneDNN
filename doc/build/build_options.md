Build Options {#dev_guide_build_options}
====================================

## Configuring the Build

Intel MKL-DNN supports the following build-time options.

| Option                      | Supported values (defaults in bold)  | Description
| :---                        | :---                                 | :---
| MKLDNN_LIBRARY_TYPE         | **SHARED**, STATIC                   | Defines the resulting library type
| MKLDNN_THREADING            | **OMP**, OMP:INTEL, OMP:COMP, TBB    | Defines the threading type
| MKLDNN_USE_MKL              | **DEF**, NONE, ML, FULL, FULL:STATIC | Defines the binary dependency on Intel MKL
| MKLDNN_GPU_BACKEND          | **NONE**, OPENCL                     | Defines the backend for GPU engine
| MKLDNN_BUILD_EXAMPLES       | **ON**, OFF                          | Controls building the examples
| MKLDNN_BUILD_TESTS          | **ON**, OFF                          | Controls building the tests
| MKLDNN_ARCH_OPT_FLAGS       | *compiler flags*                     | Specifies compiler optimization flags (see warning note below)
| MKLDNN_ENABLE_JIT_PROFILING | **ON**, OFF                          | Enables integration with Intel(R) VTune(TM) Amplifier

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

## Threading

Intel MKL-DNN can use the OpenMP or TBB threading runtimes. OpenMP threading
is the default build mode and is recommended for the best performance. TBB
support is experimental. This behavior is controlled by the `MKLDNN_THREADING`
CMake option.

### OpenMP

Intel MKL-DNN can use the Intel, GNU, or Clang OpenMP runtimes. Because
different OpenMP runtimes may not be binary-compatible, it's important to
ensure that only one OpenMP runtime is used throughout the application. Having
more than one OpenMP runtime linked to an executable may lead to undefined
behavior including incorrect results or crashes.

The Intel MKL-DNN library linked with Intel MKL (full or small libraries
version) link to the Intel OpenMP runtime included with the Intel MKL. The
Intel OpenMP runtime is binary compatible with the GNU OpenMP and Clang OpenMP
runtimes and is recommended for the best performance results.

If Intel MKL-DNN library is built standalone, it will link to the OpenMP
library supplied by the compiler. This means that there would be no conflicts
as long as both the library and the application use the same or compatible
compilers.

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

## Linking to Intel(R) MKL

Intel MKL-DNN can be configured to use Intel MKL via the `MKLDNN_USE_MKL`
CMake option. Using Intel MKL may improve performance on older platforms not
supported by the matrix-matrix multiplication routine implemented in Intel
MKL-DNN itself.

If you choose to build Intel MKL-DNN with the binary dependency, download the
Intel MKL small libraries using the provided script:

* Linux/macOS
~~~sh
cd scripts && ./prepare_mkl.sh && cd ..
~~~

* Windows
~~~bat
cd scripts && call prepare_mkl.bat && cd ..
~~~

You can also download the small libraries from
[GitHub release section](https://github.com/intel/mkl-dnn/releases). You need
to unpack the archive in the `external` directory of the repository root or
set `MKLROOT` environment variable to point to the installation location of
the small libraries of full Intel MKL library.

@note
Intel MKL small libraries are currently only compatible with OpenMP threading.
You need the full Intel MKL library to use Intel MKL and TBB threading.  Using
Intel MKL or Intel MKL small libraries will introduce additional runtime
dependencies. For additional information, refer to the Intel MKL
[system requirements](https://software.intel.com/en-us/articles/intel-math-kernel-library-intel-mkl-2019-system-requirements).

@warning
If the library is built with TBB threading, the user is expected to set the
`MKL_THREADING_LAYER` environment variable to either `tbb` or `sequential` in
order to force Intel MKL to use Intel TBB for parallelization or to be
sequential, respectively.  Without this setting, Intel MKL (the `mkl_rt`
library) uses the OpenMP threading by default.
