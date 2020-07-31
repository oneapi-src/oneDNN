# Using Arm Compute Library kernels on AArch64 (RFC)

## Introduction

The goal for this RFC is to demonstrate an approach for providing optimized
implementations of machine learning operators in oneDNN, on AArch64, based on
the optimized implementations provided by the open-source [Arm Compute Library
(ArmCL)](https://github.com/arm-software/ComputeLibrary).

Here we outline the approach, and detail a functioning forward convolution
implementation which is integrated with ArmCL at the runtime level. This is
presented as a basis for future work and includes a minimal set of changes and
additions to oneDNN's `src/cpu` directory, and the introduction of ArmCL as a
dependency to the existing AArch64 builds.

### Motivation

Considerable progress has been made recently to enable oneDNN builds on non-x64
platforms, including AArch64:

- [#658](https://github.com/oneapi-src/oneDNN/pull/658),
followed by [#685](https://github.com/oneapi-src/oneDNN/pull/685)
and [#694](https://github.com/oneapi-src/oneDNN/pull/693)
enabled oneDNN to build natively on AArch64 machines.

- ["DNNL CPU Code Organisation Adjustments"](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20200401-cpu-dir-structure)
outlined significant changes to the CPU code organisation, which
were implemented with [#698](https://github.com/oneapi-src/oneDNN/pull/698)
and included in the oneDNN v1.5 release.

- [#694](https://github.com/oneapi-src/oneDNN/pull/694) and
[#712](https://github.com/oneapi-src/oneDNN/pull/712) added support for
open-source, public, AArch64 CI using
[Drone CI](https://cloud.drone.io/oneapi-src/oneDNN).

- [#741](https://github.com/oneapi-src/oneDNN/pull/741) exposed support for
external 'vendor' BLAS libraries on AArch64, specifically Arm Performance
Libraries
[(ArmPL)](https://developer.arm.com/tools-and-software/server-and-hpc/compile/arm-compiler-for-linux/arm-performance-libraries).
This was extended to cover the recent
[freely-downloadable release of ArmPL](https://developer.arm.com/tools-and-software/server-and-hpc/downloads/arm-performance-libraries)
and added to the build documentation with [#741](https://github.com/oneapi-src/oneDNN/pull/741).


### Background

ArmCL is an open-source library for computer vision and machine learning
applications and is actively developed by a dedicated team at Arm. The
development repository is available from
[mlplatform.org](https://review.mlplatform.org/#/admin/projects/ml/ComputeLibrary),
and releases are also available on [GitHub](https://github.com/ARM-software/ComputeLibrary).
Unlike ArmPL, which targets HPC workloads, it has support for bfloat16,
fixed-point, int8 etc.

At present, ArmCL targets primarily mobile/edge SoCs, but work is ongoing to
remove bottlenecks to performance on infrastructure scale CPUs and high core
counts.

ArmCL's library architecture consists of the Core, Runtime and Graph Libraries.
The Core library contains low-level optimized implementations of the algorithms
while the Runtime library is a wrapper to the Core that provides memory
management and multi-threading interfaces. The graph library provides a
high-level API that can be used to easily chain neural network layers.
Further details are
in the Compute Library [documentation](https://github.com/ARM-software/ComputeLibrary/wiki/Documentation).

This initial implementation will demonstrate the integration at the runtime
level of ArmCL, using an Arm GEMM-based 2D convolution in the forward
propagation phase. This means that, in the inference regime, convolution calls
will be supported by ArmCL, and in the training regime only the forward pass
will be supported; the backward pass will be computed using the existing
reference implementations.


## Proposal

The source code for the initial implementation integrating oneDNN and ArmCL is
available [here](https://github.com/diaena/oneDNN/tree/armcl-integration-experimental).
The key changes required are listed below:

- Adding a dependency on ArmCL at the build stage;
- Adding AArch64 specific code inside a new directory `src/cpu/aarch64`;
- Adding a `CPU_INSTANCE_AARCH64` macro;
- Adding ArmCL operations, integrated at runtime-level.


### Adding a dependency on ArmCL at the build stage:

To build oneDNN with ArmCL, it must be assumed that a pre-built ArmCL
library is available, built from the [ArmCL
sources](https://github.com/ARM-software/ComputeLibrary). Additional
CMake modules will be needed (`FindARMCL.cmake`, `ArmCL.cmake`) which will only
impact AArch64 builds. The following steps will be required to get a oneDNN and
ArmCL build:

- The user, having built ArmCL, will need to set an environment variable
specifying ArmCL's root directory:
`export ARMCL_ROOT_DIR=/path/to/ComputeLibrary`;

- Addition of a CMake option to enable ArmCL builds: `DNNL_AARCH64_USE_ARMCL=1`,
for example, `cmake --verbose .. -DDNNL_AARCH64_USE_ARMCL=1`;

- For a debug build, the user will be required to separately build the debug
version of ArmCL, which will be entirely independent of oneDNN's CMake flags.


### Adding AArch64 specific code inside a new directory `src/cpu/aarch64`:

A new directory, `src/cpu/aarch64`, will be required for AArch64 specific code,
this is consistent with the library restructuring implemented in
[#698](https://github.com/oneapi-src/oneDNN/pull/698). The new directory would
be integrated into the CMake build system with the following changes:

- Addition of new source directory `src/cpu/aarch64`,
- Addition of a new `CMakeList.txt` in `src/cpu/aarch64` to point to the
GEMM-based sources,
- Update `src/cpu/CMakeList.txt` to conditionally include `src/cpu/aarch64`
based on `DNNL_TARGET_AARCH64` flag.
- Update `src/cpu/cpu_convolution_list.cpp` to conditionally include new headers
from `src/cpu/aarch64` using the existing `DNNL_AARCH64` macro.


### Adding a `CPU_INSTANCE_AARCH64` macro:

This ensures that this kernel is called for AArch64 builds only.

- Update the existing `DNNL_AARCH_ONLY` macro in `src/cpu/platform.hpp`
that activates with the `DNNL_AARCH` macro:

  ~~~c++
  #define DNNL_AARCH64_ONLY(...) Z_CONDITIONAL_DO(DNNL_AARCH64, __VA_ARGS__)
  ~~~

- Define the CPU instance macro for AArch64 in `src/cpu/cpu_engine.hpp`:

  ~~~c++
    #define CPU_INSTANCE_AARCH64(...) DNNL_AARCH64_ONLY(CPU_INSTANCE(__VA_ARGS__))
  ~~~

- Enclose the call to ArmCL GEMM convolution kernel inside the macro in
`src/cpu/cpu_convolution_list.hpp`:

  ~~~c++
    CPU_INSTANCE_AARCH64(arm_gemm_convolution_fwd_t)
  ~~~


### Adding ArmCL operation integrated at runtime-level:

To integrate ArmCL, new source files, `gemm_arm_convolution.cpp` and
`gemm_arm_convolution_utils.cpp`, together with header files
`gemm_arm_convolution.hpp` and `gemm_arm_convolution_utils.cpp` are
required in `src/cpu/aarch64`, the new directory created within `src/cpu`.

In runtime level integration, the kernels
used for the GEMM-based convolution operation are wrapped with the ArmCL runtime
into the function
[NEGEMMConvolutionLayer](https://arm-software.github.io/ComputeLibrary/v20.05/classarm__compute_1_1_n_e_g_e_m_m_convolution_layer.xhtml#details)
which is called and instantiated inside `gemm_arm_convolution.cpp`, for example
`acl_gemm_conv` is an instantiation of the `NEGEMMConvolutionLayer` class in
ArmCL:

~~~c++
acl_data_.acl_gemm_conv.run();
~~~

The
[`run()`](https://arm-software.github.io/ComputeLibrary/v20.05/classarm__compute_1_1_n_e_g_e_m_m_convolution_layer.xhtml#ad1717410afd0be936c6213a63c8005fb)
method in turn calls several other kernels, including data transformation
functions, for example, `im2col` and `col2im` within ArmCL, and hence does not
use oneDNN's variant of these functions. ArmCL's run-time library handles the
execution of the kernel (see,
[`CPPScheduler.cpp`](https://arm-software.github.io/ComputeLibrary/v20.05/classarm__compute_1_1_c_p_p_scheduler.xhtml#details)).

Before this run method is called, ArmCL's data (`acl_data_`) has to be
initialized inside the corresponding `gemm_arm_convolution.hpp` file using
`TensorShape` and `TensorInfo` objects:

~~~c++
// Input tensor descriptor
const arm_compute::TensorShape acl_src_shape(jcp_.iw, jcp_.ih,
                   jcp_.ic, jcp_.mb);
const arm_compute::TensorInfo acl_src_info(acl_src_shape, 1,
      arm_compute::DataType::F32, arm_compute::DataLayout::NCHW);
...
// Set padding and stride information
                const arm_compute::PadStrideInfo conv_info(
                    jcp_.stride_w, jcp_.stride_h, jcp_.l_pad, jcp_.t_pad);
...
// Initialise tensors based on tensor information
acl_data_->acl_src.allocator()->init(acl_src_info);
~~~

For each data tensor, these objects store metadata regarding the dimensions of
the data tensors, such as height, width, input and output channels, and
information about data type and memory formats required by ArmCL (Note: at
present, the implementation presented here covers only fp32). Tensors are
initialized based on the metadata provided in these objects and do not involve
additional memory.

Configuration of the ArmCL kernel primitive is done inside
`gemm_arm_convolution.hpp`, using the `acl_gemm_conv.configure()` method, which
receives the metadata about ArmCL data tensors, padding, strides, and post_ops
information from the previous steps. Inside
[`configure`](https://arm-software.github.io/ComputeLibrary/v20.05/classarm__compute_1_1_n_e_g_e_m_m_convolution_layer.xhtml#a97f4fd717623515cacaa206a889933ce)
calls may be made to an
[`allocate()`](https://arm-software.github.io/ComputeLibrary/v20.05/src_2runtime_2_tensor_allocator_8cpp_source.xhtml#l00133)
function which is independent of the memory allocations made by oneDNN's
`jit_gemm_convolution_utils::init_conf ` in scratchpad memory, and hence, will
not incur memory overheads compared to the current oneDNN implementation.

These data initialization and kernel configuration steps are carried out inside
the `init` method in `gemm_arm_convolution.hpp` with internal checks to
ensure that there is a supported implementation available.
In line with the established mechanism, unsupported configurations
return 'status::unimplemented'.

After the initialization phase, oneDNN data is imported into ArmCL's data
tensors (`acl_data_`) in `gemm_arm_convolution.cpp`, using the
[`import_memory`](https://arm-software.github.io/ComputeLibrary/v20.05/classarm__compute_1_1_tensor_allocator.xhtml#a84052cebf66a6126051a166a078253a4)
method which does not perform any memory manipulations but instead consists
of creating a couple of shared pointers and void pointers that point to oneDNN
memory:

~~~c++
status_t arm_gemm_convolution_fwd_t::execute_forward(
       const exec_ctx_t &ctx) const {
    auto src_base = CTX_IN_MEM(data_t *, DNNL_ARG_SRC);
...
acl_data_.acl_src.allocator()->import_memory(const_cast<data_t *>(src_base));
~~~


## Limitations

The implementation outlined in this RFC has a number of limitations which could
be addressed in future PRs:

- Only the forward propagation regime is supported (dnnl\_forward\_training,
dnnl\_forward\_inference, dnnl\_forward\_scoring);
- Only Conv2D is supported. There is no support for Conv3D, Conv1D or depthwise
convolutions;
- Grouped convolutions are currently not supported in ArmCL;
- Only fp32 datatype is supported at the moment;
- ArmCL requires all the tensors to be in the same data layout, NCHW or NHWC.
Blocked memory formats are not supported;
- Only one eltwise operation is supported, activation

## Results/validation
oneDNN's standard test suites were used for validation. Some of the tests there
compare the output with a reference variant like:

~~~c++
compare_data<data_t_dst>(dst_ref, c_dst, 1e-2);
~~~

and fail if it is incorrect (for example,
[test\_convolution\_eltwise\_forward](https://github.com/oneapi-src/oneDNN/blob/master/tests/gtests/test_convolution_eltwise_forward_common.hpp#L215-L221),
[test\_deconvolution](https://github.com/oneapi-src/oneDNN/blob/master/tests/gtests/test_deconvolution.cpp#L423-L429),
[test\_convolution\_forward\_f32](https://github.com/oneapi-src/oneDNN/blob/master/tests/gtests/test_convolution_forward_common.hpp#L206-L212) etc.). ArmCL-base convolution is
called in a number of tests with output validation, covering cases with
strides > 1,
dilations, padding ([test\_global\_scratchpad](https://github.com/oneapi-src/oneDNN/blob/master/tests/gtests/test_global_scratchpad.cpp#L93-L94)
also checks the case with asymmetric padding), different activation functions.
It passes these tests, meaning than it
provides the same functionality as `CPU_INSTANCE(gemm_convolution_fwd_t)` for
the same launch parameters:

```
./tests/gtests/test_convolution_forward_f32
./tests/gtests/test_convolution_eltwise_forward_f32
./tests/gtests/test_deconvolution
./tests/gtests/test_global_scratchpad
```

Overall, all 98 oneDNN tests passed, the testing was performed on AArch64 CPU with `ctest`
and the build

```
cmake [...] -DCMAKE_BUILD_TYPE=Release -DDNNL_AARCH64_USE_ARMCL=1
```


## Expected performance benefits

This proposal is intended to lay the groundwork to enable the development of
more performant AArch64 implementation which leverages the ArmCL library. We
expect the ongoing work to optimize ArmCL's implementations on infrastructure
scale CPUs and high core counts may expose opportunities to adapt ArmCL's API in
the longer term. Preliminary tests demonstrated noticeable increase of
performance in single-threaded regime on Thunder X2 AArch64 CPU.

## Limited impact

The scope of this RFC, and the PRs outlined below, is limited such that:

- There are no changes to the API;
- There is no impact on non-AArch64 builds.

## Implementation plan

We propose staging the implementation as follows:

- Step 1: introduce CMake changes to allow inclusion of ArmCL as a dependency;
- Step 2: changes to `src/cpu/platform.hpp` to support new AARCH64 macros;
- Step 3: addition of ArmCL based implementation into `src/cpu/aarch64` and
integration into `src/cpu/cpu_convolution_list.cpp`.

These steps could be implemented as separate commits in one PR, or as individual
PRs.

With this initial implementation in place, future PRs will focus on adding
missing functionality, improvements to the work scheduling in multi-threading
regime, implementation of more ArmCL operators, more user-friendly integration
with ArmCL during the build and, of course, optimization.
