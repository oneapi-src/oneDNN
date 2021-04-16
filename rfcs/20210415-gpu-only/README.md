# Proposal on introducing GPU only configurations


## Motivation
oneDNN supports multiple CPU and GPU runtimes. While it's possible to build
oneDNN only for a CPU runtime the GPU runtime always requires a CPU runtime to be
specified. For some customers the size of the oneDNN library matters a lot therefore
if the customers only need a GPU runtime then having a redundant CPU runtime
becomes a problem. One of the customers is OpenVINO whose request is the main motivation
for introducing an ability to build oneDNN only for GPU runtime.

## Proposal
The proposal is to introduce the following GPU only configurations:
* gpu_dpcpp (`DNNL_CPU_RUNTIME=NONE DNNL_GPU_RUNTIME=DPCPP`)
* gpu_ocl (`DNNL_CPU_RUNTIME=NONE DNNL_GPU_RUNTIME=OCL`)

We need to support them both because the `gpu_dpcpp` configuration is expected to
be the main one for GPU but currently OpenVINO is still using OpenCL runtime.
Fortunately, the vast majority of the code is shared between OCL and DPCPP runtimes
therefore enabling both should not increase amount of work significantly.

Currently, there is no plan to include these configurations in oneAPI as well as
enable and test GPU only configuration for NVIDIA.


### API

oneDNN provides some APIs which are CPU specific or implemented only for CPU.
There are two options to deal with it:

#### Option 1: Removing CPU Specific API
* Inside the library the API will be guarded by macros (e.g `DNNL_GPU_ONLY`)
* The `dnnl.h` and `dnnl.hpp` headers can be configured by CMake in the following way:
    ```cpp
    // dnnl.h

    #define DNNL_GPU_ONLY 1 // this macro is substituted by CMake during configuration
    // ... //
    #if DNNL_GPU_ONLY
    dnnl_status_t DNNL_API dnnl_set_max_cpu_isa(dnnl_cpu_isa_t isa);
    #endif
    // ... //
    #undef DNNL_GPU_ONLY // end of dnnl.h
    ```
* Basically, this means that a user will not be able to call any CPU specific API

#### Option 2: API Stays Intact
* Inside the library the API will be adjusted to return a status in the case of GPU
only configuration
* The status can be consistent for all CPU specific API or can be different
depending on the API
* For example, for `dnnl_set_max_cpu_isa` oneDNN
can return `runtime_error` because it's a dedicated API for CPU but for
`dnnl_gemm_s8s8s32` oneDNN can return `unimplemented`.
* The GTest will be adjusted to make sure that such APIs return proper status

The suggestion is to go with the second option as this is better aligned with the
runtime agnostic approach that oneDNN uses for almost all APIs. Also, it allows
applications that use CPU specific API under a runtime condition to link with
oneDNN.

### Cross-Engine Reorder

Since CPU engine is not available in GPU only configuration the cross-engine
reorder capability will be limited to GPU -> GPU case.

OpenVINO mentioned that they used the cross-engine reorder to avoid dealing with
runtime directly when there is a need to create `dnnl::memory` for GPU from a
host pointer. For GPU only configurations users can use `map`/`unmap` functionality
 to copy host data to GPU.

There is also an option to improve usability for such scenarios by introducing the
following API.

SYCL interoperability API:

```cpp
// dnnl_sycl.hpp

namespace dnnl {
namespace sycl_interop {

/// The two `make_memory` functions can be used to create a memory object that
/// contains a copy of data from the buffer @p host_ptr is pointing to.
/// @param host_ptr - is a host accessible pointer
inline memory make_memory(const void *host_ptr, const memory::desc &memory_desc,
    const engine &aengine, memory_kind kind, void *handle = DNNL_MEMORY_ALLOCATE);

template <typename T, int ndims = 1>
memory make_memory(const void *host_ptr, const memory::desc &memory_desc,
    const engine &aengine, cl::sycl::buffer<T, ndims> &abuffer);

}
}
```

OpenCL interoperability API:
```c
// dnnl_ocl.h

// Similar to the non-buffer `make_memory` version.
dnnl_status_t DNNL_API dnnl_ocl_interop_memory_create_from_host_ptr(
        const void *host_ptr, dnnl_memory_t *memory,
        const dnnl_memory_desc_t *memory_desc, dnnl_engine_t engine,
        void *handle);
```

This API will also be useful if current API will be split in the future because in
that case cross-reorder cannot be supported as well.

However, this can be postponed until there is a request.

### Threading CPU Runtime

While the CPU runtime is absent there is still a need for CPU threading runtime
because all threading functionality (e.g `parallel_nd`), which is used in the
testing, is using it. There are two options to handle threading in the testing:

#### Option 1: Enable CPU Threading Runtime in the Library
* Enable TBB threading for DPC++ GPU only build
* Enable OpenMP threading for OpenCL GPU only build

Note: the cases when OpenMP runtime is not available (e.g. for Xcode Clang) should
fall back to sequential runtime.

#### Option 2: Implement (reuse) Threading Layer for Testing
* Implement (reuse) functionality such as `parallel_nd` and other that is used
in the testing
* The library will use sequential CPU threading runtime

The common places in the library where threading functionality is used is
`memory_debug`, `memory_zero_pad` and `primitive_hashing`.
* The `memory_debug` is only used for CPU therefore sequential CPU threading
runtime doesn't affect it in the case of GPU only configurations
* The `memory_zero_pad` is not used as there is a dedicated `zero_pad` GPU kernel
that does the job. Therefore this is not affected as well
* The `primitive_hashing` only uses that functionality to get number of threads to
differentiate different CPU implementations. This is also not affected by sequential
threading runtime

Also, the advantage of the second option over the first one is that the library
doesn't have a dependency on a threading runtime.

Implementation of the Option 1 is zero cost but it comes with the dependency on
threading runtime. It doesn't seem to be a problem for customers because the
compilers have those threading runtimes. The suggestion is to start with
Option 1 and get rid of the dependency (Option 2) if there is a request.

### Testing
The main validation tool is benchdnn. The benchdnn heavily relies on reorders
meaning that it creates several reorders for each test case. Creation of GPU
primitives is expensive which significantly affects testing time. Currently,
benchdnn has an optimization for GPU primitives to reduce that time. The
optimization is to use CPU reorders which are created much faster. The problem
is that there are no CPU reorders in GPU only configurations. There are two options
to deal with it.

#### Option 1: Use Only GTests for Validating GPU Only Configurations
* The GPU only configurations differ from CPU + GPU ones only in the removed CPU
part of the code. The GPU part stays intact
* The existing CPU + GPU configurations will be tested in CI and Nightly to validate
correctness/performance
* The GPU only configurations will be tested by GTest to make sure that API and
infrastructure work properly as this is the main parts affected by removing
CPU runtime

#### Option 2: Use Full Validation
* Using `cl_cache` allows to achieve the same time as with CPU reorders (except
benchdnn drivers that use `fast-ref-gpu`)
* The `fast-ref-gpu` and  `cpu-isa-hints` options are ignored for GPU only
configurations
* Currently, benchdnn provides a `gemm` function for `f32` data type. The function
calls oneDNN `dnnl_sgemm`. This function is used in `ip`, `wino` and `rnn`
validation. A simple reference implementation of `sgemm` should be added to
support those benchdnn drivers
* In the future, when API will be split there will be an ability to link a oneDNN
with CPU runtime and use CPU functionality too speed up testing

Since full validation is performed for CPU + GPU configurations in CI and Nightly
and performance testing the suggestion is to go with the Option 1. If there will
be a lot bug escapes the option 2 can be implemented to improve validation.

In any case, the GTests should be adjusted to validate CPU specific API properly.
