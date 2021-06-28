# USM Support for OpenCL Runtime

## Background

OpenCL specification defines two memory kinds that OpenCL supports:
1. OpenCL buffer and image - are represented as a memory object `cl_mem`
2. Shared Virtual Memory (SVM) - is a pointer represented memory

Some time ago Intel introduced an OpenCL extension named Unified Shared Memory
(USM). The USM supports three types of memory:
* Device
* Host
* Shared

Similarly to SVM the USM is also a pointer represented memory but unlike SVM user
can explicitly specify where memory should be placed to achieve the best possible
performance. In the case of SVM it's up to the driver to decide where the
memory will be placed. Since the decision is based on driver heuristics it may or
may not be a good one.

Presently, the only memory kind that oneDNN supports for OpenCL runtime is
OpenCL buffers.

## Motivation

OpenVINO requested adding support for USM memory kind for OpenCL runtime in oneDNN.
This will provide the following benefits:
1. When OpenVINO GPU plugin works with USM memory there is no need to copy data
from USM to OpenCL buffer
2. Explicit copying data from host to OpenCL buffer works significantly (several
times) slower than implicit data copying when using USM


## API

Since USM is an OpenCL specific feature the OpenCL interoperability API should be
extended to provide users with a way to create oneDNN memory for different memory
kinds. In addition, an API to query memory kind from a oneDNN memory object should
be provided.

### OpenCL Buffer vs Memory Object

OpenCL specification defines a term `memory object` which means an object for
containing data. The `memory object` can represent either OpenCL buffer or OpenCL
image.

For some reason the existing interoperability API such as `dnnl_ocl_interop_memory_set_mem_object` has `mem_object` in its name. Perhaps,
it was done that way to allow to add support for OpenCL images in the future if
needed.

Because of introducing USM memory kind there should be introduced memory kinds
to distinguish one memory model from another. There are two options regarding
the name of the memory kinds.

#### Option 1: Use `usm` and `buffer` in the names

Pros:
* Allows to explicitly specify memory kind, that is `cl_mem` in oneDNN is always
an OpenCL buffer
* Aligned with SYCL runtime

Cons:
* The name of existing API for getting a memory object from oneDNN memory becomes
a bit inconsistent

#### Option 2: Use `usm` and `mem_object` in the names

Pros:
* Aligned with existing API for getting a memory object from oneDNN memory

Cons:
* Doesn't describe what the memory object really is

Proposal is to go with option 1 because memory kind should be descriptive. In the
case there is a need to add support for OpenCL images in the future the corresponding
memory kind will be added.

### C API

```c
/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the
/// constructed memory object will have the underlying `buffer set. In this
/// case, the buffer will be initialized as if:
/// - dnnl_memory_set_data_handle() has been called, if @p memory_kind is equal
///   to dnnl_ocl_interop_usm, or
/// - dnnl_ocl_interop_memory_set_mem_object() has been called, if @p memory_kind
///   is equal to dnnl_ocl_interop_buffer.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param memory_kind Memory allocation kind to specify the type of handle.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       dnnl_ocl_interop_usm.
///     - A pointer to OpenCL buffer. In this case the library doesn't own the
///       buffer. Requires @p memory_kind be equal to be equal to
///       dnnl_ocl_interop_buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer that corresponds to the memory allocation kind
///       @p memory_kind for the memory object. In this case the library
///       owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to
///       create memory object without an underlying buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_ocl_interop_memory_create(dnnl_memory_t *memory,
        const dnnl_memory_desc_t *memory_desc, dnnl_engine_t engine,
        dnnl_ocl_interop_memory_kind_t memory_kind, void *handle);
```

```cpp
/// Returns the memory allocation kind associated with a memory object.
///
/// @param memory Memory to query.
/// @param memory_kind Output underlying memory allocation kind of the memory
///     object.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_ocl_interop_memory_get_memory_kind(const_dnnl_memory_t memory,
        dnnl_ocl_interop_memory_kind_t *memory_kind);
```

### C++ API

```cpp
/// Creates a memory object.
///
/// Unless @p handle is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the
/// constructed memory object will have the underlying buffer set. In this
/// case, the buffer will be initialized as if:
/// - dnnl::memory::set_data_handle() had been called, if @p memory_kind is
///   equal to dnnl::ocl_interop::memory_kind::usm, or
/// - dnnl::ocl_interop::set_buffer() has been called, if @p memory_kind is
///   equal to dnnl::ocl_interop::memory_kind::buffer.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param kind Memory allocation kind to specify the type of handle.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A USM pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer. Requires @p memory_kind to be equal to
///       dnnl::ocl_interop::memory_kind::usm.
///     - A pointer to OpenCL buffer. In this case the library doesn't own the
///       buffer. Requires @p memory_kind be equal to be equal to
///       dnnl::ocl_interop::memory_kind::buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer that corresponds to the memory allocation kind
///       @p memory_kind for the memory object. In this case the library
///       owns the buffer.
///     - The DNNL_MEMORY_NONE specific value. Instructs the library to
///       create memory object without an underlying buffer.
///
/// @returns Created memory object.
inline memory make_memory(const memory::desc &memory_desc,
        const engine &aengine, memory_kind kind,
        void *handle = DNNL_MEMORY_ALLOCATE);
```

```cpp
/// Constructs a memory object from an OpenCL buffer.
///
/// @param memory_desc Memory descriptor.
/// @param aengine Engine to use.
/// @param abuffer An OpenCL buffer to use.
///
/// @returns Created memory object.
memory make_memory(const memory::desc &memory_desc, const engine &aengine,
        cl_mem &mem_object);
```

```cpp
/// Returns the memory allocation kind associated with a memory object.
///
/// @param amemory A memory object.
///
/// @returns The underlying memory allocation kind of the memory object.
inline memory_kind get_memory_kind(const memory &amemory);
```
### Header for OpenCL Types

Since OpenCL interoperability API doesn't have memory kind type the one will be
introduced in `dnnl_ocl_types.h`.

```c
/// Memory allocation kind.
typedef enum {
    /// USM (device, shared, host, or unknown) memory allocation kind.
    dnnl_ocl_interop_usm,
    /// Buffer memory allocation kind.
    dnnl_ocl_interop_buffer,
} dnnl_ocl_interop_memory_kind_t;
```

For C++ API there will be introduced `memory_kind` enum class in `dnnl_ocl.hpp`.
```cpp
namespace ocl_interop {

/// Memory allocation kind.
enum class memory_kind {
    /// USM (device, shared, host, or unknown) memory allocation kind.
    usm = dnnl_ocl_interop_usm,
    /// Buffer memory allocation kind.
    buffer = dnnl_ocl_interop_buffer,
};
}
```

## Default Memory Kind

oneDNN specifies the default memory kind that is used when creating oneDNN memory
via runtime agnostic API. In the case of SYCL runtime the default memory kind is
USM.

Since OpenCL runtime will also support two memory kinds there is a need to define
the default memory kind.

### Option 1: Keep OpenCL Buffer

Pros:
* No impact on users

Cons:
* Performance with OpenCL buffer may be inferior to USM

### Option 2: Change to USM

Pros:
* Can be faster than OpenCL buffers
* Aligned with SYCL runtime
* The USM model will be used by OpenVINO therefore extensive testing USM via benchdnn
is required. For example, USM is tested with benchdnn and gtest and OpenCL buffer
is tested with gtests, similarly to SYCL

Cons:
* Some user can be affected functional-wise, e.g. if they use `dnnl_ocl_interop_memory_get_mem_object` or `dnnl_memory_get_data_handle()`
and cast the obtained pointer to `cl_mem`
* While USM seems to be more performant memory kind users can hypothetically be
affected performance-wise for the cases that we are not aware of

Proposal is to go with option 1 to preserve user visible behavior. Organize
testing so that USM is used for benchdnn and gtest but testing OpenCL buffer will
be performed using gtests only.

## Memory Kind for Scratchpad

If the default memory kind remains OpenCL buffer then no action required.
If it changes to USM then OpenCL buffer should be used for scratchpad anyway to
manage its lifetime properly.

To use USM for scratchpad there should be implemented reference counting for
GPU primitives.

## Implementation Details

OpenCL provides a mechanism for adding platform specific extension functions
without exposing them to the OpenCL API. Instead, OpenCL provides an API to get
a pointer to the extension function by its name. (see `clGetExtensionFunctionAddressForPlatform`). This is very similar to obtaining an address of a symbol in a shared library.
To use USM with OpenCL there should be added corresponding utility functions that
are responsible for obtaining those extensions functions and for executing them.

### USM Utility Functions

```cpp
namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {
namespace usm {

// The values below introduced in `cl_ext.h` OpenCL header. To avoid dependency
// on the OpenCL header there should be introduced enum that describes USM kinds.
// #define CL_MEM_TYPE_UNKNOWN_INTEL       0x4196
// #define CL_MEM_TYPE_HOST_INTEL          0x4197
// #define CL_MEM_TYPE_DEVICE_INTEL        0x4198
// #define CL_MEM_TYPE_SHARED_INTEL        0x4199

enum class ocl_usm_kind_t {
    unknown,
    host,
    device,
    shared
};

// Some functions below take an engine so that they can query platform and get
// the corresponding extension function.

// Some functions below take a stream so that they can query an engine and platform
// as well as OpenCL queue to execute the extension functions.

// Functions to allocate/free USM buffers of different USM kinds.
void *malloc_host(engine_t *engine, size_t size);
void *malloc_device(engine_t *engine, size_t size);
void *malloc_shared(engine_t *engine, size_t size);
void free(engine_t *engine, void *ptr);

// Function to set USM memory as a kernel argument.
status_t set_kernel_arg_usm(engine_t *engine, cl_kernel kernel, int arg_index,
        const void *arg_value);

// Functions for initializing/copying USM memory.
status_t memcpy(stream_t *stream, void *dst, const void *src, size_t size);
status_t memset(stream_t *stream, void *ptr, int value, size_t size);
status_t fill(stream_t *stream, void *ptr, const void *pattern,
        size_t pattern_size, size_t size);

// Function to get USM kind for the given USM pointer.
ocl_usm_kind_t get_pointer_type(engine_t *engine, const void *ptr);

} // namespace usm
} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl
```

### Memory Storage

OpenCL runtime uses `ocl_memory_storage_t` abstraction to abstract OpenCL buffer
away. Since there will be two different memory kinds there should be implemented
memory storage abstractions for each of them.

Those abstractions will have a common base class that can be used to identify memory
kind of the derived memory storage.

```cpp
class ocl_memory_storage_base_t : public memory_storage_t {
public:
    virtual memory_kind_t memory_kind() const = 0;
};
```

The existing `ocl_memory_storage_t` will be renamed to `ocl_buffer_memory_storage_t`.
The new `ocl_usm_memory_storage_t` abstraction will be implemented to support USM
memory kind.
