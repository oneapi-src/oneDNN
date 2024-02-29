# Support OpenCL runtime in Graph API

## Motivation

OpenVINO relies on OpenCL runtime and integrates oneDNN primitive through the
OpenCL interop API under `dnnl::ocl_interop::` namespace for GPU integration.
But currently, oneDNN does not expose OpenCL interop API for Graph API and does
not support building Graph API with OpenCL GPU runtime [[#1]][1]. To support
integrating oneDNN Graph API in OpenVINO and exposing MHA optimization through
oneDNN Graph API (as discussed in a separate RFC [[#2]][2]), this RFC proposes
to add OpenCL runtime support in oneDNN Graph API. The API proposals, library
changes, and potential limitations will be discussed in the following sections.

## Proposals

### Build system

Currently, when building oneDNN Graph API (which is turned on by default), the
GPU runtime (`ONEDNN_GPU_RUNTIME`) needs to be either `NONE` or `SYCL`. The
build system will throw an error if the user builds oneDNN Graph API with OpenCL
GPU runtime (`OCL`) [[#1]][1].

It's proposed to remove the check of OpenCL GPU runtime in oneDNN build system
and allow building oneDNN Graph API with OpenCL GPU runtime. This should not
break any existing oneDNN integrations. In OpenVINO, currently oneDNN Graph API
is turned off explicitly via a CMake build option (`ONEDNN_BUILD_GRAPH=OFF`)
when building oneDNN with OpenCL runtime.

### DNNL backend

The existing DNNL backend of oneDNN Graph API is using oneDNN primitive to
deliver performance and fusion capabilities. For GPU, the DNNL backend relies on
SYCL GPU runtime and calls primitive's SYCL interop API under
`dnnl::sycl_interop::` namespace.

To support OpenCL GPU runtime, we need to extend the DNNL backend by respecting
the GPU runtime build flag and call corresponding interop API accordingly. More
specifically, when the library is built with OpenCL GPU runtime, the DNNL
backend needs to create oneDNN GPU engine and stream with OpenCL runtime objects
(cl_device, cl_context, cl_event, and cl_command_queue, etc.) and execute
compiled partitions upon them.

oneDNN has the same post-ops and data types support for SYCL runtime and OpenCL
runtime on GPU. Given that, we can reuse the existing GPU fusion patterns
defined in DNNL backend for both SYCL runtime and OpenCL runtime.

### OpenCL interop API

#### Allocator

As oneDNN Graph API exposes allocator interfaces for users to simplify memory
allocation and management in the library. To support OpenCL GPU runtime, we
propose to add new allocator interfaces accepting OpenCL runtime objects.

Below C APIs will be defined and exposed when the library is built with OpenCL
GPU runtime.

Same as the existing Graph API for SYCL interop, only USM is supported for now.

Same as the existing Graph API for SYCL device and native CPU device, allocator
is optional. If allocator is not provided by the user, a default allocator will
be used and managed by the library.

```c
/// Allocation call-back function interface for OCL. OCL allocator should be
/// used for OCL runtime. The call-back should return a USM device memory pointer.
typedef void *(*dnnl_graph_ocl_allocate_f)(
        size_t size, size_t alignment, cl_device_id device, cl_context context);

/// Deallocation call-back function interface for OCL. OCL allocator should be
/// used for OCL runtime. The call-back should release a USM device memory returned
/// by #dnnl_graph_ocl_allocate_f.
typedef void (*dnnl_graph_ocl_deallocate_f)(
        void *buf, cl_device_id device, cl_context context, cl_event *event);

/// Creates an allocator object with the allocation and deallocation call-back
/// function pointers provided by the user.
///
/// @param allocator Output allocator
/// @param ocl_malloc A pointer to OCL malloc function
/// @param ocl_free A pointer to OCL free function
/// @returns #dnnl_success on success and a status describing the
///     error otherwise.
dnnl_status_t DNNL_API dnnl_graph_ocl_interop_allocator_create(
        dnnl_graph_allocator_t *allocator, dnnl_graph_ocl_allocate_f ocl_malloc,
        dnnl_graph_ocl_deallocate_f ocl_free);
```

Corresponding C++ API will be added to a new namespace
`dnnl::graph::ocl_interop::`.

```c++
namespace dnnl {
namespace graph {
namespace ocl_interop {

/// Constructs an allocator from OCL malloc and free function pointers. OCL
/// allocator should be used for OCL runtime. Currently, only device USM
/// allocator is supported.
///
/// @param ocl_malloc The pointer to OCL malloc function
/// @param ocl_free The pointer to OCL free function
/// @returns Created allocator
inline allocator make_allocator(dnnl_graph_ocl_allocate_f ocl_malloc,
        dnnl_graph_ocl_deallocate_f ocl_free) {
    dnnl_graph_allocator_t c_allocator = nullptr;
    error::wrap_c_api(dnnl_graph_ocl_interop_allocator_create(
                              &c_allocator, ocl_malloc, ocl_free),
            "could not create allocator for ocl runtime");
    return allocator(c_allocator);
}

}
}
}
```

#### Engine

We also need to extend oneDNN engine creation to support OpenCL GPU runtime.

Below C APIs will be defined and exposed when the library is built with OpenCL
GPU runtime.

```c
/// This API is a supplement for existing oneDNN engine API:
/// dnnl_status_t DNNL_API dnnl_ocl_interop_engine_create(
///     dnnl_engine_t *engine, cl_device_id device, cl_context context);
dnnl_status_t DNNL_API dnnl_graph_ocl_interop_make_engine_with_allocator(
        dnnl_engine_t *engine, cl_device_id device, cl_context context,
        const_dnnl_graph_allocator_t alloc);

/// This API is a supplement for existing oneDNN engine API:
/// dnnl_status_t DNNL_API dnnl_ocl_interop_engine_create_from_cache_blob(
///     dnnl_engine_t *engine, cl_device_id device, cl_context context,
///     size_t size, const uint8_t *cache_blob);
dnnl_status_t DNNL_API dnnl_graph_ocl_interop_make_engine_from_cache_blob_with_allocator(
        dnnl_engine_t *engine, cl_device_id device, cl_context context,
        const_dnnl_graph_allocator_t alloc, size_t size, const uint8_t *cache_blob);
```

Corresponding C++ API will be added to a new namespace
`dnnl::graph::ocl_interop::`.

```c++
namespace dnnl {
namespace graph {
namespace ocl_interop {

/// Create an engine with OpenCL device, context, and allocator.
inline engine make_engine_with_allocator(
        cl_device_id device, cl_context context, const allocator &alloc) {
    dnnl_engine_t c_engine;
    error::wrap_c_api(dnnl_graph_ocl_interop_make_engine_with_allocator(
                              &c_engine, device, context, alloc.get()),
            "could not make an engine with allocator for ocl runtime");
    return engine(c_engine);
}

/// Create an engine from a cache blob along with OpenCL device, context,
/// and allocator.
inline engine make_engine_with_allocator(
        cl_device_id device, cl_context context, const allocator &alloc,
        const std::vector<uint8_t> &cache_blob) {
    dnnl_engine_t c_engine;
    error::wrap_c_api(dnnl_graph_ocl_interop_make_engine_from_cache_blob_with_allocator(
                              &c_engine, device, context, alloc.get(),
                              cache_blob.size(), cache_blob.data()),
            "could not make an engine with allocator for ocl runtime");
    return engine(c_engine);
}

}
}
}
```

#### Stream

It is not needed to define OpenCL interop API for stream in Graph API. We can
rely on the existing `make_stream` API in `dnnl::ocl_interop::` below.

```c++
namespace dnnl {
namespace ocl_interop {

/// Constructs an execution stream for the specified engine and OpenCL queue.
///
/// @param aengine Engine to create the stream on.
/// @param queue OpenCL queue to use for the stream.
/// @returns An execution stream.
inline stream make_stream(const engine &aengine, cl_command_queue queue) {
    dnnl_stream_t c_stream;
    error::wrap_c_api(
            dnnl_ocl_interop_stream_create(&c_stream, aengine.get(), queue),
            "could not create a stream");
    return stream(c_stream);
}

}
}
```

#### Execution

Corresponding to the `sycl::event` returned by compiled partition execution
under SYCL runtime, for OpenCL runtime, we need to expose new APIs to return
`cl_event`.

Below C APIs will be defined and exposed when the library is built with OpenCL
GPU runtime.

```c
/// Execute a compiled partition with ocl runtime.
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param stream The stream used for execution
/// @param num_inputs The number of input tensors
/// @param inputs A list of input tensors
/// @param num_outputs The number of output tensors
/// @param outputs A non-empty list of output tensors
/// @param deps Optional handle of list with `cl_event` dependencies.
/// @param ndeps Number of dependencies.
/// @param ocl_event The handle of cl_event returned by the API.
/// @returns #dnnl_success on success and a status describing the
///     error otherwise.
dnnl_status_t DNNL_API dnnl_graph_ocl_interop_compiled_partition_execute(
        const_dnnl_graph_compiled_partition_t compiled_partition,
        dnnl_stream_t stream, size_t num_inputs,
        const_dnnl_graph_tensor_t *inputs, size_t num_outputs,
        const_dnnl_graph_tensor_t *outputs, const cl_event *deps, int ndeps, cl_event *ocl_event);
```

C++ API in namespace `dnnl::graph::ocl_interop::`.

```c++
namespace dnnl {
namespace graph {
namespace ocl_interop {

/// Executes a compiled partition in a specified stream and returns a OCL
/// event.
///
/// @param c_partition Compiled partition to execute.
/// @param astream Stream object to run over
/// @param inputs Arguments map.
/// @param outputs Arguments map.
/// @param deps Optional vector with `cl_event` dependencies.
/// @returns Output event.
inline cl_event execute(compiled_partition &c_partition, stream &astream,
        const std::vector<tensor> &inputs, std::vector<tensor> &outputs,
        const std::vector<cl_event> &deps = {}) {
    // call the C API dnnl_graph_ocl_interop_compiled_partition_execute to
    // implement the logic.
}

}
}
}
```

### Testing

Benchdnn graph driver needs to be enhanced to support OpenCL GPU runtime
validation [[#3]][3] by respecting the GPU runtime build flag and call OpenCL
interop API or SYCL interop API respectively.

Since oneDNN has the same fusion capability for SYCL runtime and OpenCL runtime
on GPU, the existing input files [[#4]][4] designed for SYCL GPU validation should be
enough and can be reused to validate OpenCL GPU.

### Limitations

- Following the convention of SYCL GPU runtime in Graph API, only USM is
  supported for OpenCL GPU runtime.
- Primitive selection (`DNNL_ENABLE_PRIMITIVE`) and workload selection
  (`DNNL_ENABLE_WORKLOAD`) are not supported by Graph API as before. We mention
  them here as the features were initially requested by OpenVINO for primitive
  API.
- GPU vendor other than `INTEL` is not supported by Graph API as before. We can
  extend to support other vendors once the requirement pops up.

## References

1. https://github.com/oneapi-src/oneDNN/blob/de69d44024ab4f64b20deb7aa066a65c867f1123/src/CMakeLists.txt#L136
2. https://github.com/oneapi-src/oneDNN/pull/1745
3. https://github.com/oneapi-src/oneDNN/blob/de69d44024ab4f64b20deb7aa066a65c867f1123/tests/benchdnn/graph/utils.cpp#L77
4. https://github.com/oneapi-src/oneDNN/tree/de69d44024ab4f64b20deb7aa066a65c867f1123/tests/benchdnn/inputs/graph

[1]: https://github.com/oneapi-src/oneDNN/blob/de69d44024ab4f64b20deb7aa066a65c867f1123/src/CMakeLists.txt#L136
[2]: https://github.com/oneapi-src/oneDNN/pull/1745
[3]: https://github.com/oneapi-src/oneDNN/blob/de69d44024ab4f64b20deb7aa066a65c867f1123/tests/benchdnn/graph/utils.cpp#L77
[4]: https://github.com/oneapi-src/oneDNN/tree/de69d44024ab4f64b20deb7aa066a65c867f1123/tests/benchdnn/inputs/graph
