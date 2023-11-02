# RFC for Graph API: construct tensor without user handle

## Introduction & Motivation

Currently, in Graph API level, users need to manually allocate memory and manage
its lifecycle before constructing a tensor. Incorrect allocation and de-allocation
methods may cause some unexpected errors, such as memory release errors caused
by `test::vector`(use one context to allocate memory and another context to de-allocate
it). In fact, in scenarios like unit test, the library can take the control of
allocating and de-allocating memory by itself, since the size of the memory can be
obtained based on the logical tensor and layout, memory allocation can be performed
accordingly when a tensor is constructed. When a tensor is destroyed, the memory is
de-allocated. In order to make library more user-friendly, this RFC mainly implements
the ability to construct tensor without user-provided handle.


## Proposal

### Option 1

Referring to the `dnnl_memory_create` API in the primitive, reuse the existed
`dnnl_graph_tensor_create` API. When `handle` parameter is set to `DNNL_MEMORY_ALLOCATE`,
the library will perform memory allocation and automatically release the memory
when the current tensor is destroyed. When the `handle` parameter is a regular pointer,
the library will only set the pointer. When the `handle` parameter is `DNNL_MEMORY_NONE`,
the handle of tensor will set `nullptr`.


### API

```c
/// include/oneapi/dnnl/dnnl_graph.h

/// Creates a tensor with logical tensor, engine, and data handle.
///
/// @param tensor Output tensor.
/// @param logical_tensor Description for this tensor.
/// @param engine Engine to use.
/// @param handle Handle of the memory buffer to use as an underlying storage.
///     - A pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the tensor. In this case the library
///       owns the buffer.
///     - DNNL_MEMORY_NONE to create tensor without an underlying buffer.
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_tensor_create(dnnl_graph_tensor_t *tensor,
        const dnnl_graph_logical_tensor_t *logical_tensor, dnnl_engine_t engine,
        void *handle);
```

```c++
/// include/oneapi/dnnl/dnnl_graph.hpp:

/// Constructs a tensor object according to a given logical tensor, an
/// engine, and a memory handle.
///
/// @param lt The given logical tensor
/// @param aengine Engine to store the data on.
/// @param handle Handle of memory buffer to use as an underlying storage.
///     - A pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the tensor. In this case the library
///       owns the buffer.
///     - DNNL_MEMORY_NONE to create tensor without an underlying buffer.
tensor(const logical_tensor &lt, const engine &aengine, void *handle) {
    dnnl_graph_tensor_t t = nullptr;
    error::wrap_c_api(
            dnnl_graph_tensor_create(&t, &(lt.data), aengine.get(), handle),
            "could not create tensor object with the logical_tensor, "
            "engine, and handle");
    reset(t);
}
```

### Example

Below is pseudo-code that demonstrates the use of the C and C++ API.
```c
// Create an engine
dnnl_engine_t engine;
dnnl_engine_create(&engine, dnnl_cpu, 0);
// Create a logical tensor
dnnl_graph_logical_tensor_t lt;
const size_t id = 0;
dnnl_dims_t dims = {1, 2, 3, 4};
// Init logical tensor
dnnl_graph_logical_tensor_init_with_dims(&lt, id, dnnl_f32, 4, dims, 
    dnnl_graph_layout_type_strided, dnnl_graph_tensor_property_undef);
// Create a tensor
dnnl_graph_tensor_t tensor;
dnnl_graph_tensor_create(&tensor, &lt, engine, DNNL_MEMORY_ALLOCATE);
// Destroy the tensor
dnnl_graph_tensor_destroy(tensor);
// Destroy the engine
dnnl_engine_destroy(engine);
```

```c++
// Create an engine
dnnl::engine eng {dnnl::engine::kind::cpu, 0};
// Create a logical tensor
logical_tensor lt {0, data_type::f32, logical_tensor::dims {3, 4, 5, 6},
        layout_type::strided};
// Create tensor without data handle
tensor t(lt, eng, DNNL_MEMORY_ALLOCATE);
```
