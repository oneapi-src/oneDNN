# Introducing Sparse Memory

## Objectives

* Introduce a concept of sparse memory to oneDNN
* Introduce a Matrix Multiplication implementation that supports CSR encoding

Note: all introduced API and functionality will be experimental. In order to enable the
it the user will need to specify a CMake `ONEDNN_EXPERIMENTAL_SPARSE=ON` option.

## Motivation

Using sparse data in the deep learning workloads is becoming more popular because
it may significantly improve performance of the workloads due to reducing
memory traffic and possibly skipping unnecessary computations. And all that can
be done with an acceptable accuracy loss or even without it.

A common approach to using sparse data in the workloads is to sparsify weights
during training and use them during inference. Using sparse weights for
inference has been proven to be a viable optimization for some workloads (e.g. BERT).

Additionally, some optimized operations for sparse data are essential for accelerating
graph neural networks (GNNs) that are in active research stage now.

## Sparse Encodings

Sparse data can be encoded using different storage schemas (encodings) that requires
oneDNN to have a flexible and scalable API to support them.

In general, there are two kinds of encodings:
1. Hardware agnostic encodings such as CSR/CSC, BCSR/BCSC, COO, etc. These encodings
can be efficiently used for different hardware
2. Hardware aware encodings. This type of encodings can be efficiently used for certain
hardware that provides specialized instructions for sparse data. For example, AVX-512
instruction set includes `VPCOMPRESS*` and `VPEXPAND*` instructions that can be used to
compress and decompress sparse data, which would reduce memory traffic and therefore
provide some performance benefits. The instructions usually dictate how the sparse
data should be encoded.

## Proposal

The proposed API will be flexible and scalable enough to cover all needed encodings
but the only encoding that will be supported in the beginning is CSR. Also, there will
be introduced a matrix multiplication implementation that supports input tensors
encoded with the CSR encoding.

The general idea for enabling sparse memory is to have a memory object that can have multiple
underlying buffers (i.e. handles). This way it can support non-trivial data representations
such as sparse encodings. The ultimate goal is to have a single general API that can work with
any memory objects and memory descriptors. This is why almost all proposed C API will have a
suffix `v2` and C++ API will have either a new overloaded function/constructor or a default
argument.

The general idea for enabling the API for sparse memory and memory descriptor is to
extend the existing API by adding an additional parameter - `index` that would specify
what handle we call the API for.

### Sparse Memory Descriptor

Currently, the memory descriptor can only describe a memory object that has a single
underlying buffer. The proposal is to add a new format kind - `sparse` and a new
format descriptor - `sparse_desc_t` to describe a memory object with multiple
underlying handles.

Note: `sparse_desc_t` is an implementation detail and is not exposed to the user.

```cpp
/// Sparse encodings.
typedef enum {
    /// Undefined sparse encoding kind, used for empty memory descriptors.
    dnnl_sparse_encoding_undef = 0,
    /// Compressed Sparse Row (CSR) encoding.
    dnnl_csr,
} dnnl_sparse_encoding_t;

struct sparse_desc_t {
    static constexpr int max_metadata_types = 2;
    // Sparse encoding.
    sparse_encoding_t encoding;
    // Number of non-zero entries.
    dnnl_dim_t nnz;
    // Metadata types. Each encoding defines how to interpret these.
    // - CSR: 0th - index data type
    //        1st - pointer data type
    dnnl_data_type_t metadata_types[max_metadata_types];
};
```
#### C API

Since different encodings may have different parameters oneDNN will provide a separate API
for creating memory descriptors for different encodings. This approach allows us to cover
as many encodings as needed without complicating the API.

Below is API for creating a memory descriptor for CSR encoding.
```c
/// Creates a memory descriptor for CSR encoding.
///
/// The created memory descriptor will describe a memory object that
/// contains 3 buffers. The buffers have the following meaning and
/// assigned numbers (index):
///  - 0: values
///  - 1: indices
///  - 2: pointers
///
/// @param memory_desc Output memory descriptor.
/// @param ndims Number of dimensions
/// @param dims Array of dimensions.
/// @param data_type Elements data type.
/// @param nnz Number of non-zero entries.
/// @param indices_dt Data type of indices.
/// @param pointers_dt Data type of pointers.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_memory_desc_create_with_csr_encoding(
        dnnl_memory_desc_t *memory_desc, int ndims, const dnnl_dims_t dims,
        dnnl_data_type_t data_type, dnnl_dim_t nnz, dnnl_data_type_t indices_dt,
        dnnl_data_type_t pointers_dt);
```

The API to query size for values and metadata.
```c
/// Returns the size of the data that corresponds to the given index.
///
/// @param memory_desc Memory descriptor.
/// @param index Index of the buffer.
///
/// @returns The number of bytes required for the requested data.
size_t dnnl_memory_desc_get_size_v2(const_dnnl_memory_desc_t memory_desc, int index);
```

Each encoding defines an index interpretation (specified in the documentation). For example,
for CSR: [0] values, [1] indices, [2] pointers.

Usage example:
```c
const size_t values_size = dnnl_memory_desc_get_size_v2(csr_md, 0),;
const size_t indices_size = dnnl_memory_desc_get_size_v2(csr_md, 1);
const size_t pointers_size = dnnl_memory_desc_get_size_v2(csr_md, 2);
char *values = malloc(values_size);
char *indices = malloc(indices_size);
char *pointers = malloc(pointers_size);
// ... //
```

#### C++ API

The `memory::desc` will get a set of functions to create memory descriptors for different
sparse encodings in order to avoid issues with overloading `memory::desc(...)` and
provide a convenient API. The name of the functions will be aligned with the name of
encodings they create the memory descriptor for.

```cpp
struct memory {
    struct desc {
        /// Function for creating a memory descriptor for CSR sparse encoding.
        ///
        /// The created memory descriptor will describe a memory object that
        /// contains 3 buffers. The buffers have the following meaning and
        /// assigned numbers (index):
        ///  - 0: values
        ///  - 1: indices
        ///  - 2: pointers
        ///
        /// @param adims Tensor dimensions.
        /// @param adata_type Data precision/type.
        /// @param nnz Number of non-zero entries.
        /// @param index_dt Data type of indices.
        /// @param pointer_dt Data type of pointers.
        /// @param allow_empty A flag signifying whether construction is
        ///     allowed to fail without throwing an exception. In this case a
        ///     zero memory descriptor will be constructed. This flag is
        ///     optional and defaults to false.
        static desc csr(const dims &adims, data_type adata_type, dim nnz,
                data_type index_dt, data_type pointer_dt, bool allow_empty = false);
    };
};
```

Usage example:
```cpp
const int nnz = 12;
auto md_csr = memory::desc::csr({64, 128}, f32, nnz, s32, s32);
```

#### Submemory/Reshape/Permute axis
Reshape and permute axis functionality is defined in frameworks for some encodings.
It's possible to support it in oneDNN as well but the use cases for reshape/permute for
sparse memory descriptors are not clear yet.

Submemory seems to be more tricky. It doesn't seem to be possible to reuse
the current approach because `offset0` is a scalar. Feasibility of submemory
functionality for sparse memory descriptor has to be explored.

Since the functionality doesn't seem to be essential the suggestion is to consider it as
separate features that can be defined and implemented by request.

### Sparse Memory

As it was mentioned before the semantics of the memory object will be extended to support
multiple underlying handles therefore the following API should be implemented:
* API for creating a memory object for multiple user-provided buffers
* API for interacting with a particular buffer

#### C API

The runtime agnostic API to create a sparse memory.
```c
/// Creates a memory object with multiple handles.
///
/// @param memory Output memory object.
/// @param memory_desc Memory descriptor.
/// @param engine Engine to use.
/// @param nhandles Number of handles.
/// @param handles Handles of the memory buffers to use as underlying storages.
///     For each element of the @p handles array the following applies:
///     - A pointer to the user-allocated buffer. In this case the library
///       doesn't own the buffer.
///     - The DNNL_MEMORY_ALLOCATE special value. Instructs the library to
///       allocate the buffer for the memory object. In this case the library
///       owns the buffer.
///     - DNNL_MEMORY_NONE Instructs the library to skip allocation of the
///       memory buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_memory_create_v2(dnnl_memory_t *memory,
        const_dnnl_memory_desc_t memory_desc, dnnl_engine_t engine,
        int nhandles, void **handles);
```
Usage example:
```c
const int nhandles = 3;
void *handles[nhandles] = {};

// Initialize handles to create a sparse memory object with the provided buffers.
handles[0] = (void *)values;
handles[1] = (void *)indices;
handles[2] = (void *)pointers;

// Initialize handles to create a sparse memory object with the provided metadata buffers.
handles[0] = DNNL_MEMORY_NONE;
handles[1] = (void *)indices;
handles[2] = (void *)pointers;

// Create a sparse memory object for the given buffers.
dnnl_memory_create_v2(csr_mem, csr_md, engine, nhandles, handles);
```

The rest of the API is for interacting with the buffers. A particular buffer can be queried
using the corresponding index.

The API to do map/unmap:
```c
/// Maps a memory object and returns a host-side pointer to a memory buffer
/// with a copy of its contents. The memory buffer corresponds to the given
/// index.
///
/// Mapping enables explicit direct access to memory contents for the engines
/// that do not support it implicitly.
///
/// Mapping is an exclusive operation - a memory object cannot be used in
/// other operations until this memory object is unmapped.
///
/// @note
///     Any primitives working with @p memory should be completed before
///     the memory is mapped. Use dnnl_stream_wait to synchronize the
///     corresponding execution stream.
///
/// @note
///     The dnnl_memory_map_data_v2() and dnnl_memory_unmap_data_v2() functions are
///     mainly provided for debug and testing purposes, and their performance
///     may be suboptimal.
///
/// @param memory Memory object.
/// @param mapped_ptr Output pointer to the mapped buffer.
/// @param index Index of the buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t  dnnl_memory_map_data_v2(const_dnnl_memory_t memory, void **mapped_ptr, int index);

/// Unmaps a memory object and writes back any changes made to the previously
/// mapped memory buffer. The pointer to the mapped buffer must be obtained
/// via the dnnl_memory_map_data() call. The buffer corresponds to the given
/// index.
///
/// @note
///     The dnnl_memory_map_data_v2() and dnnl_memory_unmap_data_v2() functions are
///     mainly provided for debug and testing purposes, and their performance
///     may be suboptimal.
///
/// @param memory Memory object.
/// @param mapped_ptr Pointer to the mapped buffer that must have been
///     obtained using the dnnl_memory_map_data() function.
/// @param index Index of the buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t  dnnl_memory_unmap_data_v2(const_dnnl_memory_t memory, void *mapped_ptr, int index);
```
Usage example:
```c
void *values_buffer = NULL;
dnnl_memory_map_data_v2(csr_mem, &values_buffer, 0);
dnnl_memory_unmap_data_v2(csr_mem, values_buffer, 0);
```
The API to get/set data handle:
```c
/// Returns an underlying memory buffer that corresponds to the given index.
///
/// @param memory Memory object.
/// @param handle Data handle. For the CPU engine or when USM is used, the
///     memory buffer is a pointer to the actual data. For OpenCL it is a
///     `cl_mem`.
/// @param index Index of the buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t  dnnl_memory_get_data_handle_v2(
        const_dnnl_memory_t memory, void **handle, int index);

/// Sets an underlying memory buffer that corresponds to the given index.
///
/// @param memory Memory object.
/// @param handle Data handle. For the CPU engine or when USM is used, the
///     memory buffer is a pointer to the actual data. For OpenCL it is a
///     `cl_mem`.
/// @param index Index of the buffer.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_set_data_handle_v2(
        dnnl_memory_t memory, void *handle, int index);
```
Usage example:
```c
void *indices_buffer;
dnnl_memory_get_data_handle_v2(csr_mem, &indices_buffer, 1);
void *values_buffer = ...;
dnnl_memory_set_data_handle_v2(csr_mem, values_buffer, 0);
```

#### C++ API

The `memory` class will get one new constructor that takes a vector of handles.
```cpp
struct memory {
    /// Constructs a memory object with multiple handles.
    ///
    /// Unless @p handle is equal to #DNNL_MEMORY_NONE, the constructed memory
    /// object will have the underlying buffer set. In this case, the buffer
    /// will be initialized as if #dnnl::memory::set_data_handle() had been
    /// called.
    ///
    /// @sa memory::set_data_handle()
    ///
    /// @param md Memory descriptor.
    /// @param aengine Engine to store the data on.
    /// @param handles Handles of the memory buffers to use.
    ///     For each element of the @p handles vector the following applies:
    ///     - A pointer to the user-allocated buffer. In this case the library
    ///       doesn't own the buffer.
    ///     - The #DNNL_MEMORY_ALLOCATE special value. Instructs the library to
    ///       allocate the buffer for the memory object. In this case the
    ///       library owns the buffer.
    ///     - #DNNL_MEMORY_NONE Instructs the library to skip allocation of the
    ///       memory buffer.
    memory(const desc &md, const engine &aengine, std::vector<void *> handles);
    // ... //
};
```
Usage example:
```cpp
/// Create a memory with the user-provided buffers.
auto csr_mem = memory(csr_md, eng, {values, indices, pointers});

/// Create a memory with library allocated buffers.
auto csr_mem = memory(csr_md, eng);
```

The API to do map/unmap.
```cpp
    /// Maps a memory object and returns a host-side pointer to a memory
    /// buffer with a copy of its contents. The memory buffer corresponds to
    /// the given index.
    ///
    /// Mapping enables read/write directly from/to the memory contents for
    /// engines that do not support direct memory access.
    ///
    /// Mapping is an exclusive operation - a memory object cannot be used in
    /// other operations until it is unmapped via #dnnl::memory::unmap_data()
    /// call.
    ///
    /// @note
    ///     Any primitives working with the memory should be completed before
    ///     the memory is mapped. Use #dnnl::stream::wait() to synchronize the
    ///     corresponding execution stream.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be suboptimal.
    ///
    /// @tparam T Data type to return a pointer to.
    /// @param index Index of the buffer. Defaults to 0.
    /// @returns Pointer to the mapped memory.
    template <typename T = void>
    T *memory::map_data(int index = 0) const;

    /// Unmaps a memory object and writes back any changes made to the
    /// previously mapped memory buffer. The memory buffer corresponds to
    /// the given index.
    ///
    /// @note
    ///     The map_data and unmap_data functions are provided mainly for
    ///     debug and testing purposes and their performance may be
    ///     suboptimal.
    ///
    /// @param mapped_ptr A pointer previously returned by
    ///     #dnnl::memory::map_data().
    /// @param index Index of the buffer. Defaults to 0.
    void memory::unmap_data(void *mapped_ptr, int index = 0) const;
```
Usage example:
```cpp
void *indices = csr_mem.map_data(1);
csr_mem.unmap_data(indices, 1);
```

The API to get/set data handle.
```cpp
/// Sets an underlying memory buffer that corresponds to the given index.
///
/// @param handle Memory buffer to use. On the CPU engine or when USM is
///     used, the memory buffer is a pointer to the actual data. For OpenCL
///     it is a cl_mem. It must have at least
///     #dnnl::memory::desc::get_size() bytes allocated.
/// @param index Memory index to attach the buffer. Defaults to 0.
void memory::set_data_handle(void *handle, int index = 0) const;

/// Returns an underlying memory buffer that corresponds to the given index.
///
/// On the CPU engine, or when using USM, this is a pointer to the
/// allocated memory.
void *memory::get_data_handle(int index = 0) const;
```
Usage example:
```cpp
void *values = csr_mem.get_data_handle(0);
csr_mem.set_data_handle(values, 0);
```
### Scope of Supported Functionality

#### Primitives

A matrix multiplication implementation that supports input tensors encoded with CSR encoding
will be introduced. The implementation will have the following coverage and limitations:
* Only f32 data type
* Only s32 data type for indices and pointers
* Destination tensor is always dense
* Only one of the two input tensors can be sparse
* Optimized for AVX2 and AVX-512 instruction sets (only src tensor can be sparse)
* No bias
* No post-ops

There are no changes to the primitives(-descriptor) API. The implementation will be
selected based on the memory descriptors that were used for creating a matmul primitive
descriptor.

There are no reorders to convert a dense tensor to CSR tensor. The memory is expected to be
created for the user provided buffers with the values and metadata.

#### Common Limitations

* The API and functionality proposed in this RFC are not supported for SYCL and OpenCL
runtimes
* The interoperability API for sparse memory is not provided
* Sparse memory and memory descriptor can only be used with the Matrix Multiplication
primitive
* Sparse memory can be created only for a CPU engine