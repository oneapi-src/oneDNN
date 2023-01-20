Experimental features {#dev_guide_experimental}
===============================================

To test aggressive performance optimizations that might affect accuracy or new
API and functionality without an impact to regular users, oneDNN provides
experimental features.

## Build-time Controls

There are two kinds of experimental features:
1. Features that can be enabled at runtime with an environment variable.
To enable such experimental features, the library should be built with a CMake
option `ONEDNN_EXPERIMENTAL=ON`. Each experimental feature has to be
individually selected using environment variables.
2. Features that can be enabled only with a build time option. To enable such
experimental features, the library should be built with a CMake option that
corresponds to a particular feature.

Both kinds of experimental features can be enabled simultaneously.

## Experimental features

| Environment variable                     | Description
| :---                                     | :---
| ONEDNN_EXPERIMENTAL_BNORM_STATS_ONE_PASS | Calculate mean and variance in batch normalization(BN) in single pass ([RFC](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20210519-single-pass-bnorm)).

| Build time option                        | Description
| :---                                     | :---
| ONEDNN_EXPERIMENTAL_SPARSE               | Enable experimental API and functionality for sparse domain.

## Features details

### ONEDNN_EXPERIMENTAL_SPARSE
This option extends the existing API and adds a new one to support sparse
functionality in oneDNN.

The main change is in oneDNN memory object semantics. Now, the memory object can
have multiple underlying buffers. In the case of regular dense computations, the
memory object always contains a single buffer. But in the case of sparse
computations, the memory object always contains one buffer for values and an
arbitrary number of additional buffers for metadata.

The underlying buffers are enumerated starting with 0, meaning that each buffer
has its own number. The buffer with values always has index 0.

In most cases, the API that works with underlying buffers takes a buffer index. The
exception is the API for creating a memory object. In that case, the API takes a vector
of buffers. The order of the buffers in the vector matters and should correspond to
the buffers' indices.

oneDNN also introduces a new format kind #dnnl::memory::format_kind::sparse, 
sparse encoding. Sparse encoding (a.k.a. sparse format) is an
enumeration type that specifies how data is encoded. Currently, oneDNN only
supports CSR (Compressed sparse row) sparse encoding
(#dnnl::memory::sparse_encoding::csr).

The memory descriptor has dedicated static member functions for creating memory
descriptors for different sparse encodings.

Each encoding defines the number and meaning of the buffers.

| Sparse encoding | Buffers
| :---            | :---
| CSR             | 0 - values, 1 - indices, 2 - pointers

Pseudo-code with creating a memory object for CSR sparse encoding.

~~~cpp
    using namespace dnnl;
    const memory::dim M = 4, N = 6;
    const memory::dim nnz = 5;
    const auto values_dt = memory::data_type::f32;
    const auto indices_dt = memory::data_type::s32;
    const auto pointers_dt = memory::data_type::s32;

    // Create a memory descriptor for CSR sparse encoding.
    const auto csr_md = memory::desc::csr(
            {M, N}, // Dimensions
            values_dt, // Data type of values
            nnz, // Number of non-zero entries
            indices_dt, // Data type of indices (metadata)
            pointers_dt); // Data type of pointers (metadata)

    // A sparse matrix represented in the CSR format.
    std::vector<float> csr_values = {2.5f, 1.5f, 1.5f, 2.5f, 2.0f};
    std::vector<int32_t> csr_indices = {0, 2, 0, 5, 1};
    std::vector<int32_t> csr_pointers = {0, 1, 2, 4, 5, 5};

    // Create a memory object for the given buffers with values and metadata.
    memory csr_mem(csr_md, engine, {
        csr_values.data(), // Buffer with values
        csr_indices.data(), // Buffer with indices (metadata)
        csr_pointers.data() // Buffer with pointers (metadata)
        });

    const auto values_sz = csr_mem.get_size(0);
    const auto indices_sz = csr_mem.get_size(1);
    const auto pointers_sz = csr_mem.get_size(2);

    assert(values_sz == csr_values.size() * sizeof(float));
    assert(indices_sz == csr_indices.size() * sizeof(int32_t));
    assert(pointers_sz == csr_pointers.size() * sizeof(int32_t));

    void *values_handle = csr_mem.get_data_handle(0);
    void *indices_handle = csr_mem.get_data_handle(1);
    void *pointers_handle = csr_mem.get_data_handle(2);

    assert(values_handle == (void *)csr_values.data());
    assert(indices_handle == (void *)csr_indices.data());
    assert(pointers_handle == (void *)csr_pointers.data());
~~~

@warning
- Enabling experimental features does not guarantee that the library will utilize them
- Enabling experimental features might change the accuracy of oneDNN primitives
