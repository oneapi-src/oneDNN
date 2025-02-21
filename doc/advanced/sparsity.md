Sparsity {#dev_guide_sparsity}
===============================================

# API

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

oneDNN also introduces a new format kind dnnl::memory::format_kind::sparse.
Sparse encoding (a.k.a. sparse format) is an enumeration type that specifies
how data is encoded. Currently, oneDNN supports Compressed Sparse Row (CSR),
Sorted Co-ordinate (COO) Sparse Format, and PACKED sparse encodings
(dnnl::memory::sparse_encoding::csr, dnnl::memory::sparse_encoding::coo,
dnnl::memory::sparse_encoding::packed) for CPU engine, and, only sorted
COO (Co-ordinate Sparse Format) for GPU engine.

The memory descriptor has dedicated static member functions for creating memory
descriptors for different sparse encodings.

Each encoding defines the number and meaning of the buffers.

| Sparse encoding | Buffers                                                                    |
|:----------------|:---------------------------------------------------------------------------|
| CSR             | 0 - values, 1 - indices, 2 - pointers                                      |
| Sorted COO      | 0 - values, 1 to *ndims* - indices (*ndims* - number of tensor dimensions) |
| PACKED          | The meaning and content are unspecified                                    |

The pseudocode below demonstrates how to create a memory object
for the CSR and COO sparse encodings and use the new API to work with the
underlying handles.

## CSR Encoding:
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

## Sorted COO Encoding:
~~~cpp
    using namespace dnnl;
    const memory::dim M = 4, N = 6;
    const memory::dim nnz = 5;
    const auto values_dt = memory::data_type::f32;
    const auto indices_dt = memory::data_type::s32;

    // Create a memory descriptor for COO sparse encoding.
    const auto coo_md = memory::desc::coo(
            {M, N}, // Dimensions
            values_dt, // Data type of values
            nnz, // Number of non-zero entries
            indices_dt); // Data type of indices (metadata)

    // A sparse matrix represented in the COO format.
    std::vector<float> coo_values = {2.5f, 1.5f, 1.5f, 2.5f, 2.0f};
    std::vector<int32_t> coo_row_indices = {0, 1, 2, 2, 3};
    std::vector<int32_t> coo_col_indices = {0, 2, 0, 5, 1};
 
    // Create a memory object for the given buffers with values and metadata.
    memory coo_mem(coo_md, engine, {
        coo_values.data(), // Buffer with values
        coo_row_indices.data(), // Buffer with row indices (metadata)
        coo_col_indices.data() // Buffer with column indices (metadata)
        });

    const auto values_sz = coo_mem.get_size(0);
    const auto indices_sz = coo_mem.get_size(1);

    assert(values_sz == coo_values.size() * sizeof(float));
    assert(indices_sz == coo_row_indices.size() * sizeof(int32_t));
    assert(indices_sz == coo_col_indices.size() * sizeof(int32_t));

    void *values_handle = coo_mem.get_data_handle(0);
    void *row_indices_handle = coo_mem.get_data_handle(1);
    void *col_indices_handle = coo_mem.get_data_handle(2);

    assert(values_handle == (void *)coo_values.data());
    assert(row_indices_handle == (void *)coo_row_indices.data());
    assert(col_indices_handle == (void *)coo_col_indices.data());
~~~

A memory descriptor created for the sparse encoding PACKED cannot
be used to create a memory object. It can only be used to create
a primitive descriptor to query the actual memory descriptor
(similar to the format tag `any`).

 Primitives

# Matrix Multiplication

This option enables the matmul primitive that can work with
sparse input tensors.

## CSR encoding
Supported only for the CPU engine. Only one of the input tensors can be sparse.
The output tensor is always dense.

The following data type combinations are supported:

| Values (src, weight, dst)   | Indices  |
|:----------------------------|:---------|
| f16, f16, f16               | s32      |
| f32, f32, f32               | s32      |

The following format tags are supported for dense input/output
tensors:

* ab

See the example [here](@ref cpu_matmul_csr_cpp).

Benchdnn can be used to test matmul with a CSR input tensor as follows:
`./benchdnn --matmul --encoding=csr+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128`

For the case above, the number of non-zero elements for the source tensor is
calculated as max(4 * 1000000 * (1 - 0.99), 1).

## COO encoding
Supported only for the CPU and GPU engines. Only one of the input tensors can
be sparse. The output tensor is always dense.

The following data type combinations are supported:

| Values (src, weight, dst)   | Indices  |
|:----------------------------|:---------|
| f16, f16, f16               | s32      |
| f32, f32, f32               | s32      |

The following format tags are supported for dense weights tensor:

* ab
* ba

The following format tags are supported for dense destination tensor:

* ab

See the example [here](@ref cpu_matmul_coo_cpp).

Benchdnn can be used to test matmul with a COO input tensor as follows:
`./benchdnn --matmul --encoding=coo+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128`

For the case above, the number of non-zero elements for the source tensor is
calculated as max(4 * 1000000 * (1 - 0.99), 1).

## PACKED encoding

Only the weights tensor is allowed to be sparse. The other tensors
are always dense.

In general, it is expected that all matmul related functionality (e.g. post-ops,
scales, zero-points, etc) that is supported for the dense weights should
also work for the sparse weights.

Currently, matmul has the following limitations for the PACKED encoding:
* Supported only for the CPU engine
* Only Intel Advanced Matrix Extensions (Intel AMX) instruction set
architecture (ISA) is supported
* Only `s8` data type for the weights is supported
* Only 1 batch dimension is supported

See the example [here](@ref cpu_matmul_weights_compression_cpp).

Benchdnn can be used to test matmul with the PACKED weights tensor as follows:
`./benchdnn --matmul --dt=s8:s8:s32 --encoding=:packed+0.99: 3x512x1024:1x1024x512`

For the case above, the number of non-zero elements for the weights tensor is
calculated as max(1024 * 512 * (1 - 0.99), 1).

# Reorder

Currently, there is only one reorder for packing a dense tensor, i.e. converting
a dense tensor that is in `ab` format to a sparse tensor that is encoded with
the `PACKED` encoding.

In general, it is expected that all reorder-related functionality
(e.g. scales, zero-points, etc) that is supported for the dense
destination tensor should also work for the sparse one.

 Common Limitations
* The interoperability API to get/set data handles is not supported. Use the
runtime agnostic API to do that.
* Sparse memory and memory descriptor can only be used with the Matrix
Multiplication and Reorder primitives.
