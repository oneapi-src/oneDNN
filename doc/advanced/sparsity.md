Sparsity {#dev_guide_sparsity}
===============================================

# API

oneDNN support format kind dnnl::memory::format_kind::sparse to describe sparse tensors.
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

# CSR Encoding:
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

# Sorted COO Encoding:
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
