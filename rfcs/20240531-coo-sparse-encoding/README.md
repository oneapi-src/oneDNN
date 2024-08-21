# Introducing Co-ordinate Sparse Encodings (COO)

## Objective
The memory API in oneDNN is currently capable of handling sparse data operations but its sparsity support is limited to compressed sparse row (CSR) and packed sparse encodings only. The objective of this proposal is to extend the API to also include support for co-ordinate sparse encodings (COO).

The extension is introduced in two steps:
- defining updates to the memory descriptor API for COO encodings.
- introducing a reference matmul implementation that processes COO-encoded sparse tensors.

## Motivation
Many deep learning (DL) workloads now employ sparse tensors for their operations due to the performance boost achieved from lower memory usage and reduced computational load. 
But the storage schemas using which the data is sparsified can differ from application to application. 
Mainstream DL frameworks incorporate sparsity support by selecting from various encoding formats such as CSR (Compressed Sparse Row), COO (Co-ordinate Sparse Encoding), CSF (Compressed Sparse Fibers) and  BSC (Blocked Compressed Sparse Column). 
The memory API in oneDNN is designed to be scalable to support all the needed encodings but the currently supported ones are CSR and Packed Sparse Encodings. 
The aim is to also extend support to additional sparse data formats beginning with the COO format which is commonly used in mainstream frameworks.

## COO and CSR Encodings
Both COO and CSR are hardware-agnostic sparse encodings meaning that the storage schema can be implemented irrespective of the hardware on which it is run. Implementation-wise, the COO format is a simpler, more generalized version of CSR.
The CSR format sparsifies data using three buffers:

|   | CSR Buffers    | Contents |
|:--|:---------------|:---------|
|0  | `data`	     | Non-zero matrix values sorted in row-major order |
|1  | `indices`      | Column indices corresponding to each non-zero value | 
|2  | `pointers`     | Compressed row indices - each $n^{th}$ value denotes the number of non-zero elements between $n^{th}$ and $(n+1)^{th}$ row |
|||

Because of the compressed row, this format tends to be more efficient for large sparse matrices.
COO, on the other hand, is simpler in implementation in that the encoding comprises of a list of thruples `(values, row_index, column_index)` corresponding to the non-zero values.
This makes the COO less efficient than CSR but it has the advantage of a reduced conversion overhead and better interpretability. 
For practical cases, a sorted variant of COO is used wherein the data is encoded as a set of sorted arrays containing the values, row indices and column indices respectively:

|   | Sorted COO Buffers    | Contents |
|:--|:----------------------|:---------|
|0  | `data`	            | Non-zero values in the matrix sorted in row-major order |
|1  | `row_indices`        | Row (Dimension 0) indices for the non-zero elements | 
|2  | `col_indices`        | Column (Dimension 1) indices for the non-zero elements |
|||

## Proposal
The proposal is to extend the oneDNN sparse memory descriptor to include support for the COO encoding. Since the sparse memory API is an experimental feature in oneDNN, the extension will also be enabled by specifying `ONEDNN_EXPERIMENTAL_SPARSE=ON`.

The addition of a new encoding will update the list for the sparse format kind: 

```cpp
/// Sparse encodings
typedef enum {
  /// Undefined sparse encoding kind, used for empty memory descriptors.
  dnnl_sparse_encoding_undef = 0,
  /// Compressed Sparse Row (CSR) encoding.
  dnnl_csr,
  /// An encoding that is used for an opaque storage schema for
  /// tensors with unstructured sparsity. A memory descriptor with the
  /// packed encoding cannot be used to create a memory object. It can
  /// only be used to create a primitive descriptor to query the
  /// actual memory descriptor (similar to the format tag `any`).
  dnnl_packed,
  /// Coordinate Sparse Encoding (COO)
  dnnl_coo,
} dnnl_sparse_encoding_t;
```

The memory object implementation in oneDNN supports underlying memory buffers that allows it to handle non-trivial data representations like sparse encodings.
Based on this implementation, COO encoding will have a separate API for creating a memory descriptor which can specify its buffers. This can be done in three ways:

### Option 1 - Sorted COO (Recommended)
The memory descriptors will comprise of $n+1$ buffers for an $n$-dimensional tensor with the first buffer reserved for data (index [0]) and the remaining buffers to hold the indices for each of the n dimensions for the tensor. That is, the indices for dimension 0 will be stored in buffer (index [1]), the indices for dimension 1 will be stored in buffer (index [2]) and the indices for dimension $n$ will be stored in buffer (index [$n+1$]). The buffers for the data indices will share the datatype during declaration. 

#### C API: 
```c
/// Creates a memory descriptor for COO encoding.
///
/// The created memory descriptor will describe a memory object that
/// contains n+1 buffers for an n-dimensional tensor. 
/// The buffers have the following meaning and assigned numbers (index):
///  - 0: values
///  - 1: indices for dimension 0
///  - 2: indices for dimension 1 ...
///  - n: indices for dimension n-1
///
/// @param memory_desc Output memory descriptor.
/// @param ndims Number of dimensions.
/// @param dims Array of dimensions.
/// @param data_type Elements data type.
/// @param nnz Number of non-zero entries.
/// @param indices_dt Data type of indices.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_memory_desc_create_with_coo_encoding(
    dnnl_memory_desc_t *memory_desc, int ndims, const dnnl_dims_t dims,
    dnnl_data_type_t data_type, dnnl_dim_t nnz, dnnl_data_type_t indices_dt);
```

#### Usage:
```c
const int ndims = 2;
const int nnz = 12;
dnnl_memory_desc_t coo_md;
dnnl_status_t status = dnnl_memory_desc_create_with_coo_encoding(
    &coo_md, ndims, {128, 64}, f32, nnz, s32);
```

#### C++ API:
```cpp
struct memory {
  struct desc {
    /// Function for creating a memory descriptor for COO sparse encodings.
    ///
    /// The created memory descriptor will describe a memory object that
    /// contains n+1 buffers for an n-dimensional tensor. 
    /// The buffers have the following meaning and assigned numbers (index):
    ///  - 0: values
    ///  - 1: indices for dimension 0
    ///  - 2: indices for dimension 1 ...
    ///  - n: indices for dimension n-1
    ///
    /// @param adims Tensor dimensions.
    /// @param adata_type Data precision/type.
    /// @param nnz Number of non-zero entries.
    /// @param index_dt Data type of indices.
    /// @param allow_empty A flag signifying whether construction is
    ///     allowed to fail without throwing an exception. In this case a
    ///     zero memory descriptor will be constructed. This flag is
    ///     optional and defaults to false.
    static desc coo(const dims &adims, data_type adata_type, dim nnz,
                    data_type index_dt, bool allow_empty = false);
  }
}
```

#### Usage:
```cpp
const int nnz = 12;
auto coo_md = memory::desc::coo({128, 64}, f32, nnz, s32);
```

The following example shows how size and data type can be queried for the COO memory descriptor for the different buffers. Here, the [`dnnl_memory_desc_get_size_v2()`](https://oneapi-src.github.io/oneDNN/group_dnnl_api_memory.html#doxid-group-dnnl-api-memory-1gad8ada49d1107442436109ec1de73f370) function works for sorted COO in the same manner as for CSR with the exception of the data buffers. For CSR, this buffer contains the pointers to the compressed row indices of the sparse data whereas for the COO format, all data indices are uncompressed. Hence, querying the buffer size for any index>0 returns the same value. The same is true for querying the data type with `dnnl_memory_desc_query_v2()` as the data type is shared by all the index buffers.

```cpp
const size_t values_size = dnnl_memory_desc_get_size_v2(coo_md, 0),;
const size_t indices_size = dnnl_memory_desc_get_size_v2(coo_md, 1);
char *values = malloc(values_size);
char *row_indices = malloc(indices_size);
char *col_indices = malloc(indices_size);
```

### Memory Creation Example:

#### C API:
```c
const int nhandles = ndims + 1;
void *handles[nhandles] = {};

// Initialize handles to create a sparse memory object.
handles[0] = (void *)values;
handles[1] = (void *)row_indices;
handles[2] = (void *)col_indices;

// Create a sparse memory object for the given buffers.
dnnl_memory_create_v2(&coo_mem, coo_md, engine, nhandles, handles);
```

#### C++ API:
```cpp
// Memory creation with user-specified buffers
auto coo_mem = memory(coo_md, engine, {values, row_indices, col_indices});
```

#### Advantages of using Sorted COO:
- Storing the indices in sorted arrays allows faster data traversal as well as a simpler construction.
- The sorted order can also improve cache locality during traversal as sequential access patterns tend to be more cache-friendly.
- While COO-to-CSR conversion is not aimed to be a part of the proposed functionality, the separate buffers for the row and column indices will allow for an efficient conversion between the two formats if such a feature is required in the future.

### Option 2 - Sorted COO with a single Index buffer
For this option, the memory descriptor will comprise of only two buffers: one for the data (index [0]) and one for the indices (index [1]). The dimensions of the index buffer will be determined from the tensor rank and set to (`nnz` - number of non-zero values, `r` - rank of the tensor). The API declaration and usage will be similar to the first option. Since this method encodes the data in sorted arrays, it shares the advantages of the first option. Also, the number of buffers to be defined will not be depedent on the tensor dimensions. However, the single buffer used for storing data indices will increase conversion overhead.

### Option 3 -  COO with a thruple set 
For this option, the memory descriptor will comprise of a single buffer containing a set of thruples wherein each thruple holds data and indices for a non-zero element. The thruple will be required to handle the two different datatypes pertaining to the data and indices respectively. Using a list for storing data means that it will allow for faster dynamic changes to the memory but the traversal and conversion overheads will be higher.

## Scope of Extended Functionality

For the extended API, a reference implementation for the matrix multiplication primitive will be introduced that supports input tensors with COO encoding. The coverage of the implementation will be similar to the CSR encoding and is listed below:
* Datatype for COO tensor data: `f16`, `f32`
* Datatype for COO tensor indices: `s32`
* Destination tensor for the matmul operation is always dense
* Either the multiplier or multiplicand tensor can be sparse for the matmul operation but not both
* No bias or post-ops 

The API for the primitive and the primitive descriptor will remain unchanged. The implementation will be selected for the matmul primitive based on the memory descriptors provided during primitive creation.

(EOD)