# Introducing Packed Sparse Weights

## Motivation

During inference, more often than not, the weights tensor has to be loaded directly
from the memory because it gets evicted from the cache due to the amount of data
used throughout the workload. This may significantly impact performance of the
workloads.
In order to optimize loading of the weights, they can be pruned, compressed and
stored in a certain format to decrease the amount of loaded data and therefore
improve performance of the workloads.

## Proposal

The proposal is to implement the optimization in oneDNN as part of the existing
experimental sparse domain.

### Sparse Encoding

While pruning the weights is the workload's responsibility compressing them is the
responsibility of oneDNN.

The idea is to introduce a new sparse encoding that would instruct the implementation
to pick the storage schema. The workflow in this case would be similar to that for the
special tag `any`.
The name of the encoding will be `packed` because the word `compressed` can be
confusing as there are a lot of sparse encodings that are considered compressed
(Compressed Sparse Row/Column, Compressed Sparse Fibers). As with the tag `any`,
a memory descriptor that was created for the packed encoding cannot be used to
create a memory object. It can only be used to create a primitive descriptor to
query the actual memory descriptor.

The storage schema will be opaque for the users.

#### API

As per the original proposal on sparse memory each sparse encoding will get a
dedicated API for creating a memory descriptor.

C API:
```c
/// Creates a memory descriptor for packed sparse encoding.
///
/// The created memory descriptor cannot be used to create a memory
/// object. It can only be used to create a primitive descriptor to
/// query the actual memory descriptor (similar to the format tag
/// `any`).
///
/// @warning
///     The meaning and content of the handles of the memory object that
///     is created using the queried memory descriptor are unspecified
///     therefore using the content is an undefined behavior.
///
/// @param memory_desc Output memory descriptor.
/// @param ndims Number of dimensions
/// @param dims Array of dimensions.
/// @param data_type Elements data type.
/// @param nnz Number of non-zero entries.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_memory_desc_create_with_packed_encoding(
        dnnl_memory_desc_t *memory_desc, int ndims, const dnnl_dims_t dims,
        dnnl_data_type_t data_type, dnnl_dim_t nnz);
```

C++ API:
```cpp
/// Function for creating a memory descriptor for packed sparse
/// encoding.
///
/// The created memory descriptor cannot be used to create a memory
/// object. It can only be used to create a primitive descriptor to
/// query the actual memory descriptor (similar to the format tag
/// `any`).
///
/// @warning
///     The meaning and content of the handles of the memory object that
///     is created using the queried memory descriptor are unspecified
///     therefore using the content is an undefined behavior.
///
/// @param adims Tensor dimensions.
/// @param adata_type Data precision/type.
/// @param nnz Number of non-zero entries.
/// @param allow_empty A flag signifying whether construction is
///     allowed to fail without throwing an exception. In this case a
///     zero memory descriptor will be constructed. This flag is
///     optional and defaults to false.
static desc memory::desc::packed(const dims &adims, data_type adata_type, dim nnz,
        bool allow_empty = false);
```

#### Implementation Details

This paragraph is intended for the developers as it describes the storage schema
for the new `packed` encoding.

Requirement #1:
The best part of the optimization is that the loaded compressed weights can be
decompressed and used with the existing optimized kernels therefore no new special
computational kernels are required. To achieve that, the storage schema should be
compatible with the blocked format (e.g. `BA16a64b4a`).

Requirement #2:
To get the best performance the decompression has to be as fast as possible. The
AVX-512 instruction set includes `VPEXPAND*` instruction that can be used to
decompress sparse data. To achieve that, the storage schema should be compatible
with the instruction.

To fulfil the requirements the storage schema will be defined as follows:
* values - buffer with non-zero values
* offsets - buffer that stores offsets for each block. The block is a product of all
inner blocks, e.g. if the target format tag is BA16a64b4a then the size of the block
is 4096 elements and the number of blocks is `padded_dims[0] * padded_dims[1] / 4096`.
When the zero elements get removed we need to store the offset to the beginning of the
block in the values buffer. For example, if the block size is 5 and there are two
blocks: [01020][01203] then the buffer with offsets will have the following two
values: [0,2] because the non-zero values are stored as [12][123]. The offsets are
stored as int64 values. These offsets are needed to find the values for a particular
block that needs to be decompressed. This will fulfil the requirement #1.
* bitmask - buffer that stores a bit-per-value mask for all elements of the original
dense tensor. The bitmask is used by `VPEXPAND*` instruction. This will fulfil the
requirement #2.

The memory descriptor created for the packed encoding will describe 3 buffers
with the following indices:
* values: 0
* offsets: 1
* bitmask: 2

In order to keep information about the original dense tensor format the `sparse_desc_t`
will re-use `blocking_desc_t`. This way `sparse_desc_t` and `blocking_desc_t` can
be used interchangeably once the utility functionality is adjusted
(e.g. `memory_desc_wrapper`).

### Reorder and Matmul

The optimization has been proven useful for certain workloads (e.g. BERT) on Intel
Xeon Scalable processors with Intel Advanced Matrix Extensions (Intel AMX) therefore
the main focus for the optimization is the matrix multiplication primitive and the
reorder primitive to compress the weights.

Coverage and limitations for the matrix multiplication primitive:
* Only weights tensor is allowed to be sparse. The other tensors are always dense
* Only matmul implementations optimized for Intel Advanced Matrix Extensions
(Intel AMX) are supported
* Only `s8` data type for the weights is supported
* Only 1 batch dimension is supported
* In general, it is expected that all matmul related functionality (e.g. post-ops,
scales, zero-points, etc) that is supported for the dense weights should also work
for the sparse weights

Coverage and limitations for the reorder primitive:
* The destination tensor cannot be in a plain format
* The implementation uses a nested reorder:
    * The primitive is used to reorder the original dense tensor to a blocked format
    first
    * Then the reordered dense tensor gets compressed (values, offsets and bitmask
    buffers are getting filled)
* In general, it is expected that all reorder-related functionality (e.g. scales,
zero-points, etc)
that is supported for the dense destination tensor should also work for the sparse one


#### Implementation Details

This paragraph is intended for the developers as it describes how the decompression
is implemented.

The decompression takes place in the matrix multiplication primitive driver during
execution of the primitive.

Decompression process:
* Identify the block number that needs to be decompressed (the one that will be used
for the current iteration)
* Use the block number to find an offset in the values buffer
* Use the bitmask to decompress the compressed data with the `VPEXPAND*` instruction

#### Common Limitations
* The functionality is experimental
* Sparse memory and memory descriptor can only be used with the Matrix Multiplication
and Reorder primitives
* Sparse memory can be created only for a CPU engine