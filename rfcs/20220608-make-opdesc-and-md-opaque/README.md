# Make Memory and Operation Descriptors Opaque

## Motivation
* Reduce maintenance cost and complexity of scaling the API
    * There is no need to add `v2` operation descriptors and whole `v2` primitives
    in C++ API
    * There is no need to maintain both versions of operation descriptors inside
    the library
    * Adding new API would only require adding a single `v2` function with minimum
    changes internally
* Separate API and implementation details
    * There is no need to expose `rnn_packed_desc` or `wino_desc` as these are
    opaque
    * Remove the possibility of oneDNN API misuse by exposing a set of the API
    that is supposed to be used by users
    * There will be no way to bypass parameters check in the API
* Allow to store operation and memory descriptors internally in a way that allows
to reduce hashing/comparison overhead
* Reference semantics allow users to get rid of data copy overhead for operation
and memory descriptors

## Proposal
Make memory and operation descriptors opaque. Similar to other abstractions,
e.g. `dnnl_primitive_t`.
* The library will be responsible for creating and destroying them
* The library will provide API to query necessary information from them

## Memory Descriptor
Making memory descriptor opaque means making `dnnl_memory_desc_t` structure opaque
and not the layout it describes.

The existing `dnnl_memory_desc_t` will be redefined as follows:
```cpp
// dnnl_types.h
struct dnnl_memory_desc;
/// A memory descriptor handle.
typedef struct dnnl_memory_desc *dnnl_memory_desc_t;
```
A new alias for constant memory descriptors will be defined as follows:
```cpp
// dnnl_types.h
struct dnnl_memory_desc;
/// A memory descriptor handle.
typedef const struct dnnl_memory_desc *const_dnnl_memory_desc_t;
```
The API that takes/returns a memory descriptor will be adjusted to reflect
the new definition of `dnnl_memory_desc_t` and `const_dnnl_memory_desc_t`.
* `dnnl_memory_desc_t **` will be changed to `dnnl_memory_desc_t *`
* `dnnl_memory_desc_t *` will be changed to `dnnl_memory_desc_t`
* `const dnnl_memory_desc_t *` will be changed to `const_dnnl_memory_desc_t`
* `const dnnl_memory_desc_t **` will be changed to `const_dnnl_memory_desc_t *`

### Creation/Destruction of Memory Descriptor
Since the library will be managing creation and destruction of the memory
descriptor a new API to do that will be added.

#### C API
The existing API for initializing memory descriptor should be renamed:
* `dnnl_memory_desc_init_by_strides` -> `dnnl_memory_desc_create_with_strides`
* `dnnl_memory_desc_init_by_tag` -> `dnnl_memory_desc_create_with_tag`
* `dnnl_memory_desc_init_submemory` -> `dnnl_memory_desc_create_submemory`

A new API for destroying the memory descriptor will be introduced:
```cpp
dnnl_status_t dnnl_memory_desc_destroy(dnnl_memory_desc_t memory_desc);
```
#### C++ API
Memory descriptor will have reference semantics, similar to other abstractions.
This will require `struct memory::desc {...}` to be inherited from
`handle<dnnl_memory_desc_t>`, which will require adding a new `handle_traits`
specialization for `dnnl_memory_desc_t`.

```cpp
template <>
struct handle_traits<dnnl_memory_desc_t> {
    static dnnl_status_t destructor(dnnl_memory_desc_t p) {
        return dnnl_memory_desc_destroy(p);
    }
};
```
Adding new constructors of `struct memory::desc {...}` for creating it for the
given information about blocks doesn't seem to have applicability because oneDNN
provide a set of named tags to do that. Custom blocking will cause fallback to
the reference implementation, which are slow.

##### Default Constructor
The default constructor will create a zero memory descriptor. Zero memory descriptor
is a memory descriptor that has number of dimensions equal to 0.

##### Construction from a C Handle
Interface of the the existing constructor that takes a `dnnl_memory_desc_t` structure
will be changed from `desc(const dnnl_memory_desc_t &data)` to
`desc(dnnl_memory_desc_t md)`.

The `md` parameter is a C handle. The constructed `memory::desc` will take ownership
of the handle therefore it will be destroyed during the destruction of `memory::desc`
object.

Interfaces of the rest of the constructors will stay intact.

### Querying Memory Descriptor Information
oneDNN needs to provide an API to query information from opaque memory descriptors.

Basic information:
* Number dimensions
* Dimensions
* Data type
* Submemory offset
* Padded dimensions
* Padded offsets

In addition to the basic information memory descriptor holds information about how
data is physically arranged. Currently oneDNN can described data according to the
following formats (`format_desc`):
* `blocked`
* `wino`
* `rnn_packed`

There is a couple of options for providing additional information about memory
descriptor.

#### Option 1: More Implementation Details in API
The following information can be queried:
* Format kind: any, blocked, wino, rnn_packed or undef
* Strides: vector of strides or an empty one if format kind is not blocked
* Number of blocks: a number of blocks or 0 if format kind is not blocked
* Blocks: vector of block sizes or an empty one if format kind is not blocked
* Block indices: vector of indices or an empty one if format kind is not blocked

Pros:
* Provides sufficient information about blocked descriptor for advanced users

Cons:
* Exposing opaque format kinds that are supposed to be created only by primitives
(e.g. `wino`, `rnn_packed`) doesn't seem to have applicability
* Blocked formats (with block sizes > 1 - non-plain) are positioned as opaque
therefore providing detailed information seems to contradict its definition,
though we documented how blocked format is implemented

#### Option 2: Less Implementation Details in API

The following information can be queried:
* Format kind: any, blocked (plain), opaque or undef
    * opaque includes `wino`, `rnn_packed`, non-plain blocked
* Strides: vector of strides or an empty one if format kind is not blocked or
memory descriptor was not created with the given strides or with the given plain
format tag.

Pros:
* Hides unnecessary format kinds. Exposing opaque format kinds that are supposed to be created only by primitives (e.g. `wino`, `rnn_packed`) doesn't seem to have
applicability
* Hides implementation details of non-plain blocked formats. Non-plain blocked
formats are positioned as opaque therefore providing detailed information seems to
contradict its definition, though we documented how blocked format is implemented

Cons:
* May affect advanced users who use information about blocked formats
* May affect users that have some logic depending on format kinds

#### Option 2a: Less Implementation Details in API
The same as Option 2 but allow to query number of blocks, blocks themselves and
block indices.

Pros:
* Provides sufficient information about blocked descriptor for advanced users
* Hides unnecessary format kinds. Exposing opaque format kinds that are supposed to
be created only by primitives (e.g. `wino`, `rnn_packed`) doesn't seem to have
applicability

Cons:
* May affect users that have some logic depending on format kinds

Proposal: go with the option 2a because exposing opaque format kinds `wino` and
`rnn_packed` doesn't seem to have applicability but information about blocked
formats can be used by advanced users.

#### C API
The following API will be added to query the information. The API is similar to
the one for primitive descriptor.
```cpp
dnnl_status_t dnnl_memory_desc_query(const_dnnl_memory_desc_t memory_desc,
        dnnl_query_t what, void *result);
```
The list of memory descriptor queries:
* `dnnl_query_ndims`
* `dnnl_query_dims`
* `dnnl_query_data_type`
* `dnnl_query_submemory_offset`
* `dnnl_query_padded_dims`
* `dnnl_query_padded_offsets`
* `dnnl_query_format_kind`
* `dnnl_query_strides`
* `dnnl_query_inner_nblks`
* `dnnl_query_inner_blks`
* `dnnl_query_inner_idxs`

#### C++ API
The `memory::desc` will get the following interfaces to query the information:
```cpp
// Common API.
int memory::desc::get_ndims() const;
memory::dims memory::desc::get_dims() const;
memory::data_type memory::desc::get_data_type() const;
memory::dim memory::desc::submemory_offset() const;
memory::dims memory::desc::get_padded_dims() const;
memory::dims memory::desc::get_padded_offsets() const;
memory::format_kind memory::desc::get_format_kind() const;
// The following API is only applicable for blocked format.
memory::dims memory::desc::get_strides() const;
int memory::desc::get_inner_nblks() const;
memory::dims memory::desc::get_inner_blks() const;
memory::dims memory::desc::get_inner_idxs() const;
```

### Comparison Semantics
The current comparison semantics is defined according to `dnnl_memory_desc_equal`
API.
In order to preserve the semantics oneDNN needs to provide an API to compare
opaque memory descriptors. The following API will be added.

#### C API
```cpp
int dnnl_memory_desc_equal(const_dnnl_memory_desc_t lhs,
        const_dnnl_memory_desc_t rhs);
```

#### C++ API
```cpp
bool memory::desc::operator==(const memory::desc &other);
```
### Cloning
Memory descriptors can be returned as result of different queries. The queries
will return a pointer to opaque memory descriptors therefore oneDNN needs to
provide an API to clone them.

#### C API
```cpp
dnnl_status_t dnnl_memory_desc_clone(dnnl_memory_desc_t *memory_desc,
        const_dnnl_memory_desc_t existing_memory_desc);
```

#### C++ API
All C++ APIs that return memory descriptors should return a clone of it.

For example:
```cpp
memory::desc query_md(query what, int idx = 0) const {
    // ... //
    dnnl_memory_desc_t c_md;
    error::wrap_c_api(dnnl_memory_desc_clone(&c_md, const_c_md),
            "could not clone memory descriptor");
    return c_md ? : memory::desc(c_md) : memory::desc();
}
```

### API Use Example
```cpp
using namespace dnnl;

const int block_size = 8;
const int C = 17;
const int C_padded = div_up(17, block_size) * block_size;

const int ndims = 4;
memory::dims dims = {N, C, H, W};

memory::desc(dims, memory::data_type::f32, memory::format_tag::nChw8c);

memory::dim expect_stride_n =  C_padded * H * W;
memory::dim expect_stride_C =  H * W * block_size;
memory::dim expect_stride_h =  W * block_size;
memory::dim expect_stride_w =  block_size;
memory::dim expect_stride_8c = 1;

// New API:
const bool expect_true = true
    && md.get_dims()[0] == N
    && md.get_dims()[1] == C
    && md.get_dims()[2] == H
    && md.get_dims()[3] == W
    && md.get_padded_dims()[0] == N
    && md.get_padded_dims()[1] == C_padded
    && md.get_padded_dims()[2] == H
    && md.get_padded_dims()[3] == W
    && md.get_strides()[0] == expect_stride_n
    && md.get_strides()[1] == expect_stride_C
    && md.get_strides()[2] == expect_stride_h
    && md.get_strides()[3] == expect_stride_w
    && md.get_inner_nblks() == 1
    && md.get_inner_idxs()[0] == 1
    && md.get_inner_blks()[0] == 8;

assert(expect_true);

// Old API:
// const bool expect_true = true
//     && md.data.dims[0] == N
//     && md.data.dims[1] == C
//     && md.data.dims[2] == H
//     && md.data.dims[3] == W
//     && md.data.padded_dims[0] == N
//     && md.data.padded_dims[1] == C_padded
//     && md.data.padded_dims[2] == H
//     && md.data.padded_dims[3] == W
//     && md.data.format_desc.blocking.strides[0] == expect_stride_n
//     && md.data.format_desc.blocking.strides[1] == expect_stride_C
//     && md.data.format_desc.blocking.strides[2] == expect_stride_h
//     && md.data.format_desc.blocking.strides[3] == expect_stride_w
//     && md.data.format_desc.blocking.inner_nblks == 1
//     && md.data.format_desc.blocking.inner_idxs[0] == 1
//     && md.data.format_desc.blocking.inner_blks[0] == block_size;
```

## Operation Descriptor: Option 1 - Keep Operation Descriptor
The existing `dnnl_op_desc_t` and `const_dnnl_op_desc_t` will be redefined as
follows:

```cpp
struct dnnl_op_desc;
/// An operation descriptor handle.
typedef struct dnnl_op_desc *dnnl_op_desc_t;
/// A constant operation descriptor handle.
typedef const struct dnnl_op_desc *const_dnnl_op_desc_t;
```
The API that takes/returns an operation descriptor will be adjusted to reflect
the new definition of `dnnl_op_desc_t` and `const_dnnl_op_desc_t`.
* `dnnl_<primitive kind>_desc_t **` will be changed to `dnnl_op_desc_t *`
* `dnnl_<primitive kind>_desc_t *` will be changed to `dnnl_op_desc_t`
* `const dnnl_<primitive kind>_desc_t *` will be changed to `const_dnnl_op_desc_t`
* `const dnnl_<primitive kind>_desc_t **` will be changed to `const_dnnl_op_desc_t *`

### Creation/Destruction of Operation Descriptors
Since the library will be managing creation and destruction of the operation
descriptors a new API to do that will be added.

#### C API
The existing API for initializing operation descriptors should be renamed:
`dnnl_<primitive kind>_[propagation kind]_desc_init(...)` to
`dnnl_<primitive kind>_[propagation kind]_desc_create(...)`.

A new API for destroying the operation descriptors will be introduced:
```cpp
dnnl_status_t dnnl_op_desc_destroy(dnnl_op_desc_t op_desc);
```

#### C++ API
Operation descriptors will have reference semantics, similar to other abstractions.
This will require `struct <primitive kind>::desc {...}` to be inherited from
`handle<dnnl_op_desc_t>`, which will require adding a new `handle_traits`
specialization for `dnnl_op_desc_t`.

Interfaces of constructors of `struct <primitive kind>::desc {...}` will stay
intact.

```cpp
template <>
struct handle_traits<dnnl_op_desc_t> {
    static dnnl_status_t destructor(dnnl_op_desc_t p) {
        return dnnl_op_desc_destroy(p);
    }
};
```
### Querying Operation Descriptor Information

When it comes to operation descriptors there are two kinds of possible queries:
1. Common such as `primitive_kind`, `src_desc`, `dst_desc`, etc.
2. Primitive specific such as `lrn_alpha` or `batch_norm_epsilon`, etc.

Most of the primitive specific queries can be generalized e.g. `alpha`, `alg_kind`,
`strides`, etc. In order to avoid duplication of such primitive specific queries
they will not be tied to them that is, primitive name will not be part of the query
name.

Some primitive specific queries are too specific and cannot be generalized e.g.
`local_size` for LRN, `cell_kind` for RNN, `p` for reduction, etc. Such queries
will be tied to the primitives that is primitive name will be part of the query
name.

#### C API
The following API will be added to query the information. The API is similar to the
one for primitive descriptor.
```cpp
dnnl_status_t dnnl_op_desc_query(const_dnnl_op_desc_t op_desc,
        dnnl_query_t what, int index, void *result);
```

The list of primitive specific queries that will be generalized:
* `dnnl_query_strides`, `dnnl_query_dilation`, `dnnl_query_padding_l`,
`dnnl_query_padding_r`
    * Convolution, Pooling
* `dnnl_query_epsilon`
    * Batch Normalization
    * Layer Normalization
* `dnnl_query_flags`
    * Batch Normalization
    * Layer Normalization
    * RNN
* `dnnl_query_alg_kind`
    * Eltwise
    * LRN
    * Pooling
    * Reduction
    * Softmax
* `dnnl_query_alpha`
    * Eltwise
    * LRN
    * RNN
* `dnnl_query_beta`
    * Eltwise
    * LRN
    * RNN
* `dnnl_query_axis`
    * Shuffle
    * Softmax

The list of primitive specific queries that will not be generalized:
* LRN
    * `dnnl_query_lrn_local_size`
    * `dnnl_query_lrn_k`
* Reduction
    * `dnnl_query_reduction_p`
    * `dnnl_query_reduction_eps`
* Resampling
    * `dnnl_query_resampling_factors`
* RNN
    * `dnnl_query_rnn_cell_kind`
    * `dnnl_query_rnn_direction`
    * `dnnl_query_rnn_activation_kind`
* Pooling
    * `dnnl_query_pooling_kernel`
* Shuffle
    * `dnnl_query_shuffle_group_size`

#### C++ API
A base class for all `<primitive kind>::desc` classes will be added:
`struct op_desc`.

C++ API already has a query mechanism for memory descriptors that is implemented in
`primitive_desc_base`. In order to avoid duplication of the code the common queries
between operation and primitive descriptors will be moved to a new helper class
named `query_helper`.

* `op_desc` class will be inherited from `query_helper<dnnl_op_desc_t>`
* `primitive_desc_base` class will be inherited from
`query_helper<dnnl_primitive_desc_t>`

Interfaces of the query helper class:
```cpp
template <typename T>
struct query_helper : public handle<T> {
    dnnl::primitive::kind get_kind() const;
    memory::dim query_s64(query what) const;
    memory::desc query_md(query what, int idx = 0) const;

    memory::desc src_desc(int idx) const;
    memory::desc dst_desc(int idx) const;
    memory::desc weights_desc(int idx) const;
    memory::desc diff_src_desc(int idx) const;
    memory::desc diff_dst_desc(int idx) const;
    memory::desc diff_weights_desc(int idx) const;

    // Separate versions without the index argument for documentation
    // purposes.
    memory::desc src_desc() const;
    memory::desc dst_desc() const;
    memory::desc weights_desc() const;
    memory::desc diff_src_desc() const;
    memory::desc diff_dst_desc() const;
    memory::desc diff_weights_desc() const;

    using query = query_helper;
};
```

* Query interfaces of `primitive_desc_base`, `primitive_desc`,
`rnn_primitive_desc_base` and `<primitive kind>::primitive_desc` classes will stay
unchanged
* Each `<primitive_kind>::desc` class will be extended with new interfaces to
provide query capabilities and for documentation purposes
* The generalized operation descriptor query interfaces will reside in the base
operation descriptor class `op_desc`
* Primitive specific queries will reside in the corresponding
`<primitive_kind>::desc` classes

The base class for all operation descriptors `op_desc` will have the following
interfaces:

```cpp
struct op_desc : public query_helper<dnnl_op_desc_t> {
    // Used by (de-)convolution, pool
    dims get_strides() const;
    dims get_dilation() const;
    dims get_padding_l() const;
    dims get_padding_r() const;

    // Used by layer normalization, batch normalization, reduction
    float get_epsilon() const;

    // Used by batch normalization, layer normalization, RNN
    uint32_t get_flags() const;

    // Used by eltwise, LRN, Pooling, Reduction, Softmax
    dnnl::algorithm get_algorithm() const;

    // Used by eltwise, LRN, RNN
    float get_alpha() const;

    // Used by eltwise, LRN, RNN
    float get_beta() const;

    // Used by shuffle, softmax
    int get_axis() const;

    // Used by LRN
    dim get_local_size() const;
    float get_k() const;

    // Used by reduction
    float get_p() const;

   // Used by resampling
   std::vector<float> get_factors() const;

   // Used by RNN
   dnnl::algorithm get_cell_kind() const;
   dnnl::rnn_direction get_direction() const;
   dnnl::algorithm get_activation_kind() const;

    // Used by pooling
    dims get_kernel() const;
    // Used by shuffle
    dim get_group_size() const;

    // Used by all primitives
    dnnl::prop_kind get_prop_kind() const;
};
```

All operation descriptors will get new interfaces to query information from them.
<details> <summary> Convolution and Deconvolution </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_algorithm() const;
dnnl::prop_kind desc::get_prop_kind() const;
memory::dims desc::get_strides() const;
memory::dims desc::get_dilation() const;
memory::dims desc::get_padding_l() const;
memory::dims desc::get_padding_r() const;

// Additional for forward.
memory::desc desc::src_desc() const;
memory::desc desc::weights_desc() const;
memory::desc desc::bias_desc() const;
memory::desc desc::dst_desc() const;

// Additional for backward data.
memory::desc desc::diff_src_desc() const;
memory::desc desc::weights_desc() const;
memory::desc desc::diff_dst_desc() const;

// Additional for backward weights.
memory::desc desc::src_desc() const;
memory::desc desc::diff_weights_desc() const;
memory::desc desc::diff_bias_desc() const;
memory::desc desc::diff_dst_desc() const;
```
</details>

<details> <summary> Local Response Normalization (LRN) </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_algorithm() const;
dnnl::prop_kind desc::get_prop_kind() const;
float desc::get_alpha() const;
float desc::get_beta() const;
dim desc::get_local_size() const;
float desc::get_k() const;
memory::desc desc::src_desc() const;

// Additional for forward.
memory::desc desc::dst_desc() const;

// Additional for backward.
memory::desc desc::diff_src_desc() const;
memory::desc desc::diff_dst_desc() const;
```
</details>

<details> <summary> Eltwise </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_algorithm() const;
dnnl::prop_kind desc::get_prop_kind() const;
float desc::get_alpha() const;
float desc::get_beta() const;
memory::desc desc::src_desc() const;
memory::desc desc::dst_desc() const;

// Additional for backward.
memory::desc desc::diff_src_desc() const;
memory::desc desc::diff_dst_desc() const;
```
</details>

<details> <summary> Softmax </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_algorithm() const;
dnnl::prop_kind desc::get_prop_kind() const;
int desc::get_axis() const;
memory::desc desc::dst_desc() const;

// Additional for forward.
memory::desc desc::src_desc() const;

// Additional for backward.
memory::desc desc::diff_src_desc() const;
memory::desc desc::diff_dst_desc() const;
```
</details>

<details> <summary> Batch Normalization </summary>

```cpp
// For all propagation kinds.
dnnl::prop_kind desc::get_prop_kind() const;
float desc::get_epsilon() const;
uint32_t desc::get_flags() const;
memory::desc desc::src_desc() const;
memory::desc desc::weights_desc() const;
memory::desc desc::mean_desc() const;
memory::desc desc::variance_desc() const;

// Additional for forward.
memory::desc desc::dst_desc() const;

// Additional for backward.
memory::desc desc::diff_src_desc() const;
memory::desc desc::diff_dst_desc() const;
memory::desc desc::diff_weights_desc() const;
```
</details>

<details> <summary> Layer Normalization </summary>

```cpp
// For all propagation kinds.
dnnl::prop_kind desc::get_prop_kind() const;
float desc::get_epsilon() const;
uint32_t desc::get_flags() const;
memory::desc desc::src_desc() const;
memory::desc desc::weights_desc() const;
memory::desc desc::mean_desc() const;
memory::desc desc::variance_desc() const;

// Additional for forward.
memory::desc desc::dst_desc() const;

// Additional for backward.
memory::desc desc::diff_src_desc() const;
memory::desc desc::diff_dst_desc() const;
memory::desc desc::diff_weights_desc() const;
```
</details>

<details> <summary> Inner Product </summary>

```cpp
// For all propagation kinds.
dnnl::prop_kind desc::get_prop_kind() const;

// Additional for forward data.
memory::desc desc::src_desc() const;
memory::desc desc::weights_desc() const;
memory::desc desc::bias_desc() const;
memory::desc desc::dst_desc() const;

// Additional for backward data.
memory::desc desc::diff_src_desc() const;
memory::desc desc::weights_desc() const;
memory::desc desc::diff_dst_desc() const;

// Additional for backward weights.
memory::desc desc::src_desc() const;
memory::desc desc::diff_weights_desc() const;
memory::desc desc::diff_bias_desc() const;
memory::desc desc::diff_dst_desc() const;
```
</details>

<details> <summary> RNN: Vanilla </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_cell_kind() const;
dnnl::prop_kind desc::get_prop_kind() const;
dnnl::algorithm desc::get_activation_kind() const;
dnnl::rnn_direction desc::get_direction() const;
dnnl::rnn_flags desc::get_flags() const;
float desc::get_alpha() const;
float desc::get_beta() const;
memory::desc desc::src_layer_desc() const;
memory::desc desc::src_iter_desc() const;
memory::desc desc::weights_layer_desc() const;
memory::desc desc::weights_iter_desc() const;
memory::desc desc::bias_desc() const;
memory::desc desc::dst_layer_desc() const;
memory::desc desc::dst_iter_desc() const;

// Additional for backward.
memory::desc desc::diff_src_layer_desc() const;
memory::desc desc::diff_src_iter_desc() const;
memory::desc desc::diff_weights_layer_desc() const;
memory::desc desc::diff_weights_iter_desc() const;
memory::desc desc::diff_bias_desc() const;
memory::desc desc::diff_dst_layer_desc() const;
memory::desc desc::diff_dst_iter_desc() const;
```
</details>

<details> <summary> RNN: LSTM </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_cell_kind() const;
dnnl::prop_kind desc::get_prop_kind() const;
dnnl::rnn_direction desc::get_direction() const;
dnnl::rnn_flags desc::get_flags() const;
memory::desc desc::src_layer_desc() const;
memory::desc desc::src_iter_desc() const;
memory::desc desc::src_iter_c_desc() const;
memory::desc desc::weights_layer_desc() const;
memory::desc desc::weights_iter_desc() const;
memory::desc desc::weights_peephole_desc() const;
memory::desc desc::weights_projection_desc() const;
memory::desc desc::bias_desc() const;
memory::desc desc::dst_layer_desc() const;
memory::desc desc::dst_iter_desc() const;
memory::desc desc::dst_iter_c_desc() const;

// Additional for backward.
memory::desc desc::diff_src_layer_desc() const;
memory::desc desc::diff_src_iter_desc() const;
memory::desc desc::diff_src_iter_c_desc() const;
memory::desc desc::diff_weights_layer_desc() const;
memory::desc desc::diff_weights_iter_desc() const;
memory::desc desc::diff_weights_peephole_desc() const;
memory::desc desc::diff_weights_projection_desc() const;
memory::desc desc::diff_bias_desc() const;
memory::desc desc::diff_dst_layer_desc() const;
memory::desc desc::diff_dst_iter_desc() const;
memory::desc desc::diff_dst_iter_c_desc() const;
```
</details>

<details> <summary> RNN: GRU </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_cell_kind() const;
dnnl::prop_kind desc::get_prop_kind() const;
dnnl::rnn_direction desc::get_direction() const;
dnnl::rnn_flags desc::get_flags() const;
memory::desc desc::src_layer_desc() const;
memory::desc desc::src_iter_desc() const;
memory::desc desc::weights_layer_desc() const;
memory::desc desc::weights_iter_desc() const;
memory::desc desc::bias_desc() const;
memory::desc desc::dst_layer_desc() const;
memory::desc desc::dst_iter_desc() const;

// Additional for backward.
memory::desc desc::diff_src_layer_desc() const;
memory::desc desc::diff_src_iter_desc() const;
memory::desc desc::diff_weights_layer_desc() const;
memory::desc desc::diff_weights_iter_desc() const;
memory::desc desc::diff_bias_desc() const;
memory::desc desc::diff_dst_layer_desc() const;
memory::desc desc::diff_dst_iter_desc() const;
```
</details>

<details> <summary> RNN: LBR GRU </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_cell_kind() const;
dnnl::prop_kind desc::get_prop_kind() const;
dnnl::rnn_direction desc::get_direction() const;
dnnl::rnn_flags desc::get_flags() const;
memory::desc desc::src_layer_desc() const;
memory::desc desc::src_iter_desc() const;
memory::desc desc::weights_layer_desc() const;
memory::desc desc::weights_iter_desc() const;
memory::desc desc::bias_desc() const;
memory::desc desc::dst_layer_desc() const;
memory::desc desc::dst_iter_desc() const;

// Additional for backward.
memory::desc desc::diff_src_layer_desc() const;
memory::desc desc::diff_src_iter_desc() const;
memory::desc desc::diff_weights_layer_desc() const;
memory::desc desc::diff_weights_iter_desc() const;
memory::desc desc::diff_bias_desc() const;
memory::desc desc::diff_dst_layer_desc() const;
memory::desc desc::diff_dst_iter_desc() const;
```
</details>

<details> <summary> RNN: AUGRU </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_cell_kind() const;
dnnl::prop_kind desc::get_prop_kind() const;
dnnl::rnn_direction desc::get_direction() const;
dnnl::rnn_flags desc::get_flags() const;

memory::desc desc::src_layer_desc() const;
memory::desc desc::src_iter_desc() const;
memory::desc desc::attention_desc() const;
memory::desc desc::weights_layer_desc() const;
memory::desc desc::weights_iter_desc() const;
memory::desc desc::bias_desc() const;
memory::desc desc::dst_layer_desc() const;
memory::desc desc::dst_iter_desc() const;

// Additional for backward.
memory::desc desc::diff_src_layer_desc() const;
memory::desc desc::diff_src_iter_desc() const;
memory::desc desc::diff_attention_desc() const;
memory::desc desc::diff_weights_layer_desc() const;
memory::desc desc::diff_weights_iter_desc() const;
memory::desc desc::diff_bias_desc() const;
memory::desc desc::diff_dst_layer_desc() const;
memory::desc desc::diff_dst_iter_desc() const;
```
</details>

<details> <summary> RNN: LBR AUGRU </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_cell_kind() const;
dnnl::prop_kind desc::get_prop_kind() const;
dnnl::rnn_direction desc::get_direction() const;
dnnl::rnn_flags desc::get_flags() const;

memory::desc desc::src_layer_desc() const;
memory::desc desc::src_iter_desc() const;
memory::desc desc::attention_desc() const;
memory::desc desc::weights_layer_desc() const;
memory::desc desc::weights_iter_desc() const;
memory::desc desc::bias_desc() const;
memory::desc desc::dst_layer_desc() const;
memory::desc desc::dst_iter_desc() const;

// Additional for backward.
memory::desc desc::diff_src_layer_desc() const;
memory::desc desc::diff_src_iter_desc() const;
memory::desc desc::diff_attention_desc() const;
memory::desc desc::diff_weights_layer_desc() const;
memory::desc desc::diff_weights_iter_desc() const;
memory::desc desc::diff_bias_desc() const;
memory::desc desc::diff_dst_layer_desc() const;
memory::desc desc::diff_dst_iter_desc() const;
```
</details>

<details> <summary> Shuffle </summary>

```cpp
// For all propagation kinds.
dnnl::prop_kind desc::get_prop_kind() const;
int desc::get_axis() const;
dim desc::get_group_size() const;

// Additional for forward.
memory::desc desc::src_desc() const;
memory::desc desc::dst_desc() const;

// Additional for backward.
memory::desc desc::diff_src_desc() const;
memory::desc desc::diff_dst_desc() const;
```
</details>

<details> <summary> Binary </summary>

```cpp
dnnl::algorithm desc::get_algorithm() const;
memory::desc desc::src_desc(int idx = 0) const;
memory::desc desc::src0_desc() const;
memory::desc desc::src1_desc() const;
memory::desc desc::dst_desc() const;
```
</details>

<details> <summary> Matmul </summary>

```cpp
memory::desc desc::src_desc() const;
memory::desc desc::weights_desc() const;
memory::desc desc::bias_desc() const;
memory::desc desc::dst_desc() const;
```
</details>

<details> <summary> Pooling </summary>

```cpp
// For all propagation kinds.
dnnl::algorithm desc::get_algorithm() const;
dnnl::prop_kind desc::get_prop_kind() const;
dims desc::get_kernel() const;
memory::dims desc::get_strides() const;
memory::dims desc::get_dilation() const;
memory::dims desc::get_padding_l() const;
memory::dims desc::get_padding_r() const;

// Additional for forward.
memory::desc desc::src_desc() const;
memory::desc desc::dst_desc() const;

// Additional for backward.
memory::desc desc::diff_src_desc() const;
memory::desc desc::diff_dst_desc() const;
```
</details>

<details> <summary> PReLU </summary>

```cpp
// For all propagation kinds.
dnnl::prop_kind desc::get_prop_kind() const;
memory::desc desc::src_desc() const;
memory::desc desc::dst_desc() const;
memory::desc desc::weights_desc() const;

// Additional for backward.
memory::desc desc::diff_src_desc() const;
memory::desc desc::diff_dst_desc() const;
memory::desc desc::diff_weights_desc() const;
```
</details>

<details> <summary> Reduction </summary>

```cpp
float desc::get_p() const;
float desc::get_epsilon() const;
memory::desc desc::src_desc() const;
memory::desc desc::dst_desc() const;
```
</details>

### Comparison Semantics
The current comparison semantics is defined as follows - two operation descriptors
are considered equal when all data members of those are equal.

In order to preserve the semantics oneDNN needs to provide an API to compare opaque
operation descriptors.

#### C API
```cpp
int dnnl_op_desc_equal(const_dnnl_op_desc_t lhs, const_dnnl_op_desc_t rhs);
```

#### C++ API
```cpp
bool op_desc::operator==(const op_desc &other);
```

### Cloning
Operation descriptors can be returned as a result of different queries. The
queries will return a pointer to opaque operation descriptors therefore oneDNN
needs to provide an API to clone them.

#### C API
```cpp
dnnl_status_t dnnl_op_desc_clone(dnnl_memory_desc_t *memory_desc,
        const_dnnl_memory_desc_t existing_memory_desc);
```
Current C++ API doesn't provide any API to query operation descriptor from the
primitive descriptor therefore there is nothing to modify.

### API Use Example
An example that shows the differences between old and new APIs.

```cpp
// C API
dnnl_op_desc_t cd; // Opaque.
dnnl_convolution_forward_desc_create(&cd, dnnl_forward, ...);

// Query `strides`.
dnnl_dim_t *strides;
dnnl_op_desc_query(cd, dnnl_query_strides, 0, (void *)&strides);
dnnl_dim_t conv_stride_h = strides[0];
dnnl_dim_t conv_stride_w = strides[1];
dnnl_op_desc_destroy(cd); // New function.
// `strides` pointer is invalidated.

// C++ API
memory::dims strides;
{
    auto cd = convolution_forward::desc(prop_kind::forward_inference, ...);
    strides = cd.get_strides();
    // dnnl_dims_t strides = cd.data.strides; // Does not work anymore.
}
// `strides` is valid even though `cd` was destroyed.
```
## Operation Descriptor: Option 2 - Remove Operation Descriptor
This option suggests to remove operation descriptor completely. Primitive descriptor
will take the primitives parameters upon creation directly.

The rational behind this option is that the operation descriptor abstraction
seems to be redundant because it merely serves as an interim step of primitive
creation without additional responsibilities.

Originally, operation descriptor was meant to be a lightweight abstraction for
holding primitive specific parameters. This is why it was decided to define it as
a POD data type. Operation descriptor makes it possible to have a single API for
primitive descriptor creation as operation descriptor is passed to the API as
`void *`. Opaque operation descriptor is no longer a lightweight abstraction
therefore having it in the API becomes a burden that makes the API more verbose and
heavier without good reason.

### Creation/Destruction of Primitive Descriptor
Since operation descriptor is no longer available the API for creating primitive
descriptor must be adjusted to take primitive parameters that previously were passed
with the operation descriptor. The number of APIs for creating primitive descriptor
will be equal to the number of APIs for creating operation descriptors. But since the
latter will be removed the total number of APIs will stay almost unchanged.

#### C API

Currently, there are two ways to create a primitive descriptor:
1. Directly via the corresponding API
2. Using primitive descriptor iterator

The API for creating a primitive descriptor and primitive descriptor iterator are
almost identical:

```cpp
// Primitive descriptor.
dnnl_status_t dnnl_primitive_desc_create(
        dnnl_primitive_desc_t *primitive_desc,
        const_dnnl_op_desc_t op_desc, const_dnnl_primitive_attr_t attr,
        dnnl_engine_t engine, const_dnnl_primitive_desc_t hint_forward_primitive_desc);
```

```cpp
// Primitive descriptor iterator.
dnnl_status_t dnnl_primitive_desc_iterator_create(
        dnnl_primitive_desc_iterator_t *iterator,
        const_dnnl_op_desc_t op_desc, const_dnnl_primitive_attr_t attr,
        dnnl_engine_t engine, const_dnnl_primitive_desc_t hint_forward_primitive_desc);
```

This means that removing operation descriptor will require adjust both APIs, which
will lead to duplication. To avoid the duplication the proposal is to add iterator
capabilities to primitive descriptor and remove primitive descriptor iterator from API.

From API perspective, there will be introduced a new API for primitive descriptor:
```cpp
dnnl_status_t dnnl_primitive_desc_next_impl(dnnl_primitive_desc_t primitive_desc);
```
This will be aligned with C++ API, where primitive descriptor provides `next_impl()`
API.

From implementation perspective, `primitive_desc_iface_t` will be adjusted so that
it contains primitive descriptor iterator instead of primitive descriptor itself.

The current API for creating primitive descriptor will be replaced with a set of the
following APIs:
```cpp
// Depending on primitive kind and propagation kind the API will take different
// arguments.
dnnl_<primitive kind>_[propagation kind]_primitive_desc_create(...);
```

#### C++ API

Concat, sum and reorder primitives don't have operation descriptors, all parameters
are passed directly to primitive descriptor constructor.
All other primitives will be aligned with concat, sum and reorder.

* All nested `desc` classes will be removed
* Each primitive descriptor class will get constructors that take the arguments that
were previously passed to the `desc` classes constructors
* The parameters of the primitive descriptor constructors will be in the following order:
    * FWD: `primitive_desc(const engine &aengine, ..., const primitive_attr &attr = default_attr(), bool allow_empty = false)`
    * BWD: `primitive_desc(const engine &aengine, ..., const primitive_desc &hint_fwd_pd, const primitive_attr &attr = default_attr(), bool allow_empty = false)`

Iterator API will stay unchanged.

### Querying Operation Descriptor Information

Primitive descriptor will not be able to return an operation descriptor therefore
the corresponding query will be removed.

#### C API
The proposal for query capabilities described in the option 1 will be used with the
following changes:
* The API for querying information from `op_desc_t` will not be added
* The existing `dnnl_primitive_desc_query` API will handle all new queries

#### C++ API
The proposal for query capabilities described in the option 1 will be used with the
following changes:
1. No `query_helper` is needed, all the queries will be implemented in
`primitive_desc_base`
2. Query API for each primitive will be implemented in each primitive descriptor
instead of operation descriptor

## Operation Descriptor: Conclusion

### Option 1
* This option allows to hide implementation details of an operation descriptor
and has little impact on users of the C++ API
* On the other hand, there will be an opaque abstraction that doesn't bring any value
but users will still have to create it

### Option 2
* This option also hides implementation details of an operation descriptor
(it may still exist inside the library) and gets rid of a requirement for users to
create it.
* On the other hand it impacts C++ API significantly, though the changes on the user
side are expected to be straightforward - they will need to skip operation descriptor
creation and proceed with creating primitive descriptor using the same arguments that
were previously  used to create operation descriptor
* Query API will be more compact. For the first option there will be a query API that
return an operation descriptor, which then can be used to query information. With the
second option that information can be queried directly from primitive descriptor

Unless we get too much negative feedback for the second option, it looks like it's
a better way to make oneDNN API more scalable.

## Open Questions
* Is there a need to expose `extra` information about memory descriptor?
    * Other than setting a custom adjustment scale there doesn't seem to be much
    sense in doing it
* Is there a need to provide a way to re-initialize memory and operation
descriptor?
    * On the one hand, with the existing API it's possible to create a
    descriptor once and initialize it many times. With the proposed API a
    descriptor can be initialized only once during creation.
    * On the other hand, C++ API is the main API and descriptors are normally
    created once and not re-used therefore the ability to re-initialize
    descriptors doesn't look very useful.

