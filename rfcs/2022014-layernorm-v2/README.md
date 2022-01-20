# RFC: LayerNormalization v2

## Introduction

Several primitives in the library were designed with operation descriptors
containing only `data_desc` member, which covers source and destination memory
descriptors. There was an assumption that such operations do not change memory
format layout through execution nor require a different output data type. And
never would. Those primitives are batch and layer normalizations, eltwise,
prelu, lrn, and shuffle.

These operations, except shuffle, assume that only floating-point types will be
used on input and output of same data type will be produced. It is quite
essential due to the nature and service of these operations, especially,
normalizations. Also, outputs of these operations are clearly distinguished only
in floating-point range of values.

Unfortunately, that assumption did not consider int8 flows when computations may
contain mixture of quantized data and floating-point data. Mentioned primitives
are still executed in floating-point data types such as `f32`, `bf16` or `f16`,
but it usually happens that the following primitives would use quantized output
from mentioned primitives. It means that every time such situation happens, a
reorder from floating-point type to integer type appears.

In Transformer-like models, including BERT, the layer normalizations appear in
every Encoder block. The number of blocks may vary but consider for BERT a base
case of 12 Encoders resulting in 12 reorders for int8 quantization.

Reorder can easily be incorporated into the layer normalization operator, or any
other operation, by letting it store the answer in requested data type. The
problem that `data_desc` does not allow to do that and the only possible
solution is to extend layer normalization descriptor to accept additional memory
descriptor. Quantized output of mentioned primitives would not be meaningful in
terms of operation essence, but this is what's used in production for inference
and was tested accordingly from accuracy point of view.

### Motivation

Major motivation is free performance improvements for Transformer-like models
for int8 inference.

## Proposal

### Description

There are no options to cover different memory descriptors and algorithms other
than creating a new operation descriptor, `dnnl_layer_normalization_v2_desc_t`.
The prior descriptor `dnnl_layer_normalization_desc_t` will also need to be
renamed `dnnl_layer_normalization_v1_desc_t`, but would otherwise be unchanged.
The new descriptor will be extended over previous one with `dnnl_memory_desc_t
dst_desc` and `dnnl_memory_desc_t diff_dst_desc`. `data_desc` analogues will be
replaced with `src_desc` correspondently. Implementation details will be similar
to pooling_v2 and softmax_v2. We will also mark `layer_normalization_v1` API as
deprecated, but in docs and release notes only. No code changes to not frustrate
our users dealing with warning and potential errors (with DNNL_WERROR=ON).

#### C API

```c
/// dnnl_types.h

/// Kinds of primitives. Used to implement a way to extend the library with new
/// primitives without changing the ABI.
typedef enum {
    ...
    /// A layer normalization version 2 primitive (layer normalization with
    /// destination memory descriptor and algorithm kind).
    dnnl_layer_normalization_v2,
    ...
} dnnl_primitive_kind_t;

/// A descriptor of a Layer Normalization operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #dnnl_layer_normalization.
    dnnl_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #dnnl_forward_training,
    /// #dnnl_forward_inference, #dnnl_backward, and #dnnl_backward_data.
    dnnl_prop_kind_t prop_kind;
    /// Source memory descriptor.
    dnnl_memory_desc_t src_desc;
    /// Source gradient memory descriptor.
    dnnl_memory_desc_t diff_src_desc;
    /// Scale and shift data and gradient memory descriptors.
    ///
    /// Scaleshift memory descriptor uses 2D #dnnl_ab
    /// format[2, normalized_dim] where 1-st dimension contains gamma parameter,
    /// 2-nd dimension contains beta parameter. Normalized_dim is equal to the
    /// last logical dimension of the data tensor across which normalization is
    /// performed.
    dnnl_memory_desc_t data_scaleshift_desc;
    dnnl_memory_desc_t diff_data_scaleshift_desc;
    /// Mean and variance data memory descriptors.
    ///
    /// Statistics (mean and variance) memory descriptor is the k-dimensional
    /// tensor where k is equal to data_tensor_ndims - 1 and may have any plain
    /// (stride[last_dim] == 1) user-provided format.
    dnnl_memory_desc_t stat_desc;
    /// Destination memory descriptor.
    dnnl_memory_desc_t dst_desc;
    /// Destination gradient memory descriptor.
    dnnl_memory_desc_t diff_dst_desc;
    /// Layer normalization epsilon parameter.
    float layer_norm_epsilon;
    unsigned flags;
} dnnl_layer_normalization_v2_desc_t;

/// Primitive descriptor query specification
typedef enum {
    ...
    dnnl_query_layer_normalization_v2_d, ///< layer normalization version 2
                                         ///  descriptor
    ...
} dnnl_query_t;
```

```c
/// dnnl.h

// Initializes a descriptor for layer normalization forward propagation
/// primitive.
///
/// @note
///     In-place operation is supported: the dst can refer to the same memory
///     as the src assuming they use the same data type.
///
/// @param lnorm_desc Output descriptor for a layer normalization primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #dnnl_format_kind_undef, then the memory
///     descriptor for stats is derived from @p dst_desc by removing the last
///     dimension.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref dnnl_normalization_flags_t).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_layer_normalization_forward_desc_init_v2(
        dnnl_layer_normalization_v2_desc_t *lnrm_desc,
        dnnl_prop_kind_t prop_kind,
        const dnnl_memory_desc_t *src_desc, const dnnl_memory_desc_t *dst_desc,
        const dnnl_memory_desc_t *stat_desc, float epsilon, unsigned flags);

//// Initializes a descriptor for a layer normalization backward propagation
/// primitive.
///
/// @note
///     In-place operation is supported: the diff_dst can refer to the same
///     memory as the diff_src.
///
/// @param lnrm_desc Output descriptor for layer normalization primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_backward_data and #dnnl_backward (diffs for all parameters are
///     computed in this case).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff desitination memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #dnnl_format_kind_undef, then the memory
///     descriptor for stats is derived from @p dst_desc by removing the last
///     dimension.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref dnnl_normalization_flags_t).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_layer_normalization_backward_desc_init_v2(
        dnnl_layer_normalization_v2_desc_t *lnrm_desc,
        dnnl_prop_kind_t prop_kind, const dnnl_memory_desc_t *diff_src_desc,
        const dnnl_memory_desc_t *diff_dst_desc,
        const dnnl_memory_desc_t *dst_desc,
        const dnnl_memory_desc_t *stat_desc, float epsilon, unsigned flags);
```