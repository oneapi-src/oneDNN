# RFC: Softmax v2

## Introduction

Several primitives in the library were designed with operation descriptors
containing only `data_desc` member, which covers source and destination memory
descriptors. There was an assumption that such operations do not change memory
format layout through execution nor require a different output data type. And
never would. Those primitives are batch and layer normalizations, eltwise,
prelu, lrn, shuffle and softmax.

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

In Transformer-like models, including BERT, softmax layer appears in every
Encoder and Decoder block. The number of blocks may vary but we may consider a
base case when 12 of each are present resulting in 24 reorders bringing their
overhead.

Reorder can easily be incorporated into softmax operation, or any other
operation, by letting it store the answer in requested data type. The problem
that `data_desc` does not allow to do that and the only possible solution is to
extend softmax descriptor to accept additional memory descriptor. Quantized
output of mentioned primitives would not be meaningful in terms of operation
essence, but this is what's used in production for inference and was tested
accordingly from accuracy point of view.

### Motivation

Major motivation is free performance improvements for Transformer-like models
for int8 inference. Additional motivation is to eliminate `logsoftmax` primitive
API which was a result of preserving ABI for users in earlier software versions.

### Additional Feature Requests

oneDNN team was also asked to provide several features through softmax
primitives. One of them was temperature argument, which is supported only in
MXNet.

It was mentioned in Python API though, e.g., R API (first Google link for MXNet
doc) does not have it. None of other frameworks are supporting this parameter.
Following documentation references were used:
[Tensorflow](https://www.tensorflow.org/api_docs/python/tf/nn/softmax),
[PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html),
[OpenVINO](https://docs.openvino.ai/latest/openvino_docs_ops_activation_SoftMax_1.html).
In its core, it would divide the data by `temperature`, see
[MXNet documentation](https://mxnet.apache.org/versions/master/api/python/docs/api/npx/generated/mxnet.npx.softmax.html)
for details. Same can be achieved if the preceding layer would apply
`binary_div` post-op with `mask` of `0`. Due to the lack of broader adoption and
existence of simple work around, such feature will not be added to oneDNN
softmax primitive.

Another long-lasting request is so-called `masked` softmax. It comes from
Transformer-based models when Decoder applies Encoder-Decoder Attention layer.
At this point the machine is not allowed to see future tokens of the sentence,
that's why this "mask" should hide the future tokens starting from the processed
token. Mask makes softmax input data shorter.

Due to the nice property of softmax that data on the left end of range does not
contribute much or at all (depending on the range) to final answer and likely
vanishes, there is a trick to perform this optimization on the user side, which
will be explained shortly. Thus, usually matmul precedes by softmax. The mask
can be incorporated into bias which would fill specific spots with `-inf` or
`INT32_MIN` values and fuse its addition into matmul. Then softmax will handle a
full, non-masked, length but those points will not contribute to the final
answer and will not spoil the accuracy of the model. Such trick allows to handle
a desired behavior much faster than softmax could do it.

Solution via softmax primitive has API, integration, and implementation
complexity. That's why masked feature won't be added to softmax primitive as
well.

## Proposal

### Option 1

There are no options to cover different memory descriptors and algorithms other
than creating a new operation descriptor, `dnnl_softmax_v2_desc_t`. It will be
extended over previous one with `dnnl_alg_kind_t alg_kind`,
`dnnl_memory_desc_t dst_desc` and `dnnl_memory_desc_t diff_dst_desc`.
`data_desc` analogues will be replaced with `src_desc` correspondently.
Implementation details will be similar to pooling_v2. We will also mark
`softmax_v1` and `logsoftmax` API as deprecated, but in docs and release notes
only. No code changes to not frustrate our users dealing with warning and
potential errors (with DNNL_WERROR=ON).

#### C API

```c
/// dnnl_types.h

/// Kinds of primitives. Used to implement a way to extend the library with new
/// primitives without changing the ABI.
typedef enum {
    ...
    /// A softmax version 2 primitive (softmax with destination memory
    /// descriptor and algorithm kind).
    dnnl_softmax_v2,
    ...
} dnnl_primitive_kind_t;

/// Kinds of algorithms.
typedef enum {
    ...
    /// Softmax, numerically stable
    dnnl_softmax_accurate,
    /// Logsoftmax, numerically stable
    dnnl_softmax_log,
} dnnl_alg_kind_t;

/// A descriptor of a Softmax operation.
typedef struct {
    /// The kind of primitive. Used for self-identifying the primitive
    /// descriptor. Must be #dnnl_softmax_v2.
    dnnl_primitive_kind_t primitive_kind;
    /// The kind of propagation. Possible values: #dnnl_forward_training and
    /// #dnnl_forward_inference, and #dnnl_backward_data.
    dnnl_prop_kind_t prop_kind;
    /// Source memory descriptor.
    dnnl_memory_desc_t src_desc;
    /// Source gradient memory descriptor.
    dnnl_memory_desc_t diff_src_desc;
    /// The axis along which to perform the softmax.
    int softmax_axis;
    /// Softmax algorithm. Possible values: #dnnl_softmax_accurate and
    /// #dnnl_softmax_log.
    dnnl_alg_kind_t alg_kind;
    /// Destination memory descriptor.
    dnnl_memory_desc_t dst_desc;
    /// Destination gradient memory descriptor.
    dnnl_memory_desc_t diff_dst_desc;
} dnnl_softmax_v2_desc_t;

/// Primitive descriptor query specification
typedef enum {
    ...
    dnnl_query_softmax_v2_d, ///< softmax version 2 descriptor
    ...
} dnnl_query_t;
```

```c
/// dnnl.h

/// Initializes a descriptor for softmax v2 forward propagation primitive.
///
/// @param softmax_desc Output descriptor for a softmax primitive.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param alg_kind Softmax algorithm kind: either #dnnl_softmax_accurate, or
///     #dnnl_softmax_log.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param softmax_axis Axis over which softmax is computed.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_softmax_v2_forward_desc_init(
        dnnl_softmax_v2_desc_t *softmax_desc, dnnl_prop_kind_t prop_kind,
        dnnl_alg_kind_t alg_kind, const dnnl_memory_desc_t *src_desc,
        const dnnl_memory_desc_t *dst_desc, int softmax_axis);

/// Initializes a descriptor for softmax v2 backward propagation primitive.
///
/// @param softmax_desc Output descriptor for a softmax primitive.
/// @param alg_kind Softmax algorithm kind: either #dnnl_softmax_accurate, or
///     #dnnl_softmax_log.
/// @param diff_src_desc Diff source memory descriptors.
/// @param diff_dst_desc Diff destination memory descriptors.
/// @param dst_desc Destination memory descriptor.
/// @param softmax_axis Axis over which softmax is computed.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_softmax_v2_backward_desc_init(
        dnnl_softmax_v2_desc_t *softmax_desc, dnnl_alg_kind_t alg_kind,
        const dnnl_memory_desc_t *diff_src_desc,
        const dnnl_memory_desc_t *diff_dst_desc,
        const dnnl_memory_desc_t *dst_desc, int softmax_axis);
```
