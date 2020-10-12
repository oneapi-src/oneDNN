# RFC: Max pooling to support global indices as one of the outputs

## Motivation

There are some
[neural networks for semantic segmentation](https://arxiv.org/pdf/1505.04366.pdf)
which implement the network as two chains: one half is a convolutional chain
with pooling operations in between, another half is a deconvolutional chain with
**unpooling** operations in between. An unpooling operation does the opposite
to pooling, it upsamples the image based on the indices obtained from "paired"
pooling and the output from a previous operation. Currently, oneDNN does not
give an opportunity to an end user to obtain global indices of local max values.
The request from PyTorch framework is to add such ability to the max pooling
operation. The similar request came from ONNX Runtime Team to support their
feature of running forward and backward pooling in different sub-graphs which
may execute on different vendor hardware and/or software.

We are not requested to implement an unpooling primitive so far as it is of low
priority and used in not very popular models (it's not in PyTorch's Model Zoo).
But even if we were to implement unpooling, we can't rely on workspace memory
from pooling since in general case, the library expects that pooling to be
called using only oneDNN implementation but not native framework's or user's
code. But even with the call to oneDNN pooling, using the same workspace would
be really complicated since the memory layout of unpooling may easily differ
from pooling one, and parameters of pooling and unpooling do not have to
coincide - they may have different kernel, padding, strides and output sizes.

The last piece of motivation is such functionality presents in most frameworks,
where integration with oneDNN exists, so, covering it would be a nice thing from
the library side.

## Definition

Analyzing the definition of the operation, it was figured out that TensorFlow,
PyTorch and ONNX define "global" term differently. Other frameworks were not
inspected.

List of definitions:
* [PyTorch](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/DilatedMaxPool2d.cpp)
  code sample since documentation does not give a hint what indices look like.
* [ONNX](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool)
* [TensorFlow](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/max-pool-with-argmax)

The summary of each definition is that each side implements it in their own way:
* PyTorch only uses spatial values considering them global. Since natively
  supported formats are nchw and nhwc, such notation is agnostic to memory
  layout and can be considered as spatial logical offset.
* ONNX uses full tensor values to specify the global max argument value. ONNX
  supports only nchw memory layout, thus, can also be considered as providing
  logical offset for a final value.
* TensorFlow has two modes: to involve a mini-batch dimension into the final
  value or not. But it always has spatial and channels dimensions to be included
  in the final value. Since the default logical dimensions layout is nhwc, the
  final value also represents logical offset (refer to the doc description).

To satisfy each side, there is no other way but to provide an extensible
solution with adding options there for future support of indices values
flavor.

## Additional challenges

It is also a known fact that implementations are different across frameworks and
the key part for this RFC is what index to put in case the max argument in the
kernel window is not the only one. Some implementations pick the first one, or
the smallest index value, others pick the last one, or the biggest index value.
This detail affects the backward path result of `diff_src` tensor.

It is worth mentioning that oneDNN already received a request from PyTorch to
provide a mechanism to control the index order. Using current implementation
oneDNN supports (picking the first index) results in internal unit tests
failures.

## Proposal

### Option 1 (recommended) - new primitive.

Since max argument indices should have memory layout represented, one of the
ways to specify it is through a new pooling descriptor. Since we are preserving
ABI, that would mean a new set of API calls to utilize a new version of pooling
v3 descriptor.

In addition to a new descriptor, it is proposed to add a new algorithm. It will
instruct the implementation to dump global indices to a second output marked as
`DNNL_ARG_DST_1`. By introducing a new algorithm, we preserve performance for
the case when output indices are not desired, since it will double the amount of
memory to write for a kernel.

~~~c
/* dnnl_types.h */

typedef enum {
    ...
    /// Average pooling exclude padding
    dnnl_pooling_avg_exclude_padding = 0x3ff,
    /// Max pooling with second output of global indices
    dnnl_pooling_max_with_indices = 0x4ff,
    ...
} dnnl_alg_kind_t;
~~~

And the last piece proposed is attributes extension for additional control over
"challenges" and different definitions mentioned above. Attributes will be
responsible for the final global indices’ values. This could be alternatively
done by extending `alg_kind` values for each flavor, but since these attributes’
values do not require any checks and just modify the definition of the
operation, it will allow to keep API slightly cleaner by asking either global
indices are required or not.

Based on user's choices for attributes' values, once passing to pooling
primitive descriptor, they will modify the behavior of the final values in
global indices memory. Global indices memory will be allowed to have user's
specified layout or `any`. In latter case the layout will coincide with
destination memory layout for best performance.

~~~c
/* dnnl_types.h */

typedef enum {
    /// Pooling to use the first max argument index
    dnnl_pooling_index_order_first,
    /// Pooling to use the last max argument index
    dnnl_pooling_index_order_last,
} dnnl_pooling_index_order_t;

typedef enum {
    /// Pooling to use the full tensor logical offset. Final indices values are
    /// in range of [0, N x C x D x H x W).
    dnnl_pooling_indices_values_all_dims,
    /// Pooling to use the tensor logical offset without mini-batch. Final
    /// indices values are in range of [0, C x D x H x W).
    dnnl_pooling_indices_values_c_spatial,
    /// Pooling to use the tensor logical offset of spatial values only. Final
    /// indices values are in range of [0, D x H x W).
    dnnl_pooling_indices_values_spatial,
} dnnl_pooling_indices_values_t;

/* dnnl.h */

/// Sets the primitive attributes pooling index order.
///
/// @param attr Primitive attributes.
/// @param index_order Pooling index order. The possible values are:
///     #dnnl_pooling_index_order_first (default) and
///     #dnnl_pooling_index_order_last.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_pooling_index_order(
        dnnl_primitive_attr_t attr, dnnl_pooling_index_order_t index_order);

/// Returns the primitive attributes pooling index order.
///
/// @param attr Primitive attributes.
/// @param index_order Output pooling index order.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_pooling_index_order(
        const_dnnl_primitive_attr_t attr,
        dnnl_pooling_index_order_t *index_order);

/// Sets primitive attributes pooling global indices values. Does not take any
/// effect for any alg_kind values except `dnnl_pooling_max_with_indices`.
///
/// @param attr Primitive attributes.
/// @param indices_values Pooling indices values. The possible values are:
///     #dnnl_pooling_indices_values_all_dims (default),
///     #dnnl_pooling_indices_values_c_spatial and
///     #dnnl_pooling_indices_values_spatial.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_set_pooling_indices_values(
        dnnl_primitive_attr_t attr,
        dnnl_pooling_indices_values_t indices_values);

/// Returns the primitive attributes pooling global indices values.
///
/// @param attr Primitive attributes.
/// @param indices_values Output pooling global indices values.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_attr_get_pooling_indices_values(
        const_dnnl_primitive_attr_t attr,
        dnnl_pooling_indices_values_t *indices_values);
~~~

Here is the declaration of `dnnl_pooling_v3_desc_t` and correspondent calls to
its creation:

~~~c
/* dnnl_types.h */

typedef struct {
    /// Same fileds as in dnnl_pooling_v2_desc_t
    ...
    /// Pooling maximum arguments output indices memory descriptor.
    dnnl_memory_desc_t indices_desc;
} dnnl_pooling_v3_desc_t;

/* dnnl.h */

/// Initializes a descriptor for pooling v3 (pooling with global indices
/// support) forward propagation primitive.
///
/// Same arguments as for dnnl_pooling_v2_forward_desc_init(...);
/// ...
/// @param indices_desc Pooling maximum arguments output indices memory
///     descriptor. If `alg_kind` is set to `dnnl_pooling_max_with_indices`,
///     cannot be NULL. In other cases, will be ignored.
/// ...
dnnl_status_t DNNL_API dnnl_pooling_v3_forward_desc_init(
        dnnl_pooling_v3_desc_t *pool_desc, dnnl_prop_kind_t prop_kind,
        dnnl_alg_kind_t alg_kind, const dnnl_memory_desc_t *src_desc,
        const dnnl_memory_desc_t *dst_desc,
        const dnnl_memory_desc_t *indices_desc, const dnnl_dims_t strides,
        const dnnl_dims_t kernel, const dnnl_dims_t dilation,
        const dnnl_dims_t padding_l, const dnnl_dims_t padding_r);

/// Initializes a descriptor for pooling v3 (pooling with global indices
/// support) backward propagation primitive.
///
/// Same arguments as for dnnl_pooling_v2_backward_desc_init(...);
/// ...
/// @param indices_desc Pooling maximum arguments output indices memory
///     descriptor. If `alg_kind` is set to `dnnl_pooling_max_with_indices`,
///     cannot be NULL. In other cases, will be ignored.
/// ...
dnnl_status_t DNNL_API dnnl_pooling_v3_backward_desc_init(
        dnnl_pooling_v3_desc_t *pool_desc, dnnl_alg_kind_t alg_kind,
        const dnnl_memory_desc_t *diff_src_desc,
        const dnnl_memory_desc_t *diff_dst_desc,
        const dnnl_memory_desc_t *indices_desc, const dnnl_dims_t strides,
        const dnnl_dims_t kernel, const dnnl_dims_t dilation,
        const dnnl_dims_t padding_l, const dnnl_dims_t padding_r);
~~~

### Option 2 - new primitive without a new alg_kind.

This option coincides with Option 1, but makes `dnnl_pooling_max` algorithm to
dump global indices unconditionally instead, requiring users to modify the code
and specify the second input, which will break existing pooling API and is not
user-friendly. Another drawback is sacrificing performance for existing
implementation.

### Option 3 - new algorithm and attributes extension only.

The alternative of putting `indices_desc` into a new pooling descriptor to
represent a memory layout through attributes as well. This would look like
binary post-ops API. Since we need only memory format and data type, this could
utilize `dnnl_format_tag_t tag` value and/or `dnnl_dims_t strides` array and
`dnnl_data_type_t data_type` value instead a whole memory descriptor.

No API example provided yet as requires broad group discussion whether this
option is viable.

## Implementation

Disclaimer: this paragraph does not belong to the proposal itself but sheds some
light on potential implementation choices just for information purposes.

The idea is to keep global indices internally for spatial case and when the
outer parallel loops finished, modify the value if `c_spatial` or `all_dims`
options was requested. It also will reorder the memory layout if it was asked
to be different from destination memory layout, since using same formats
internally will help to speed up computations.

## Backward

Existing max pooling implementation for best performance purpose requires
passing a workspace, which signature is encrypted from an end user and based on
the implementation. The idea is still to request the workspace on forward
despite the fact the global indices were requested. This demand allows the user
to use faster implementation on backward by using `dnnl_pooling_max` algorithm
and passing the workspace (not global indices) since there is an expectation
that the library will still have a single JIT-ted implementation for all
pooling flavors.

In case `dnnl_pooling_max_with_indices` is requested on backward, it will demand
for global indices memory and provide an implementation using global indices
(not workspace), which is expected to perform worse (though it is a pure
speculation at this point of time).

## Testing

Benchdnn will require attributes extension, pooling driver will require new
options to pass the flavor of global indices output and the order. Reference
implementation will require adjustment to respect those new options and their
values as well. Further comparison will consider point-to-point comparison of
both destination value and global index value.

EOD.
