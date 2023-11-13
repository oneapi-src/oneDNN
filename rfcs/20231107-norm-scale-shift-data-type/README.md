# Introducing API for Specifying Data Type of Scale and Shift Tensors

## Motivation

Some frameworks may require scale and shift tensors to have the same
data type as the output tensor. Currently, oneDNN normalization primitives
such as batch normalization, layer normalization and group normalization
require the scale and shift tensors to always be in `f32`. In certain scenarios
(e.g. when the output tensor is in `bf16`) this requirement forces the frameworks
to convert the scale and shift tensors to `f32` to use the oneDNN normalization
primitives, which incurs unnecessary overhead.

## Proposal

Currently, oneDNN doesn't provide any API to specify the data type for the
scale and shift tensors. The proposal is to introduce such API.

The allowed data types will be `f32`, `bf16` and `f16`.

The proposal will be focused on the layer normalization primitive because the
feature has been requested specifically for it, but the proposed API can be scaled
up to the batch and group normalization primitives.

### Separate Data Types vs Single Data Type

Hypothetically, since the scale and shift tensors are separate their data types
could be different too. However, based on the feedback from the frameworks they
don't anticipate such uses cases. If such use cases come up at some point in
the future a new API can be introduced to specify the separate data types.
As of now, using a single data type seems to be more reasonable.

### Data Type Alone vs Memory Descriptor

There are options for the user to specify the data type by either providing it
directly or via a memory descriptor. Usually, when it comes to tensors oneDNN
requires users to describe them with a memory descriptor. On the other hand, the
only useful information from the memory descriptor for the scale and shift tensors
would be data type because the tensors are always 1D and the number of elements in
them is always equal to the number of channels.

Requiring users to create a memory descriptor for the scale and shift tensors seems
redundant and may incur unnecessary overhead without providing any benefits therefore
the proposal is to go with the data type option.

### API

Assuming that there is only a single data type for the scale and shift tensors and
it is specified directly, the API would be defined as follows.

```c
/// Creates a primitive descriptor for a layer normalization forward propagation
///     primitive with a user-provided data type for the scale and shift
///     memory objects.
///
/// @note
///     In-place operation is supported: the dst can refer to the same memory
///     as the src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_forward_training and #dnnl_forward_inference.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #dnnl_format_kind_undef, then the memory
///     descriptor for stats is derived from @p src_desc by removing the last
///     dimension.
/// @param scale_shift_data_type Data type of scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref dnnl_normalization_flags_t).
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_layer_normalization_forward_primitive_desc_create_v2(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t src_desc,
        const_dnnl_memory_desc_t dst_desc, const_dnnl_memory_desc_t stat_desc,
        dnnl_data_type_t scale_shift_data_type, float epsilon, unsigned flags,
        const_dnnl_primitive_attr_t attr);
```
```c
/// Creates a primitive descriptor for a layer normalization backward
///     propagation primitive with a user-provided data type for the
///     scale and shift memory objects.
///
/// @note
///     In-place operation is supported: the diff_dst can refer to the same
///     memory as the diff_src.
///
/// @param primitive_desc Output primitive_descriptor.
/// @param engine Engine to use.
/// @param prop_kind Propagation kind. Possible values are
///     #dnnl_backward_data and #dnnl_backward (diffs for all parameters are
///     computed in this case).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param src_desc Source memory descriptor.
/// @param stat_desc Memory descriptor for mean and variance. If this
///     parameter is NULL, a zero memory descriptor, or a memory descriptor
///     with format_kind set to #dnnl_format_kind_undef, then the memory
///     descriptor for stats is derived from @p src_desc by removing the last
///     dimension.
/// @param diff_scale_shift_data_type Data type of diff scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param scale_shift_data_type Data type of scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref dnnl_normalization_flags_t).
/// @param hint_fwd_pd Primitive descriptor for a respective forward propagation
///     primitive.
/// @param attr Primitive attributes (can be NULL).
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_layer_normalization_backward_primitive_desc_create_v2(
        dnnl_primitive_desc_t *primitive_desc, dnnl_engine_t engine,
        dnnl_prop_kind_t prop_kind, const_dnnl_memory_desc_t diff_src_desc,
        const_dnnl_memory_desc_t diff_dst_desc,
        const_dnnl_memory_desc_t src_desc, const_dnnl_memory_desc_t stat_desc,
        dnnl_data_type_t diff_scale_shift_data_type, dnnl_data_type_t scale_shift_data_type,
        float epsilon, unsigned flags, const_dnnl_primitive_desc_t hint_fwd_pd,
        const_dnnl_primitive_attr_t attr);
```

C++ API:
```cpp
/// Constructs a primitive descriptor for a layer normalization forward
/// propagation primitive with a user-provided data type for the scale and shift
/// memory objects.
///
/// @param aengine Engine to use.
/// @param aprop_kind Propagation kind. Possible values are
///     #dnnl::prop_kind::forward_training, and
///     #dnnl::prop_kind::forward_inference.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param stat_desc Statistics memory descriptors.
/// @param scale_shift_data_type Data type of scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref
///     dnnl::normalization_flags).
/// @param attr Primitive attributes to use. Attributes are optional
///     and default to empty attributes.
/// @param allow_empty A flag signifying whether construction is
///     allowed to fail without throwing an exception. In this case an
///     empty object will be produced. This flag is optional and
///     defaults to false.
layer_normalization_forward::primitive_desc::primitive_desc(
        const engine &aengine, prop_kind aprop_kind, const memory::desc &src_desc,
        const memory::desc &dst_desc, const memory::desc &stat_desc,
        memory::data_type scale_shift_data_type, float epsilon, normalization_flags flags,
        const primitive_attr &attr = default_attr(), bool allow_empty = false);

/// Constructs a primitive descriptor for a layer normalization forward
/// propagation primitive with a user-provided data type for the scale and shift
/// memory objects.
///
/// @param aengine Engine to use.
/// @param aprop_kind Propagation kind. Possible values are
///     #dnnl::prop_kind::forward_training, and
///     #dnnl::prop_kind::forward_inference.
/// @param src_desc Source memory descriptor.
/// @param dst_desc Destination memory descriptor.
/// @param diff_scale_shift_data_type Data type of diff scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param scale_shift_data_type Data type of scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref
///     dnnl::normalization_flags).
/// @param attr Primitive attributes to use. Attributes are optional
///     and default to empty attributes.
/// @param allow_empty A flag signifying whether construction is
///     allowed to fail without throwing an exception. In this case an
///     empty object will be produced. This flag is optional and
///     defaults to false.
layer_normalization_forward::primitive_desc::primitive_desc(
        const engine &aengine, prop_kind aprop_kind, const memory::desc &src_desc,
        const memory::desc &dst_desc, memory::data_type diff_scale_shift_data_type,
        memory::data_type scale_shift_data_type, float epsilon, normalization_flags flags,
        const primitive_attr &attr = default_attr(), bool allow_empty = false);

```
```cpp
/// Constructs a primitive descriptor for a layer normalization backward
/// propagation primitive with a user-provided data type for the scale and shift
/// memory objects.
///
/// @param aengine Engine to use.
/// @param aprop_kind Propagation kind. Possible values are
///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
///     (diffs for all parameters are computed in this case).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param src_desc Source memory descriptor.
/// @param stat_desc Statistics memory descriptors.
/// @param diff_scale_shift_data_type Data type of diff scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param scale_shift_data_type Data type of scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref
///     dnnl::normalization_flags).
/// @param attr Primitive attributes to use. Attributes are optional
///     and default to empty attributes.
/// @param hint_fwd_pd Primitive descriptor for a layer normalization
///     forward propagation primitive. It is used as a hint for
///     deciding which memory format to use.
/// @param allow_empty A flag signifying whether construction is
///     allowed to fail without throwing an exception. In this case an
///     empty object will be produced. This flag is optional and
///     defaults to false.
layer_normalization_backward::primitive_desc::primitive_desc(
        const engine &aengine, prop_kind aprop_kind, const memory::desc &diff_src_desc,
        const memory::desc &diff_dst_desc, const memory::desc &src_desc,
        const memory::desc &stat_desc, memory::data_type diff_scale_shift_data_type,
        memory::data_type scale_shift_data_type, float epsilon, normalization_flags flags,
        const layer_normalization_forward::primitive_desc &hint_fwd_pd,
        const primitive_attr &attr = default_attr(), bool allow_empty = false);

/// Constructs a primitive descriptor for a layer normalization backward
/// propagation primitive with a user-provided data type for the scale and shift
/// memory objects.
///
/// @param aengine Engine to use.
/// @param aprop_kind Propagation kind. Possible values are
///     #dnnl::prop_kind::backward_data and #dnnl::prop_kind::backward
///     (diffs for all parameters are computed in this case).
/// @param diff_src_desc Diff source memory descriptor.
/// @param diff_dst_desc Diff destination memory descriptor.
/// @param src_desc Source memory descriptor.
/// @param diff_scale_shift_data_type Data type of diff scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param scale_shift_data_type Data type of scale and shift memory. If neither scale
///     nor shift flag are specified the parameter is ignored.
/// @param epsilon Layer normalization epsilon parameter.
/// @param flags Layer normalization flags (@ref
///     dnnl::normalization_flags).
/// @param attr Primitive attributes to use. Attributes are optional
///     and default to empty attributes.
/// @param hint_fwd_pd Primitive descriptor for a layer normalization
///     forward propagation primitive. It is used as a hint for
///     deciding which memory format to use.
/// @param allow_empty A flag signifying whether construction is
///     allowed to fail without throwing an exception. In this case an
///     empty object will be produced. This flag is optional and
///     defaults to false.
layer_normalization_backward::primitive_desc::primitive_desc(
        const engine &aengine, prop_kind aprop_kind, const memory::desc &diff_src_desc,
        const memory::desc &diff_dst_desc, const memory::desc &src_desc,
        memory::data_type diff_scale_shift_data_type, memory::data_type scale_shift_data_type,
        float epsilon, normalization_flags flags,
        const layer_normalization_forward::primitive_desc &hint_fwd_pd,
        const primitive_attr &attr = default_attr(), bool allow_empty = false);
```