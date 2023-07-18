Group Normalization {#dev_guide_group_normalization}
====================================================

>
> [API Reference](@ref dnnl_api_group_normalization)
>

## General

The group normalization primitive performs a forward or backward group
normalization operation on tensors with numbers of dimensions equal to 3 or more.

### Forward

The group normalization operation is defined by the following formulas. We show
formulas only for 2D spatial data which are straightforward to generalize to
cases of higher and lower dimensions. Variable names follow the standard
@ref dev_guide_conventions.

\f[
    \dst(n, g \cdot C_G + c_g, h, w) =
       \gamma(g \cdot C_G + c_g) \cdot
       \frac{\src(n, g \cdot C_G + c_g, h, w) - \mu(n, g)} {\sqrt{\sigma^2(n, g) + \varepsilon}}
       + \beta(g \cdot C_G + c_g),
\f]

where

- \f$C_G = \frac{C}{G}\f$,

- \f$c_g \in [0, C_G).\f$,

- \f$\gamma(c), \beta(c)\f$ are optional scale and shift for a channel
(see #dnnl_use_scale and #dnnl_use_shift flags),

- \f$\mu(n, g), \sigma^2(n, g)\f$ are mean and variance for a group of channels in a batch
(see #dnnl_use_global_stats flag), and

- \f$\varepsilon\f$ is a constant to improve numerical stability.

Mean and variance are computed at runtime or provided by a user. When mean and
variance are computed at runtime, the following formulas are used:

- \f$\mu(n, g) = \frac{1}{(C/G)HW} \sum\limits_{c_ghw} \src(n, g \cdot C_G + c_g, h, w)_{}\f$,

- \f$\sigma^2(n, g) = \frac{1}{(C/G)HW} \sum\limits_{c_ghw} {}_{} (\src(n, g \cdot C_G + c_g, h, w) - \mu(n, g))^2\f$.

The \f$\gamma(c)\f$ and \f$\beta(c)\f$ tensors are considered learnable.


@note
* The group normalization primitive computes population mean and variance and
  not the sample or unbiased versions that are typically used to compute
  running mean and variance.
* Using the mean and variance computed by the group normalization primitive,
  running mean and variance \f$\hat\mu\f$ and \f$\hat\sigma^2\f$ can be
  computed as \f[
    \hat\mu := \alpha \cdot \hat\mu + (1 - \alpha) \cdot \mu, \\
    \hat\sigma^2 := \alpha \cdot \hat\sigma^2 + (1 - \alpha) \cdot \sigma^2.
  \f]

#### Difference Between Forward Training and Forward Inference

 * If mean and variance are computed at runtime (i.e., #dnnl_use_global_stats
   is not set), they become outputs for the propagation kind
   #dnnl_forward_training (because they would be required during the backward
   propagation) and are not exposed for the propagation kind
   #dnnl_forward_inference.

### Backward

The backward propagation computes
\f$\diffsrc(n, c, h, w)\f$,
\f$\diffgamma(c)^*\f$, and \f$\diffbeta(c)^*\f$
based on
\f$\diffdst(n, c, h, w)\f$, \f$\src(n, c, h, w)\f$, \f$\mu(n, g)\f$,
\f$\sigma^2(n, g)\f$, \f$\gamma(c) ^*\f$, and \f$\beta(c) ^*\f$.

The tensors marked with an asterisk are used only when the primitive is
configured to use \f$\gamma(c)\f$ and \f$\beta(c)\f$ (i.e.,
#dnnl_use_scale or #dnnl_use_shift are set).

## Execution Arguments

Depending on the [flags](@ref dnnl_normalization_flags_t) and
[propagation kind](@ref dnnl_prop_kind_t), the group normalization primitive
requires different inputs and outputs.  For clarity, a summary is shown below.

| Flags                                                        | #dnnl_forward_inference                                                                                            | #dnnl_forward_training                                                                                                                                                                   | #dnnl_backward                                                                                                                                                                                | #dnnl_backward_data                                                                                              |
|:-------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
| #dnnl_normalization_flags_none                               | *Inputs*: \src <br><br> *Outputs*: \dst                                                                            | *Inputs*: \src <br><br> *Outputs*: \dst, \f$\mu\f$, \f$\sigma^2\f$                                                                                                                       | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \diffsrc                                                                                                              | Same as for #dnnl_backward                                                                                       |
| #dnnl_use_global_stats                                       | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \dst                                                 | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \dst                                                                                                                       | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \diffsrc                                                                                                              | Same as for #dnnl_backward                                                                                       |
| #dnnl_use_scale                                              | *Inputs*: \src, \f$\gamma\f$  <br><br> *Outputs*: \dst                                                             | *Inputs*: \src, \f$\gamma\f$ <br><br> *Outputs*: \dst, \f$\mu\f$, \f$\sigma^2\f$                                                                                                         | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$ <br><br> *Outputs*: \diffsrc, \f$\diffgamma\f$                                                                              | Not supported                                                                                                    |
| #dnnl_use_shift                                              | *Inputs*: \src, \f$\beta\f$ <br><br> *Outputs*: \dst                                                               | *Inputs*: \src, \f$\beta\f$ <br><br> *Outputs*: \dst, \f$\mu\f$, \f$\sigma^2\f$                                                                                                          | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\beta\f$ <br><br> *Outputs*: \diffsrc, \f$\diffbeta\f$                                                                                | Not supported                                                                                                    |
| #dnnl_use_global_stats \| #dnnl_use_scale \| #dnnl_use_shift | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \dst                      | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \dst                                                                                            | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \diffsrc, \f$\diffgamma\f$, \f$\diffbeta\f$                                                | Not supported                                                                                                    |

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive Input/Output      | Execution Argument Index                                                  |
|-----------------------------|---------------------------------------------------------------------------|
| \src                        | DNNL_ARG_SRC                                                              |
| \f$\gamma\f$                | DNNL_ARG_SCALE                                                            |
| \f$\beta\f$                 | DNNL_ARG_SHIFT                                                            |
| mean (\f$\mu\f$)            | DNNL_ARG_MEAN                                                             |
| variance (\f$\sigma^2\f$)   | DNNL_ARG_VARIANCE                                                         |
| \dst                        | DNNL_ARG_DST                                                              |
| \diffdst                    | DNNL_ARG_DIFF_DST                                                         |
| \diffsrc                    | DNNL_ARG_DIFF_SRC                                                         |
| \f$\diffgamma\f$            | DNNL_ARG_DIFF_SCALE                                                       |
| \f$\diffbeta\f$             | DNNL_ARG_DIFF_SHIFT                                                       |
| \f$\text{binary post-op}\f$ | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_1 |

## Implementation Details

### General Notes

1. The different flavors of the primitive are partially controlled by the @p
   flags parameter that is passed to the primitive descriptor creation
   function (e.g., dnnl::group_normalization_forward::primitive_desc()).
   Multiple flags can be set using the bitwise OR operator (`|`).

2. For forward propagation, the mean and variance might be either computed at
   runtime (in which case they are outputs of the primitive) or provided by
   a user (in which case they are inputs). In the latter case, a user must set
   the #dnnl_use_global_stats flag. For the backward propagation, the mean and
   variance are always input parameters.

3. Both forward and backward propagation support in-place operations, meaning
   that \src can be used as input and output for forward propagation, and
   \diffdst can be used as input and output for backward propagation. In case of
   an in-place operation, the original data will be overwritten. Note, however,
   that backward propagation requires the original \src, hence the corresponding
   forward propagation should not be performed in-place.

### Data Type Support

The operation supports the following combinations of data types:

| Propagation        | Source / Destination | Mean / Variance / ScaleShift |
|:-------------------|:---------------------|:-----------------------------|
| forward / backward | f32, bf16, f16       | f32                          |
| forward            | s8                   | f32                          |

@warning
    There might be hardware- or implementation-specific restrictions. Check the
    [Implementation Limitations](@ref dg_gnorm_impl_limits) section below.

### Data Representation

#### Mean and Variance

The mean (\f$\mu\f$) and variance (\f$\sigma^2\f$) are separate 2D tensors of
size \f$N \times G\f$.

The format of the corresponding memory object must be #dnnl_nc (#dnnl_ab).

#### Scale and Shift

If #dnnl_use_scale or #dnnl_use_shift are used, the scale (\f$\gamma\f$) and
shift (\f$\beta\f$) are separate 1D tensors of shape \f$C\f$.

The format of the corresponding memory object must be #dnnl_x (#dnnl_a).

#### Source, Destination, and Their Gradients

The group normalization primitive expects data to be
 \f$N \times C \times SP_n \times \cdots \times SP_0\f$ tensor.

The group normalization primitive is optimized for the following memory formats:

| Spatial | Logical tensor | Implementations optimized for memory formats         |
|:--------|:---------------|:-----------------------------------------------------|
| 1D      | NCW            | #dnnl_ncw (#dnnl_abc), #dnnl_nwc (#dnnl_acb)         |
| 2D      | NCHW           | #dnnl_nchw (#dnnl_abcd), #dnnl_nhwc (#dnnl_acdb)     |
| 3D      | NCDHW          | #dnnl_ncdhw (#dnnl_abcde), #dnnl_ndhwc (#dnnl_acdeb) |


### Post-Ops and Attributes

Attributes enable you to modify the behavior of the group normalization
primitive. The following attributes are supported by the group normalization
primitive:

| Propagation | Type      | Operation                                            | Description                                                   | Restrictions                                                                       |
|:------------|:----------|:-----------------------------------------------------|:--------------------------------------------------------------|:-----------------------------------------------------------------------------------|
| forward     | attribute | [Scales](@ref dnnl::primitive_attr::set_scales_mask) | Scales the corresponding tensor by the given scale factor(s). | Supported only for int8 group normalization and one scale per tensor is supported. |
| forward     | Post-op   | [Binary](@ref dnnl::post_ops::append_binary)         | Applies a @ref dnnl_api_binary operation to the result        | General binary post-op restrictions                                                |
| forward     | Post-op   | [Eltwise](@ref dnnl::post_ops::append_eltwise)       | Applies an @ref dnnl_api_eltwise operation to the result.     |                                                                                    |

@anchor dg_gnorm_impl_limits
## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. GPU is not supported.

## Performance Tips

1. Mixing different formats for inputs and outputs is functionally supported but
   leads to highly suboptimal performance.

2. Use in-place operations whenever possible (see caveats in General Notes).


## Examples

[Group Normalization Primitive Example](@ref group_normalization_example_cpp)

@copydetails group_normalization_example_cpp_short
