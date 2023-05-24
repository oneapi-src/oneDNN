Batch Normalization {#dev_guide_batch_normalization}
====================================================

>
> [API Reference](@ref dnnl_api_batch_normalization)
>

## General

The batch normalization primitive performs a forward or backward batch
normalization operation on tensors with number of dimensions equal to 2 or more.

### Forward

The batch normalization operation is defined by the following formulas. We show
formulas only for 2D spatial data which are straightforward to generalize to
cases of higher and lower dimensions. Variable names follow the standard
@ref dev_guide_conventions.

\f[
    \dst(n, c, h, w) =
       \gamma(c) \cdot
       \frac{\src(n, c, h, w) - \mu(c)} {\sqrt{\sigma^2(c) + \varepsilon}}
       + \beta(c),
\f]

where

- \f$\gamma(c), \beta(c)\f$ are optional scale and shift for a channel
(see #dnnl_use_scale and #dnnl_use_shift flags),

- \f$\mu(c), \sigma^2(c)\f$ are mean and variance for a channel (see
  #dnnl_use_global_stats flag), and

- \f$\varepsilon\f$ is a constant to improve numerical stability.

Mean and variance are computed at runtime or provided by a user. When mean and
variance are computed at runtime, the following formulas are used:

- \f$\mu(c) = \frac{1}{NHW} \sum\limits_{nhw} \src(n, c, h, w)_{}\f$,

- \f$\sigma^2(c) = \frac{1}{NHW} \sum\limits_{nhw} {}_{} (\src(n, c, h, w) - \mu(c))^2\f$.

The \f$\gamma(c)\f$ and \f$\beta(c)\f$ tensors are considered learnable.

In training mode, the primitive also optionally supports:

* fusion with ReLU activation with zero negative slope applied to the result
  (see #dnnl_fuse_norm_relu flag).

* fusion with binary addition and ReLU activation with zero negative slope
  applied to the result (see #dnnl_fuse_norm_add_relu flag).

@note
* The batch normalization primitive computes population mean and variance and
  not the sample or unbiased versions that are typically used to compute
  running mean and variance.
* Using the mean and variance computed by the batch normalization primitive,
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

 * If batch normalization is created with ReLU fusion (i.e.,
   #dnnl_fuse_norm_relu or #dnnl_fuse_norm_add_relu are set), for the propagation
   kind #dnnl_forward_training the primitive would produce a `workspace`
   memory as one extra output. This memory is required to compute the backward
   propagation. When the primitive is executed with propagation kind
   #dnnl_forward_inference, the workspace is not produced. Behavior would
   be the same as creating a batch normalization primitive with ReLU as a
   post-op (see section below).

### Backward

The backward propagation computes
\f$\diffsrc(n, c, h, w)\f$,
\f$\diffgamma(c)^*\f$, and \f$\diffbeta(c)^*\f$
based on
\f$\diffdst(n, c, h, w)\f$, \f$\src(n, c, h, w)\f$, \f$\mu(c)\f$,
\f$\sigma^2(c)\f$, and \f$\gamma(c) ^*\f$.

The tensors marked with an asterisk are used only when the primitive is
configured to use \f$\gamma(c)\f$ and \f$\beta(c)\f$ (i.e.,
#dnnl_use_scale or #dnnl_use_shift are set).

## Execution Arguments

Depending on the [flags](@ref dnnl_normalization_flags_t) and
[propagation kind](@ref dnnl_prop_kind_t), the batch normalization primitive
requires different inputs and outputs.  For clarity, a summary is shown below.

| Flags                                                        | #dnnl_forward_inference                                                                                            | #dnnl_forward_training                                                                                                                                                                   | #dnnl_backward                                                                                                                                                                                | #dnnl_backward_data                                                                                              |
|:-------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------|
| #dnnl_normalization_flags_none                               | *Inputs*: \src <br><br> *Outputs*: \dst                                                                            | *Inputs*: \src <br><br> *Outputs*: \dst, \f$\mu\f$, \f$\sigma^2\f$                                                                                                                       | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \diffsrc                                                                                                              | Same as for #dnnl_backward                                                                                       |
| #dnnl_use_global_stats                                       | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \dst                                                 | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \dst                                                                                                                       | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$ <br><br> *Outputs*: \diffsrc                                                                                                              | Same as for #dnnl_backward                                                                                       |
| #dnnl_use_scale                                              | *Inputs*: \src, \f$\gamma\f$  <br><br> *Outputs*: \dst                                                             | *Inputs*: \src, \f$\gamma\f$ <br><br> *Outputs*: \dst, \f$\mu\f$, \f$\sigma^2\f$                                                                                                         | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$ <br><br> *Outputs*: \diffsrc, \f$\diffgamma\f$                                                                              | Not supported                                                                                                    |
| #dnnl_use_shift                                              | *Inputs*: \src, \f$\beta\f$ <br><br> *Outputs*: \dst                                                               | *Inputs*: \src, \f$\beta\f$ <br><br> *Outputs*: \dst, \f$\mu\f$, \f$\sigma^2\f$                                                                                                          | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\beta\f$ <br><br> *Outputs*: \diffsrc, \f$\diffbeta\f$                                                                                | Not supported                                                                                                    |
| #dnnl_use_global_stats \| #dnnl_use_scale \| #dnnl_use_shift | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \dst                      | *Inputs*: \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \dst                                                                                            | *Inputs*: \diffdst, \src, \f$\mu\f$, \f$\sigma^2\f$, \f$\gamma\f$, \f$\beta\f$ <br><br> *Outputs*: \diffsrc, \f$\diffgamma\f$, \f$\diffbeta\f$                                                | Not supported                                                                                                    |
| `flags` \| #dnnl_fuse_norm_relu                              | *Inputs*: same as with `flags` <br><br> *Outputs*: same as with `flags`                                            | *Inputs*: same as with `flags` <br><br> *Outputs*: same as with `flags`, [Workspace](@ref dev_guide_inference_and_training_aspects_workspace)                                            | *Inputs*: same as with `flags`, [Workspace](@ref dev_guide_inference_and_training_aspects_workspace) <br><br> *Outputs*: same as with `flags`                                                 | Same as for #dnnl_backward if `flags` do not contain #dnnl_use_scale or #dnnl_use_shift; not supported otherwise |
| `flags` \| #dnnl_fuse_norm_add_relu                          | *Inputs*: same as with `flags` and \f$\src_1\f$ for fused binary addition <br><br> *Outputs*: same as with `flags` | *Inputs*: same as with `flags` and \f$\src_1\f$ for fused binary addition <br><br> *Outputs*: same as with `flags`, [Workspace](@ref dev_guide_inference_and_training_aspects_workspace) | *Inputs*: same as with `flags`, [Workspace](@ref dev_guide_inference_and_training_aspects_workspace) <br><br> *Outputs*: same as with `flags` and \f$\diffsrc_1\f$ for fused binary addition | Same as for #dnnl_backward if `flags` do not contain #dnnl_use_scale or #dnnl_use_shift; not supported otherwise |

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive Input/Output    | Execution Argument Index |
|---------------------------|--------------------------|
| \src                      | DNNL_ARG_SRC             |
| \f$\src_1\f$              | DNNL_ARG_SRC_1           |
| \f$\gamma\f$              | DNNL_ARG_SCALE           |
| \f$\beta\f$               | DNNL_ARG_SHIFT           |
| mean (\f$\mu\f$)          | DNNL_ARG_MEAN            |
| variance (\f$\sigma^2\f$) | DNNL_ARG_VARIANCE        |
| \dst                      | DNNL_ARG_DST             |
| workspace                 | DNNL_ARG_WORKSPACE       |
| \diffdst                  | DNNL_ARG_DIFF_DST        |
| \diffsrc                  | DNNL_ARG_DIFF_SRC        |
| \f$\diffsrc_1\f$          | DNNL_ARG_DIFF_SRC_1      |
| \f$\diffgamma\f$          | DNNL_ARG_DIFF_SCALE      |
| \f$\diffbeta\f$           | DNNL_ARG_DIFF_SHIFT      |

## Implementation Details

### General Notes

1. The different flavors of the primitive are partially controlled by the @p
   flags parameter that is passed to the primitive descriptor creation
   function (e.g., dnnl::batch_normalization_forward::primitive_desc()).
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
   that backward propagation requires original \src, hence the corresponding
   forward propagation should not be performed in-place.

4. As mentioned above, the batch normalization primitive can be fused with
   binary addition and ReLU activation (#dnnl_fuse_norm_add_relu).
   In this case:
   - on the forward propagation the primitive has one additional input,
     \f$\src_1\f$, that should have memory descriptor equal to primitive
     `dst_desc` memory descriptor.
   - on the backward propagation the primitive has one additional output,
     \f$\diffsrc_1\f$, that should have memory descriptor equal to primitive
     `diff_dst_desc` memory descriptor.

5. As mentioned above, the batch normalization primitive can be fused with
   ReLU activation (#dnnl_fuse_norm_relu) or binary addition and ReLU activation
   (#dnnl_fuse_norm_add_relu) even in the training mode. In this case, on the
   forward propagation the primitive has one additional output, `workspace`,
   that should be passed during the backward propagation.

### Data Type Support

The operation supports the following combinations of data types:

| Propagation        | Source / Destination | Mean / Variance / ScaleShift |
|:-------------------|:---------------------|:-----------------------------|
| forward / backward | f32, bf16, f16       | f32                          |
| forward            | s8                   | f32                          |

@warning
    There might be hardware- or implementation-specific restrictions. Check the
    [Implementation Limitations](@ref dg_bnorm_impl_limits) section below.

### Data Representation

#### Mean and Variance

The mean (\f$\mu\f$) and variance (\f$\sigma^2\f$) are separate 1D tensors of
size \f$C\f$.

The format of the corresponding memory object must be #dnnl_x (#dnnl_a).

#### Scale and Shift

If #dnnl_use_scale or #dnnl_use_shift are used, the scale (\f$\gamma\f$) and
shift (\f$\beta\f$) are separate 1D tensors of shape \f$C\f$.


The format of the corresponding memory object must be #dnnl_nc (#dnnl_ab).

#### Source, Destination, and Their Gradients

Like other CNN primitives, the batch normalization primitive expects data
to be \f$N \times C \times SP_n \times \cdots \times SP_0\f$ tensor.

The batch normalization primitive is optimized for the following memory formats:

| Spatial | Logical tensor | Implementations optimized for memory formats                       |
|:--------|:---------------|:-------------------------------------------------------------------|
| 0D      | NC             | #dnnl_nc (#dnnl_ab)                                                |
| 1D      | NCW            | #dnnl_ncw (#dnnl_abc), #dnnl_nwc (#dnnl_acb), *optimized^*         |
| 2D      | NCHW           | #dnnl_nchw (#dnnl_abcd), #dnnl_nhwc (#dnnl_acdb), *optimized^*     |
| 3D      | NCDHW          | #dnnl_ncdhw (#dnnl_abcde), #dnnl_ndhwc (#dnnl_acdeb), *optimized^* |

Here *optimized^* means the format that
[comes out](@ref memory_format_propagation_cpp)
of any preceding compute-intensive primitive.

### Post-Ops and Attributes

Post-ops and attributes enable you to modify the behavior of the batch
normalization primitive by chaining certain operations after the batch
normalization operation. The following post-ops are supported by batch
normalization primitives:

| Propagation | Type    | Operation | Description                                                                                                         |
|:------------|:--------|:----------|:--------------------------------------------------------------------------------------------------------------------|
| forward     | post-op | eltwise   | Applies an @ref dnnl_api_eltwise operation to the result (currently only #dnnl_eltwise_relu algorithm is supported) |

@note As mentioned in @ref dev_guide_attributes, the post-ops should be used
for inference only. For instance, using ReLU as a post-op would not produce the
additional output `workspace` that is required to compute backward propagation
correctly. Hence, in case of training one should use the #dnnl_fuse_norm_relu or
#dnnl_fuse_norm_add_relu directly.

@anchor dg_bnorm_impl_limits
## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. For the data types that have forward propagation support only, mean and
   variance must be provided by a user (i.e., #dnnl_use_global_stats is set).

3. CPU implementations do not support the fusion with binary addition and ReLU
   activation (#dnnl_fuse_norm_add_relu).

## Performance Tips

1. For backward propagation, use the same memory format for \src, \diffdst,
   and \diffsrc (the format of the \diffdst and \diffsrc are always the
   same because of the API). Different formats are functionally supported but
   lead to highly suboptimal performance.

2. Use in-place operations whenever possible (see caveats in General Notes).

3. GPU implementations support an experimental algorithm with single pass
   statistics calculations. Please review
   [experimental features](@ref dev_guide_experimental) for more details.

## Examples

[Batch Normalization Primitive Example](@ref batch_normalization_example_cpp)

@copydetails batch_normalization_example_cpp_short
