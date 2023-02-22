Resampling {#dev_guide_resampling}
=====================================

>
> [API reference](@ref dnnl_api_resampling)
>

## General

The resampling primitive computes forward or backward resampling operation on
1D, 2D, or 3D spatial data. Resampling performs spatial scaling of original
tensor using one of the supported interpolation algorithms:
- Nearest Neighbor
- Linear (or Bilinear for 2D spatial tensor, Trilinear for 3D spatial tensor).

Resampling operation is defined by the source tensor and scaling factors in
each spatial dimension. Upsampling and downsampling are the alternative terms
for resampling that are used when all scaling factors are greater (upsampling)
or less (downsampling) than one.

The resampling operation is defined by the following formulas. We show formulas
only for 2D spatial data which are straightforward to generalize to cases of
higher and lower dimensions. Variable names follow the standard
@ref dev_guide_conventions.

Let \src and \dst be \f$N \times C \times IH \times IW\f$ and \f$N
\times C \times OH \times OW\f$ tensors respectively. Let
\f$ F_h = \frac{OH}{IH} \f$ and \f$ F_w = \frac{OW}{IW} \f$ define scaling
factors in each spatial dimension.

The following formulas show how oneDNN computes resampling for nearest neighbor
and bilinear interpolation methods.
To further simplify the formulas, we assume the following:
\f$\src(n, ic, ih, iw) = \begin{cases}
\src(n, ic, ih, 0), & \text{if}\ iw < 0 \\
\src(n, ic, ih, iw), & \text{if}\ IW - 1 \geq iw \geq 0 \\
\src(n, ic, ih, IW - 1), & \text{if}\ iw > IW - 1
\end{cases}\f$

Same assumptions apply for \f$ih\f$. Definitions of \f$ih\f$ and \f$iw\f$ are
provided below with a correspondent algorithm.

### Forward

#### Nearest Neighbor Resampling

\f[\dst(n, c, oh, ow) =  \src(n, c, ih, iw)\f]

where

- \f$ih = [\frac{oh + 0.5} {F_h} - 0.5]\f$,
- \f$iw = [\frac{ow + 0.5} {F_w} - 0.5]\f$.

#### Bilinear Resampling

\f[
    \dst(n, c, oh, ow) =
            \src(n, c, ih_0, iw_0) \cdot (1 - W_{ih}) \cdot (1 - W_{iw}) + \\
            \src(n, c, ih_1, iw_0) \cdot W_{ih} \cdot (1 - W_{iw}) + \\
            \src(n, c, ih_0, iw_1) \cdot (1 - W_{ih}) \cdot W_{iw} + \\
            \src(n, c, ih_1, iw_1) \cdot W_{ih} \cdot W_{iw} \\
\f]

where
- \f$ih_0 = \left\lfloor{\frac {oh + 0.5} {F_h} - 0.5}\right\rfloor\f$,
- \f$ih_1 = \left\lceil {\frac {oh + 0.5} {F_h} - 0.5}\right\rceil\f$,
- \f$iw_0 = \left\lfloor{\frac {ow + 0.5} {F_w} - 0.5}\right\rfloor\f$,
- \f$iw_1 = \left\lceil {\frac {ow + 0.5} {F_w} - 0.5}\right\rceil\f$,
- \f$W_{ih} = \frac{oh + 0.5}{F_h} - 0.5 - ih_0\f$,
- \f$W_{iw} = \frac{ow + 0.5}{F_w} - 0.5 - iw_0\f$.


#### Difference Between Forward Training and Forward Inference

There is no difference between the #dnnl_forward_training
and #dnnl_forward_inference propagation kinds.

### Backward

The backward propagation computes \diffsrc based on \diffdst.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output      | Execution argument index                                                  |
|-----------------------------|---------------------------------------------------------------------------|
| \src                        | DNNL_ARG_SRC                                                              |
| \dst                        | DNNL_ARG_DST                                                              |
| \diffsrc                    | DNNL_ARG_DIFF_SRC                                                         |
| \diffdst                    | DNNL_ARG_DIFF_DST                                                         |
| \f$\text{binary post-op}\f$ | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_1 |

## Implementation Details

### General Notes

1. Resampling implementation supports data with arbitrary data tag (nchw, nhwc,
   nChw16c, etc.) but memory tags for `src` and `dst` are expected to be the
   same. Resampling primitive supports `dst` and `diff_src` memory tag
   #dnnl::memory::format_tag::any and can define destination format based on
   source format.
2. Resampling primitive descriptor can be created by specifying the source and
   destination memory descriptors, only the source descriptor and floating
   point factors, or the source and destination memory descriptors and factors.
   In case when user does not provide the destination descriptor, the
   destination dimensions are deduced using the factors:
   \f$
     output\_spatial\_size = \left\lfloor{
        \frac{input\_spatial\_size} {F}
     }\right\rfloor
   \f$.

@note
    Implementation of resampling algorithm uses factors as defined by the
    relation \f$F = \frac{output\_spatial\_ size} {
    input\_spatial\_size}\f$ that do not necessarily equal to the ones passed
    by the user.


### Data Types

Resampling primitive supports the following combination of data types for
source and destination memory objects:

| Propagation        | Source                      | Destination                 |
|:-------------------|:----------------------------|:----------------------------|
| forward / backward | f32, bf16, f16, s32, s8, u8 | f32, s32, bf16, s8, u8, f16 |

### Post-Ops and Attributes

The following attributes are supported:

| Type    | Operation                                      | Description                                                                    | Restrictions                        |
|:--------|:-----------------------------------------------|:-------------------------------------------------------------------------------|:------------------------------------|
| Post-op | [Sum](@ref dnnl::post_ops::append_sum)         | Adds the operation result to the destination tensor instead of overwriting it. |                                     |
| Post-op | [Eltwise](@ref dnnl::post_ops::append_eltwise) | Applies an @ref dnnl_api_eltwise operation to the result.                      |                                     |
| Post-op | [Binary](@ref dnnl::post_ops::append_binary)   | Applies a @ref dnnl_api_binary operation to the result                         | General binary post-op restrictions |

## Implementation Limitations

1. No primitive specific limitations. Refer to @ref dev_guide_data_types for
   limitations related to data types support.

## Performance Tips

N/A

## Example

[Resampling Primitive Example](@ref resampling_example_cpp)

@copydetails resampling_example_cpp_short
