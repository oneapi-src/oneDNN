# Convolution {#dev_guide_op_convolution}

[Convolution definition in oneDNN specification](https://spec.oneapi.io/onednn-graph/latest/ops/convolution/Convolution_1.html)

## General

Convolution op performs the cross correlation between src tensor and weight
tensor, which is defined as by the following formulas.

Let \src, \weights and \dst tensors have shape \f$N \times IC \times IH \times
IW\f$, \f$OC \times IC \times KH \times KW\f$, and \f$N \times OC \times OH
\times OW\f$ respectively.

Furthermore, let the remaining convolution parameters be:

| Parameter | Depth      | Height     | Width      | Comment
| :--| :--        | :--        | :--        |:--
| Paddings: Front, top, and left    | \f$PD_L\f$ | \f$PH_L\f$ | \f$PW_L\f$ | In the attributes we use `pads_begin` to indicate the corresponding vector of paddings |
| Padding: Back, bottom, and right | \f$PD_R\f$ | \f$PH_R\f$ | \f$PW_R\f$ | In the attributes we use `pads_end` to indicate the corresponding vector of paddings  |
| Stride                               | \f$SD\f$   | \f$SH\f$   | \f$SW\f$   | In the attributes we use `strides` to indicate the corresponding vector of strides |
| Dilation                             | \f$DD\f$   | \f$DH\f$   | \f$DW\f$   | In the attributes we use `dilations` to indicate the corresponding vector of dilations|

To further simplify the formulas, we assume that the attribute `data_format` and
`weights_format` are set to `NCX` and `OIX` respectively. `NCX` means the fist
axis represents batch dimension, the second axis represents channel dimension
and the rest represents spatial dimensions. `OIX` means the first axis
represents output channel dimension, the second axis represents input channel
dimension and the rest represents weights spatial dimensions.

### Regular Convolution

\f[\dst(n, oc, oh, ow) =  \bias(oc) +
       \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
        \src(n, ic, oh \cdot SH + kh - PH_L, ow \cdot SW + kw - PW_L)
        \cdot
        \weights(oc, ic, kh, kw).\f]

Here:

- \f$OH = \left\lfloor{\frac{IH - KH + PH_L + PH_R}{SH}} \right\rfloor + 1,\f$

- \f$OW = \left\lfloor{\frac{IW - KW + PW_L + PW_R}{SW}} \right\rfloor + 1.\f$

### Convolution with group

The attribute `groups` is set to \f$>1\f$.

\f[
    \dst(n, g \cdot OC_G + oc_g, oh, ow) =
        \bias(g \cdot OC_G + oc_g) \\
        +
        \sum_{ic_g=0}^{IC_G-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
            \src(n, g \cdot IC_G + ic_g, oh \cdot SH + kh - PH_L,
                    ow \cdot SW + kw - PW_L)
            \cdot
            \weights(g, oc_g, ic_g, kh, kw),
\f]

where

- \f$IC_G = \frac{IC}{G}\f$,

- \f$OC_G = \frac{OC}{G}\f$, and

- \f$oc_g \in [0, OC_G).\f$

### Convolution with dilation

The attribute `dilation` contains the element which is \f$>1\f$.

\f[
    \dst(n, oc, oh, ow) =
        \bias(oc) \\
        +
        \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
            \src(n, ic, oh \cdot SH + kh \cdot DH - PH_L,
                    ow \cdot SW + kw \cdot DW - PW_L)
            \cdot
            \weights(oc, ic, kh, kw).
\f]

Here:

- \f$OH = \left\lfloor{\frac{IH - DKH + PH_L + PH_R}{SH}}
        \right\rfloor + 1,\f$ where \f$DKH = 1 + (KH - 1) \cdot DH\f$, and

- \f$OW = \left\lfloor{\frac{IW - DKW + PW_L + PW_R}{SW}}
        \right\rfloor + 1,\f$ where \f$DKW = 1 + (KW - 1) \cdot DW\f$.

## Operation Attributes

Attribute Name | Description | Value Type |Supported Values | Required or Optional
-- | -- | --| --|--
[strides](@ref dnnl::graph::op::attr::strides) | controls the strides the filter tensor is moved when computing cross-correlation |s64 |a s64 list containing positive values  | Required
[pads_begin](@ref dnnl::graph::op::attr::pads_begin) | controls number of zeros to be add to the front/top/left of spatial dimensions|s64 | a s64 list containing non-negative values  | Required
[pads_end](@ref dnnl::graph::op::attr::pads_end) | controls number of zeros to be add to the back/bottom/right of spatial dimensions |s64 |a s64 list containing non-negative values | Required
[dilations](@ref dnnl::graph::op::attr::dilations) | controls the amount of stretching the kernel before convolving ([visualization link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#dilated-convolution-animations)) | s64| a s64 list containing positive values (>1 means dilated convolution) | Required
[auto_pad](@ref dnnl::graph::op::attr::auto_pad)| controls how the padding is calculated|string | `none` (default), `same_upper`, `same_lower`, `valid` | Optional
[groups](@ref dnnl::graph::op::attr::groups) | controls how input channels and output channels are divided into |s64 |a positive s64 value, `1` by default | Optional
[data_format](@ref dnnl::graph::op::attr::data_format) |controls how to interpret the shape of `src` and `dst`.| string|`NCX`, `NXC` (default) | Optional
[weights_format](@ref dnnl::graph::op::attr::filter_format) |controls how to interpret the shape of `weights`| string|`OIX`, `XIO` (default) | Optional

## Execution Arguments

Argument Name | Description | Required or Optional
-- | -- | --
`src` | input tensor | Required
`weights` | weights tensor | Required
`bias` |a 1-D tensor adds to channel dimension of src|Optional
`dst` |output tensor|Required

@note
The shape of weights is
\f$(out\_channels, in\_channels / groups, spatial\_shape)\f$ for `OIX` format or
\f$(spatial\_shape, in\_channels / groups, out\_channels)\f$ for `XIO` format.
Both \f$in\_channels\f$ and \f$out\_channels\f$ must be divisible by *groups*
attribute.

## Supported Data Types

Convolution op supports the following data types.

Src | Weights | Bias | Dst
--|--|-- | --
f32 | f32 | f32 |f32
bf16 | bf16 | bf16 |bf16
f16 | f16 | f16 |f16
