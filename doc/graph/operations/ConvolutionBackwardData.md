ConvolutionBackwardData {#dev_guide_op_convolutionbackwarddata}
===============================================================

## General

ConvolutionBackwardData operation accepts \f$\diffdst\f$, \weights and optional
dst shape as inputs, and compute the \f$\diffsrc\f$.

If `auto_pad` attribute is specified to one of `valid`, `same_upper` and
`same_lower`, `pads_begin` and `pads_end` attributes will be ignored. The
paddings will be calculated by following the below formula:

Let the parameters be:

| Parameter                        | Depth      | Height     | Width      | Comment                                                                                |
|:---------------------------------|:-----------|:-----------|:-----------|:---------------------------------------------------------------------------------------|
| Paddings: Front, top, and left   | \f$PD_L\f$ | \f$PH_L\f$ | \f$PW_L\f$ | In the attributes we use `pads_begin` to indicate the corresponding vector of paddings |
| Padding: Back, bottom, and right | \f$PD_R\f$ | \f$PH_R\f$ | \f$PW_R\f$ | In the attributes we use `pads_end` to indicate the corresponding vector of paddings   |
| Stride                           | \f$SD\f$   | \f$SH\f$   | \f$SW\f$   | In the attributes we use `strides` to indicate the corresponding vector of strides     |
| Dilation                         | \f$DD\f$   | \f$DH\f$   | \f$DW\f$   | In the attributes we use `dilations` to indicate the corresponding vector of dilations |

Firstly, \f$total\_padding\f$ is calculated according to \f$src\_shape\f$ and \f$dst\_shape\f$.
Let \f$src\_h\f$ be height dimension of \f$src\_shape\f$ and \f$dst\_h\f$ be
height dimension of \f$dst\_shape\f$.

\f[
    total\_padding_h = SH \times (src\_h - 1) + ((KH -1 ) \times DH + 1) - dst\_h + output\_padding_h
\f]

If `auto_pad` attribute is specified as `valid`:

\f[
  PD_L = 0 \\
  PD_R = 0
\f]

If `auto_pad` attribute is specified as `same_lower`:

\f[
  PD_L = floor(total\_padding / 2) \\
  PD_R = total\_padding - PD_L
\f]

If `auto_pad` attribute is specified as `same_upper`:

\f[
  PD_L = total\_padding - PD_R \\
  PD_R = floor(total\_padding / 2)
\f]

where:

- \f$dst\_shape\f$ is either an attribute or an input tensor,

- \f$output\_padding\f$ is an optional attribute.

## Operation attributes

| Attribute Name                                               | Description                                                                                                                                                                               | Value Type | Supported Values                                                     | Required or Optional |
|:-------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:---------------------------------------------------------------------|:---------------------|
| [strides](@ref dnnl::graph::op::attr::strides)               | Controls the strides the weights tensor is moved when computing convolution                                                                                                               | s64        | A s64 list containing positive values                                | Required             |
| [pads_begin](@ref dnnl::graph::op::attr::pads_begin)         | Controls number of zeros to be add to the front/top/left of spatial dimensions                                                                                                            | s64        | A s64 list containing non-negative values                            | Required             |
| [pads_end](@ref dnnl::graph::op::attr::pads_end)             | Controls number of zeros to be add to the back/bottom/right of spatial dimensions                                                                                                         | s64        | A s64 list containing non-negative values                            | Required             |
| [dilations](@ref dnnl::graph::op::attr::dilations)           | Controls the amount of stretching the kernel before convolution ([visualization link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#dilated-convolution-animations)) | s64        | A s64 list containing positive values (>1 means dilated convolution) | Required             |
| [auto_pad](@ref dnnl::graph::op::attr::auto_pad)             | Controls how the padding is calculated                                                                                                                                                    | string     | `none` (default), `same_upper`, `same_lower`, `valid`                | Optional             |
| [output_padding](@ref dnnl::graph::op::attr::output_padding) | Adds additional amount of padding per each spatial axis in `dst`.                                                                                                                         | s64        | A s64 list containing non-negative values, all zeros by default      | Optional             |
| [groups](@ref dnnl::graph::op::attr::groups)                 | Controls how input channels and output channels are divided into                                                                                                                          | s64        | A positive s64 value, `1` by default                                 | Optional             |
| [data_format](@ref dnnl::graph::op::attr::data_format)       | Controls how to interpret the shape of `src` and `dst`.                                                                                                                                   | string     | `NCX`, `NXC` (default)                                               | Optional             |
| [weights_format](@ref dnnl::graph::op::attr::weights_format) | Controls how to interpret the shape of `weights`.                                                                                                                                         | string     | `OIX`, `XIO` (default)                                               | Optional             |
| [dst_shape](@ref dnnl::graph::op::attr::dst_shape)           | Denotes the shape of the `dst` tensor.                                                                                                                                                    | s64        | A s64 list containing positive values                                | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0   | `diff_dst`      | Required             |
| 1   | `weights`       | Required             |
| 2   | `dst_shape`     | Optional             |

@note
The shape of \weights is
\f$(out\_channels, in\_channels / groups, spatial\_shape)\f$ for `OIX` format or
\f$(spatial\_shape, in\_channels / groups, out\_channels)\f$ for `XIO` format.
Both \f$in\_channels\f$ and \f$out\_channels\f$ must be divisible by *groups*
attribute.

@note Either `dst_shape` input or `dst_shape` attribute should be provided. If
both provided, `dst_shape` input will precede over `dst_shape` attribute.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |

## Supported data types

ConvolutionBackwardData operation supports the following data type combinations.

| Diff_dst | Weights | Diff_src | Dst_shape |
|:---------|:--------|:---------|:----------|
| f32      | f32     | f32      | s32       |
| bf16     | bf16    | bf16     | s32       |
| f16      | f16     | f16      | s32       |
