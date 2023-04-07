ConvolutionBackwardWeights {#dev_guide_op_convolutionbackwardweights}
=====================================================================

## General

ConvolutionBackwardWeights operation accepts \src, \f$\diffdst\f$ and optional
weights shape as inputs, and compute the \f$\diffweights\f$.

## Operation attributes

| Attribute Name                                               | Description                                                                                                                                                                               | Value Type | Supported Values                                                     | Required or Optional |
|:-------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:---------------------------------------------------------------------|:---------------------|
| [strides](@ref dnnl::graph::op::attr::strides)               | Controls the strides the weights tensor is moved when computing convolution                                                                                                               | s64        | A s64 list containing positive values                                | Required             |
| [pads_begin](@ref dnnl::graph::op::attr::pads_begin)         | Controls number of zeros to be add to the front/top/left of spatial dimensions                                                                                                            | s64        | A s64 list containing non-negative values                            | Required             |
| [pads_end](@ref dnnl::graph::op::attr::pads_end)             | Controls number of zeros to be add to the back/bottom/right of spatial dimensions                                                                                                         | s64        | A s64 list containing non-negative values                            | Required             |
| [dilations](@ref dnnl::graph::op::attr::dilations)           | Controls the amount of stretching the kernel before convolution ([visualization link](https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#dilated-convolution-animations)) | s64        | A s64 list containing positive values (>1 means dilated convolution) | Required             |
| [auto_pad](@ref dnnl::graph::op::attr::auto_pad)             | Controls how the padding is calculated                                                                                                                                                    | string     | `none` (default), `same_upper`, `same_lower`, `valid`                | Optional             |
| [groups](@ref dnnl::graph::op::attr::groups)                 | Controls how input channels and output channels are divided into                                                                                                                          | s64        | A positive s64 value, `1` by default                                 | Optional             |
| [data_format](@ref dnnl::graph::op::attr::data_format)       | Controls how to interpret the shape of `src` and `dst`.                                                                                                                                   | string     | `NCX`, `NXC` (default)                                               | Optional             |
| [weights_format](@ref dnnl::graph::op::attr::weights_format) | Controls how to interpret the shape of `weights`.                                                                                                                                         | string     | `OIX`, `XIO` (default)                                               | Optional             |
| [weights_shape](@ref dnnl::graph::op::attr::weights_shape)   | Denotes the shape of the `weights` tensor.                                                                                                                                                | s64        | A s64 list containing positive values                                | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name   | Required or Optional |
|:------|:----------------|:---------------------|
| 0     | `src`           | Required             |
| 1     | `diff_dst`      | Required             |
| 2     | `weights_shape` | Optional             |

@note
The shape of \weights is
\f$(out\_channels, in\_channels / groups, spatial\_shape)\f$ for `OIX` format or
\f$(spatial\_shape, in\_channels / groups, out\_channels)\f$ for `XIO` format.
Both \f$in\_channels\f$ and \f$out\_channels\f$ must be divisible by *groups*
attribute.

**Note** Either `weights_shape` input or `weights_shape` attribute should be
provided. If both provided, `weights_shape` input will precede over
`weights_shape` attribute.

### Outputs

| Index | Argument Name  | Required or Optional |
|:------|:---------------|:---------------------|
| 0     | `diff_weights` | Required             |

## Supported data types

ConvolutionBackwardWeights operation supports the following data type
combinations.

| Src  | Diff_dst | Diff_weights | Weights_shape |
|:-----|:---------|:-------------|:--------------|
| f32  | f32      | f32          | s32           |
| bf16 | bf16     | bf16         | s32           |
| f16  | f16      | f16          | s32           |
