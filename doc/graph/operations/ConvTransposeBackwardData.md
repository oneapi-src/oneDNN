ConvTransposeBackwardData {#dev_guide_op_convtransposebackwarddata}
===================================================================

## General

ConvTransposeBackwardData operation takes \f$\diffdst\f$ and \weights and
computes \f$\diffsrc\f$.

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

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_dst`    | Required             |
| 1     | `weights`     | Required             |
@note
The shape of \weights is
\f$(in\_channels / groups, out\_channels, spatial\_shape)\f$ for `IOX` format or
\f$(spatial\_shape, out\_channels, in\_channels / groups)\f$ for `XOI` format.
Both \f$in\_channels\f$ and \f$out\_channels\f$ must be divisible by *groups*
attribute.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |

## Supported data types

ConvTransposeBackwardData operation supports the following data type
combinations.

| Diff_dst | Weights | Diff_src |
|:---------|:--------|:---------|
| f32      | f32     | f32      |
| bf16     | bf16    | bf16     |
| f16      | f16     | f16      |
