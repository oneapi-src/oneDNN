MaxPoolBackward {#dev_guide_op_maxpoolbackward}
===============================================

## General

AvgPoolBackward operation accepts \src tensor and \f$\diffdst\f$ tensor, and
calculates \f$\diffsrc\f$ tensor.

## Operation attributes

| Attribute Name                                         | Description                                                                                                                                                                                       | Value Type | Supported Values                                                                  | Required or Optional |
|:-------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:----------------------------------------------------------------------------------|:---------------------|
| [strides](@ref dnnl::graph::op::attr::strides)         | Controls the strides the window is moved.                                                                                                                                                         | s64        | A s64 list containing positive values                                             | Required             |
| [pads_begin](@ref dnnl::graph::op::attr::pads_begin)   | Controls number of zeros to be add to the front/top/left of spatial dimensions, the attribute will be ignored when `auto_pad` attribute is specified to `same_upper`, `same_lower` or `valid`.    |s64 | A s64 list containing non-negative values                                                 | Required             |
| [pads_end](@ref dnnl::graph::op::attr::pads_end)       | Controls number of zeros to be add to the back/bottom/right of spatial dimensions, the attribute will be ignored when `auto_pad` attribute is specified to `same_upper`, `same_lower` or `valid`. | s64        | A s64 list containing non-negative values                                         | Required             |
| [kernel](@ref dnnl::graph::op::attr::kernel)           | Size of pooling window                                                                                                                                                                            | s64        | A s64 list containing positive values                                             | Required             |
| [auto_pad](@ref dnnl::graph::op::attr::auto_pad)       | Controls how the paddings are calculated.                                                                                                                                                         | string     | `none` (default), `same_upper`, `same_lower`, `valid`                             | Optional             |
| [dilations](@ref dnnl::graph::op::attr::dilations)     | Denotes the distance in width and height between elements in the window.                                                                                                                          | s64        | A s64 list containing positive values, a list of `1`s (default) means no dilation | Optional             |
| [data_format](@ref dnnl::graph::op::attr::data_format) | Controls how to interpret the shape of `src` and `dst`.                                                                                                                                           | string     | `NCX`, `NXC` (default)                                                            | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `diff_dst`    | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |

## Supported data types

MaxPoolBackward operation supports the following data type combinations.

| Src  | Diff_dst | Diff_src |
|:-----|:---------|:---------|
| f32  | f32      | f32      |
| bf16 | bf16     | bf16     |
| f16  | f16      | f16      |
