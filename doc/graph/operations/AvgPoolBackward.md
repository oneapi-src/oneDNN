AvgPoolBackward {#dev_guide_op_avgpoolbackward}
===============================================

## General

AvgPoolBackward operation accepts \f$\diffdst\f$ tensor and \f$\srcshape\f$
tensor (optional), and calculates \f$\diffsrc\f$ tensor.

## Operation attributes

| Attribute Name                                         | Description                                                                                                                                                                                       | Value Type |Supported Values                                       | Required or Optional |
|:-------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------|:------------------------------------------------------|:---------------------|
| [strides](@ref dnnl::graph::op::attr::strides)         | Controls the strides the window is moved.                                                                                                                                                         | s64        | A s64 list containing positive values                 | Required             |
| [pads_begin](@ref dnnl::graph::op::attr::pads_begin)   | Controls number of zeros to be add to the front/top/left of spatial dimensions, the attribute will be ignored when `auto_pad` attribute is specified to `same_upper`, `same_lower` or `valid`.    | s64        | A s64 list containing non-negative values             | Required             |
| [pads_end](@ref dnnl::graph::op::attr::pads_end)       | Controls number of zeros to be add to the back/bottom/right of spatial dimensions, the attribute will be ignored when `auto_pad` attribute is specified to `same_upper`, `same_lower` or `valid`. | s64        | A s64 list containing non-negative values             | Required             |
| [kernel](@ref dnnl::graph::op::attr::kernel)           | Size of pooling window.                                                                                                                                                                           | s64        | A s64 list containing positive values                 | Required             |
| [exclude_pad](@ref dnnl::graph::op::attr::exclude_pad) | Controls whether the padded values are counted.                                                                                                                                                   | bool       | True, False                                           | Required             |
| [auto_pad](@ref dnnl::graph::op::attr::auto_pad)       | Controls how the paddings are calculated.                                                                                                                                                         | string     | `none` (default), `same_upper`, `same_lower`, `valid` | Optional             |
| [data_format](@ref dnnl::graph::op::attr::data_format) | Controls how to interpret the shape of `src` and `dst`.                                                                                                                                           | string     | `NCX`, `NXC` (default)                                | Optional             |
| [src_shape](@ref dnnl::graph::op::attr::src_shape)     | Denotes the shape of input of forward op.                                                                                                                                                         | s64        | A s64 list containing positive values.                | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_dst`    | Required             |
| 1     | `src_shape`   | Optional             |

@note Either `src_shape` input or `src_shape` attribute should be provided. If
both provided, `src_shape` input will precede over `src_shape` attribute.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |

## Supported data types

AvgPoolBackward operation supports the following data type combinations.

| Diff_dst | Diff_src | Src_shape |
|:---------|:---------|:----------|
| f32      | f32      | s32       |
| bf16     | bf16     | s32       |
| f16      | f16      | s32       |
