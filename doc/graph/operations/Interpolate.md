Interpolate {#dev_guide_op_interpolate}
=======================================

## General

Interpolate layer performs interpolation on \src tensor at spatial dimensions.

## Operation attributes

| Attribute Name                                                                              | Description                                                                                               | Value Type | Supported Values                                         | Required or Optional |
|:--------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------|:-----------|:---------------------------------------------------------|:---------------------|
|[mode](@ref dnnl::graph::op::attr::mode)                                                     | Specifies type of interpolation.                                                                          | string     | `nearest`, `linear`, `bilinear`, `trilinear`             | Required             |
|[coordinate_transformation_mode](@ref dnnl::graph::op::attr::coordinate_transformation_mode) | Specifies how to transform the coordinate in the resized tensor to the coordinate in the original tensor. | string     | `half_pixel`(default), `align_corners`                   | Optional             |
|[sizes](@ref dnnl::graph::op::attr::sizes)                                                   | Specifies dst shape for spatial axes.                                                                     | s64        | A s64 list containing positive values, `none` is default | Optional             |
|[scales](@ref dnnl::graph::op::attr::scales)                                                 | Specifies `scales` for spatial axes.                                                                      | f32        | A f32 list, `none` is default                            | Optional             |
|[data_format](@ref dnnl::graph::op::attr::data_format)                                       | Controls how to interpret the shape of `src` and `dst`.                                                   | string     | `NCX`, `NXC` (default)                                   | Optional             |

@note Either `sizes` or `scales` should be provided. When `sizes` is
used, `scales` will be ignored.

@note
    The attribute `coordinate_transformation_mode` is the name of transformation
    mode in string format.\n
    Here `scale[x]` is `dst_shape[x]/src_shape[x]` and `x_resized` is a
    coordinate in axis `x`,for any axis `x` from the src axis.\n
    For `half_pixel`: the coordinate in the original tensor axis `x` is
    calculated as `((x_resized + 0.5) / scale[x]) - 0.5`.\n
    For `align_corners`: the coordinate in the original tensor axis `x` is
    calculated as 0 if `dst_shape[x] == 1` else  `x_resized * (src_shape[x] - 1)
    / (dst_shape[x] - 1)`.

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `sizes`       | Optional             |

@note `sizes` is a 1D tensor describing output shape for spatial axes. It is a
non-differentiable tensor.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

@note The shape of the dst matches src
shape except spatial dimensions. For spatial dimensions, they should match sizes
from sizes or calculated from `scales` attribute.

## Supported data types

Interpolate operation supports the following data type combinations.

| Src  | Dst  | Sizes |
|:-----|:-----|:------|
| f32  | f32  | s32   |
| bf16 | bf16 | s32   |
| f16  | f16  | s32   |
