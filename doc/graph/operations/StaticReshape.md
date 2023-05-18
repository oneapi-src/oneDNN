StaticReshape {#dev_guide_op_staticreshape}
===========================================

## General

StaticReshape operation changes dimensions of \src tensor according to the
specified shape. The volume of \src is equal to \dst, where volume is the
product of dimensions. \dst may have a different memory layout from \src.
StaticReshape operation is not guaranteed to return a view or a copy of \src
when \dst is in-placed with the \src. StaticReshape can be used where if shape
is stored in a constant node or available during graph building stage. Then
shape can be passed via `shape` attribute.

## Operation attributes

| Attribute Name                                           | Description                                                | Value Type | Supported Values           | Required or Optional |
|:---------------------------------------------------------|:-----------------------------------------------------------|:-----------|:---------------------------|:---------------------|
| [shape](@ref dnnl::graph::op::attr::shape)               | Specifies rules used for auto-broadcasting of src tensors. | string     | `none`, `numpy` (default)  | Required             |
| [special_zero](@ref dnnl::graph::op::attr::special_zero) | Controls how zero values in shape are interpreted.         | bool       | `true`, `false` | Required                        |

@note `shape`: dimension `-1` means that this dimension is calculated to keep
the same overall elements count as the src tensor. That case that more than
one `-1` in the shape is not supported.

@note `special_zero`: if false, `0` in the shape is interpreted as-is (for
example a zero-dimension tensor); if true, then all `0`s in shape implies the
copying of corresponding dimensions from src into dst.

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:-------------------- |
| 0     | `dst`         | Required             |

## Supported data types

StaticReshape operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| bf16 | bf16 |
| f16  | f16  |
