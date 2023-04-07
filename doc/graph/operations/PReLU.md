PReLU {#dev_guide_op_prelu}
===========================

## General

PReLU operation performs element-wise parametric ReLU operation on a given
input tensor, based on the following mathematical formula:

\f[ dst = \begin{cases} src & \text{if}\ src \ge 0 \\
    \alpha src & \text{if}\ src < 0 \end{cases} \f]

## Operation attributes

| Attribute Name                                                             | Description                                                             | Value Type | Supported Values         | Required or Optional |
|:---------------------------------------------------------------------------|:------------------------------------------------------------------------|:-----------|:-------------------------|:---------------------|
| [data_format](@ref dnnl::graph::op::attr::data_format)                     | Denotes the data format of the input and output data.                   | string     | `NCX`, `NXC`(default)    | Optional             |
| [per_channel_broadcast](@ref dnnl::graph::op::attr::per_channel_broadcast) | Denotes whether to apply per_channel broadcast when slope is 1D tensor. | bool       | `false`, `true`(default) | Optional             |

### Broadcasting Rules

Only slope tensor supports broadcasting semantics. Slope tensor is
uni-directionally broadcasted to \src if one of the following rules is met:

- 1: slope is 1D tensor and `per_channel_broadcast` is set to `true`, the
  length of slope tensor is equal to the length of \src of channel dimension.

- 2: slope is 1D tensor and `per_channel_broadcast` is set to `false`, the
  length of slope tensor is equal to the length of \src of the rightmost
  dimension.

- 3: slope is nD tensor, starting from the rightmost dimension,
  \f$input.shape_i == slope.shape_i\f$ or \f$slope.shape_i == 1\f$ or
  slope dimension i is empty.

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `slope`       | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

PReLU operation supports the following data type combinations.

| Src  | Dst  | Slope |
|:-----|:-----|:------|
| f32  | f32  | f32   |
| bf16 | bf16 | bf16  |
| f16  | f16  | f16   |
