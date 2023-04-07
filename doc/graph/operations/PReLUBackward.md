PReLUBackward {#dev_guide_op_prelubackward}
===========================================

## General

PReLUBackward operation computes gradient for PReLU.

## Operation attributes

| Attribute Name                                        | Description                                           | Value Type | Supported Values      | Required or Optional |
|:------------------------------------------------------|:------------------------------------------------------|:-----------|:----------------------|:---------------------|
|[data_format](@ref dnnl::graph::op::attr::data_format) | Denotes the data format of the input and output data. | string     | `NCX`, `NXC`(default) | Optional             |

### Broadcasting Rules

Only slope tensor supports broadcasting semantics. Slope tensor is
uni-directionally broadcasted to \src if one of the following rules is met:

1. PyTorch case: slope is 1D tensor and broadcast per channel, length of
  slope is equal to the length of \src in channel dimension.

2. PyTorch case: slope is 1D tensor and broadcast per tensor, length of slope
  is equal to 1.

3. Tensorflow case: slope is nD tensor and its dimensions must be equal to
  the \src dimensions starting from the second element:
  \f$ slope\_shape = input\_forward\_shape[1:] \f$

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `slope`       | Required             |
| 2     | `diff_dst`    | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |
| 1     | `diff_slope`  | Required             |

## Supported data types

PReLUBackward operation supports the following data type combinations.

| Src  | Slope | Diff_dst | Diff_src | Diff_slope |
|:-----|:------|:---------|:---------|:-----------|
| f32  | f32   | f32      | f32      | f32        |
| bf16 | bf16  | bf16     | bf16     | bf16       |
| f16  | f16   | f16      | f16      | f16        |
