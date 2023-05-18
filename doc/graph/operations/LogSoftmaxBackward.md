LogSoftmaxBackward {#dev_guide_op_logsoftmaxbackward}
=====================================================

## General

LogSoftmaxBackward operation computes gradient for LogSoftmax.

## Operation attributes

| Attribute Name                           | Description                                                                                                        | Value Type | Supported Values                      | Required or Optional |
|:-----------------------------------------|:-------------------------------------------------------------------------------------------------------------------|:-----------|:--------------------------------------|:---------------------|
| [axis](@ref dnnl::graph::op::attr::axis) | Represents the axis of which the LogSoftmax is calculated. Negative value means counting dimensions from the back. | s64        | Arbitrary s64 value (`-1` in default) | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_dst`    | Required             |
| 1     | `dst`         | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |

## Supported data types

LogSoftmaxBackward operation supports the following data type combinations.

| Diff_dst | Dst  | Diff_src |
|:---------|:-----|:---------|
| f32      | f32  | f32      |
| bf16     | bf16 | bf16     |
| f16      | f16  | f16      |
