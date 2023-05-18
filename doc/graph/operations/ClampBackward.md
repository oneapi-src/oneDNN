ClampBackward {#dev_guide_op_clampbackward}
===========================================

## General

ClampBackward operation computes gradient for Clamp.

## Operation attributes

| Attribute Name                                 | Description                                                                                                                       | Value Type | Supported Values          | Required or Optional |
|:-----------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|:-----------|:--------------------------|:---------------------|
| [min](@ref dnnl::graph::op::attr::min)         | The lower bound of values in the output. Any value in the input that is smaller than the bound, is replaced with the `min` value. | f32        | Arbitrary valid f32 value | Required             |
| [max](@ref dnnl::graph::op::attr::max)         | The upper bound of values in the output. Any value in the input that is greater than the bound, is replaced with the `max` value. | f32        | Arbitrary valid f32 value | Required             |
| [use_dst](@ref dnnl::graph::op::attr::use_dst) | If true, use `dst` of Clamp operation to calculate the gradient. Otherwise, use `src`.                                            | bool       | `true` (default), `false` | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src` / `dst` | Required             |
| 1     | `diff_dst`    | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_src`    | Required             |

## Supported data types

ClampBackward operation supports the following data type combinations.

| Src / Dst | Diff_dst | Diff_src |
|:----------|:---------|:---------|
| f32       | f32      | f32      |
| f16       | f16      | f16      |
| bf16      | bf16     | bf16     |
