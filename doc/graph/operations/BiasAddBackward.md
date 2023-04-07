BiasAddBackward {#dev_guide_op_biasaddbackward}
===============================================

## General

BiasAddBackward operation computes the gradients on the bias tensor for
BiasAdd operator. This op accumulates all the values from \f$\diffdst\f$ into
the channel dimension, the axis depends on the layout of \src tensor.

## Operation attributes

| Attribute Name                                         | Description                                                        | Value Type | Supported Values         | Required or Optional |
|:-------------------------------------------------------|:-------------------------------------------------------------------|:-----------|:-------------------------|:---------------------|
| [data_format](@ref dnnl::graph::op::attr::data_format) | Controls how to interpret the shape of `diff_dst` and `diff_bias`. | string     | `NCX` , `NXC` (default)  | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_dst`    | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `diff_bias`   | Required             |

## Supported data types

BiasAddBackward operation supports the following data type combinations.

| Diff_dst | Diff_bias |
|:---------|:----------|
| f32      | f32       |
| bf16     | bf16      |
| f16      | f16       |
