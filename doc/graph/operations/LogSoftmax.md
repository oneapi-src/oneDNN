LogSoftmax {#dev_guide_op_logsoftmax}
=====================================

## General

LogSoftmax operation applies the \f$ \log(softmax(src)) \f$ function to an 
n-dimensional input Tensor. The formulation can be simplified as:
\f[ dst_i = \log\Big( \frac{exp(src_i)}{\sum_{j}^{ } exp(src_j)} \Big) \f]

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
| 0     | `src`         | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

LogSoftmax operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| bf16 | bf16 |
| f16  | f16  |
