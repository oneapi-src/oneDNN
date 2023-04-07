Clamp {#dev_guide_op_clamp}
===========================

## General
 
Clamp operation represents clipping activation function, it applies following 
formula on every element of \src tensor (the variable names follow the standard 
@ref dev_guide_conventions):

\f[ clamp(src_i) = min(max(src_i, min\_value), max\_value) \f]

## Operation attributes

| Attribute Name                         | Description                                                                                                                       | Value Type | Supported Values          | Required or Optional |
|:---------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|:-----------|:--------------------------|:---------------------|
| [min](@ref dnnl::graph::op::attr::min) | The lower bound of values in the output. Any value in the input that is smaller than the bound, is replaced with the `min` value. | f32        | Arbitrary valid f32 value | Required             |
| [max](@ref dnnl::graph::op::attr::max) | The upper bound of values in the output. Any value in the input that is greater than the bound, is replaced with the `max` value. | f32        | Arbitrary valid f32 value | Required             |

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

Clamp operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| f16  | f16  |
| bf16 | bf16 |
