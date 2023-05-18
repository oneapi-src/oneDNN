BiasAdd {#dev_guide_op_biasadd}
===============================

## General

Add bias to channel dimension of input. This is a special `Add` with bias
restricted to be 1-D. Broadcasting is supported.

  \f[ \dst(n,c,h,w) = \src(n,c,h,w) + \bias(c) \f]

## Operation attributes

| Attribute Name                                         | Description                                             | Value Type | Supported Values         | Required or Optional |
|:-------------------------------------------------------|:--------------------------------------------------------|:-----------|:-------------------------|:---------------------|
| [data_format](@ref dnnl::graph::op::attr::data_format) | Controls how to interpret the shape of `src` and `dst`. | string     | `NCX` , `NXC` (default)  | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `bias`        | Required             |

@note `bias` is a 1D tensor to be added to `src` tensor. The size should be the
same as size of channel dimension of `src` tensor.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

BiasAdd operation supports the following data type combinations.

| Src  | Bias | Dst  |
|:-----|:-----|:-----|
| f32  | f32  | f32  |
| bf16 | bf16 | bf16 |
| f16  | f16  | f16  |
