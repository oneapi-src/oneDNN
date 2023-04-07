SoftPlus {#dev_guide_op_softplus}
=================================

## General

SoftPlus operation applies following formula on every element of \src tensor 
(the variable names follow the standard @ref dev_guide_conventions):

\f[ dst = 1 / beta * \ln(e^{beta*src} + 1.0) \f]

## Operation attributes

| Attribute Name                          | Description                         | Value Type | Supported Values ----------------------| Required or Optional |
|:----------------------------------------|:------------------------------------|:-----------|:---------------------------------------|:---------------------|
|[beta](@ref dnnl::graph::op::attr::beta) | Value for the SoftPlus formulation. | f32        | Arbitrary f32 value (`1.f` by default) | Optional             |

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

SoftPlus operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| bf16 | bf16 |
| f16  | f16  |
