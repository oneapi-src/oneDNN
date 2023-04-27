Pow {#dev_guide_op_pow}
=======================


## General

Pow operation performs an element-wise power operation on a given input tensor
with a single value attribute beta as its exponent. It is based on the following
mathematical formula:

  \f[ dst_{i} = {src_{i} ^ \beta} \f]

## Operation attributes

| Attribute Name                             | Description                              | Value Type | Supported Values     | Required or Optional |
|:-------------------------------------------|:-----------------------------------------|:-----------|:---------------------|:---------------------|
| [beta](@ref dnnl::graph::op::attr::beta)   | exponent, \f$ \beta \f$ in the formula.  | f32        | Arbitrary f32 value. | Required             |

## Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |

## Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

Pow operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| bf16 | bf16 |
| f16  | f16  |
