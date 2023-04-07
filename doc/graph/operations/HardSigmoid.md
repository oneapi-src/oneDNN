HardSigmoid {#dev_guide_op_hardsigmoid}
=======================================

## General

HardSigmoid operation applies the following formula on every element of \src
tensor (the variable names follow the standard @ref dev_guide_conventions):

\f[ dst = \text{max}(0, \text{min}(1, \alpha src + \beta)) \f]

## Operation attributes

| Attribute Name                             | Description                    | Value Type | Supported Values     | Required or Optional |
|:-------------------------------------------|:-------------------------------|:-----------|:---------------------|:---------------------|
| [alpha](@ref dnnl::graph::op::attr::alpha) | \f$ \alpha \f$ in the formula. | f32        | Arbitrary f32 value. | Required             |
| [beta](@ref dnnl::graph::op::attr::beta)   | \f$ \beta \f$ in the formula.  | f32        | Arbitrary f32 value. | Required             |

## Execution arguments

The inputs and outputs must be provided according to the index order shown below
when constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

HardSigmoid operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| bf16 | bf16 |
| f16  | f16  |
