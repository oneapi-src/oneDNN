Elu {#dev_guide_op_elu}
=======================

## General

Elu operation applies following formula on every element of \src tensor (the
variable names follow the standard @ref dev_guide_conventions):

\f[ dst = \begin{cases} \alpha(e^{src} - 1) & \text{if}\ src < 0 \\
    src & \text{if}\ src \ge 0 \end{cases} \f]

## Operation attributes

| Attribute Name                             | Description                    | Value Type | Supported Values                 | Required or Optional |
|:-------------------------------------------|:-------------------------------|:-----------|:---------------------------------|:---------------------|
| [alpha](@ref dnnl::graph::op::attr::alpha) | Scale for the negative factor. | f32        | Arbitrary non-negative f32 value | Required             |

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

Elu operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| f16  | f16  |
| bf16 | bf16 |
