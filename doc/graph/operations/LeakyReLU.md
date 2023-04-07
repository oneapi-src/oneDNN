LeakyReLU {#dev_guide_op_leakyrelu}
===================================

## General

LeakyReLU operation is a type of activation function based on ReLU. It has a
small slope for negative values with which LeakyReLU can produce small,
non-zero, and constant gradients with respect to the negative values. The slope
is also called the coefficient of leakage.

Unlike @ref dev_guide_op_prelu, the coefficient \f$\alpha\f$ is constant and
defined before training.

LeakyReLU operation applies following formula on every element of \src tensor
(the variable names follow the standard @ref dev_guide_conventions):

\f[ dst = \begin{cases} src & \text{if}\ src \ge 0 \\
    \alpha src & \text{if}\ src < 0 \end{cases} \f]

## Operation attributes

| Attribute Name                             | Description                          | Value Type | Supported Values                                        | Required or Optional |
|:-------------------------------------------|:-------------------------------------|:-----------|:--------------------------------------------------------|:---------------------|
| [alpha](@ref dnnl::graph::op::attr::alpha) | Alpha is the coefficient of leakage. | f32        | Arbitrary f32 value but usually a small positive value. | Required             |

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

LeakyReLU operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| bf16 | bf16 |
| f16  | f16  |
