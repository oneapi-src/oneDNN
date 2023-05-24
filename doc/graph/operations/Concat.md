Concat {#dev_guide_op_concat}
=============================

## General

Concat operation concatenates \f$N\f$ tensors over `axis` (here designated
\f$C\f$) and is defined as (the variable names follow the standard
@ref dev_guide_conventions):

\f[
    \dst(\overline{ou}, c, \overline{in}) =
        \src_i(\overline{ou}, c', \overline{in}),
\f]

where \f$c = C_1 + .. + C_{i-1} {}_{} + c'\f$.

## Operation attributes

| Attribute Name                           | Description                                            | Value Type | Supported Values                                          | Required or Optional |
|:-----------------------------------------|:-------------------------------------------------------|:-----------|:----------------------------------------------------------|:---------------------|
| [axis](@ref dnnl::graph::op::attr::axis) | Specifies dimension along which concatenation happens. | s64        | A s64 value in the range of [-r, r-1] where r = rank(src) | Required             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src_i`       | Required             |

@note At least one input tensor is required. Data types and ranks of all input
tensors should match. The dimensions of all input tensors should be the same
except for the dimension specified by `axis` attribute.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

Concat operation supports the following data type combinations.

| Src_i | Dst  |
|:------|:-----|
| f32   | f32  |
| f16   | f16  |
| bf16  | bf16 |
