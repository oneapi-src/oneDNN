Multiply{#dev_guide_op_multiply}
================================

## General

Multiply operation performs element-wise multiply operation with two given tensors applying
multi-directional broadcast rules.

  \f[
    \dst(\overline{x}) =
        \src_0(\overline{x}) \times \src_1(\overline{x}),
\f]

## Operation attributes

| Attribute Name                                               | Description                                                | Value Type | Supported Values         | Required or Optional |
|:-------------------------------------------------------------|:-----------------------------------------------------------|:-----------|:-------------------------|:---------------------|
| [auto_broadcast](@ref dnnl::graph::op::attr::auto_broadcast) | Specifies rules used for auto-broadcasting of src tensors. |string      | `none`,`numpy` (default) | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src_0`       | Required             |
| 1     | `src_1`       | Required             |

@note Both src shapes should match and no auto-broadcasting is allowed if
`auto_broadcast` attributes is `none`. `src_0` and `src_1` shapes can be
different and auto-broadcasting is allowed if `auto_broadcast` attributes is
`numpy`. Broadcasting is performed according to auto_broadcast value.

### Outputs

| Index | Argument Name | Required or Optional |
|:------| --------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

Multiply operation supports the following data type combinations.

| Src_0 / Src_1 | Dst  |
|:--------------|:-----|
| f32           | f32  |
| bf16          | bf16 |
| f16           | f16  |
