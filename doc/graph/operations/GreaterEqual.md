GreaterEqual{#dev_guide_op_greaterequal}
========================================

## General

The GreaterEqual operation performs an element-wise greater-than-or-equal
comparison between two given tensors. This operation applies
the multi-directional broadcast rules to ensure compatibility between
the tensors of different shapes.

\f[ dst = \begin{cases} true & \text{if}\ src_0 \ge src_1 \\
    false & \text{if}\ src_0 < src_1 \end{cases} \f]

## Operation Attributes

| Attribute Name                                               | Description                                                | Value Type | Supported Values         | Required or Optional |
|:-------------------------------------------------------------|:-----------------------------------------------------------|:-----------|:-------------------------|:---------------------|
| [auto_broadcast](@ref dnnl::graph::op::attr::auto_broadcast) | Specifies rules used for auto-broadcasting of src tensors. | string     | `none`,`numpy` (default) | Optional             |

## Execution Arguments

### Input

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src_0`       | Required             |
| 1     | `src_1`       | Required             |

@note Both src shapes should match and no auto-broadcasting is allowed if
the `auto_broadcast` attribute is `none`. `src_0` and `src_1` shapes can be
different and auto-broadcasting is allowed if the `auto_broadcast` attribute
is `numpy`. Broadcasting is performed according to the `auto_broadcast` value.

### Output

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported Data Types

The GreaterEqual operation supports the following data type combinations.

| Src_0 / Src_1 | Dst      |
|:--------------|:---------|
| f32           | boolean  |
| bf16          | boolean  |
| f16           | boolean  |
| s32           | boolean  |
