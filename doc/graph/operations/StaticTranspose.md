StaticTranspose {#dev_guide_op_statictranspose}
===============================================

## General

StaticTranspose operation rearranges the dimensions of \src. \dst may have a
different memory layout from \src. StaticTranspose operation is not guaranteed
to return a view or a copy of \src when \dst is in-placed with the \src.

\f[

dst[src(order[0]), src(order[1]),\cdots, src(order[N-1])]\ =src[src(0), src(1),\cdots, src(N-1)]

\f]

## Operation attributes

| Attribute Name                             | Description                                   | Value Type | Supported Values                                                                                          | Required or Optional |
|:-------------------------------------------|:----------------------------------------------|:-----------|:----------------------------------------------------------------------------------------------------------|:---------------------|
| [order](@ref dnnl::graph::op::attr::order) | Specifies permutation to be applied on `src`. | s64        | A s64 list containing the element in the range of [-N, N-1], negative value means counting from last axis | Required             |

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

StaticTranspose operation supports the following data type combinations.

| Src  | Dst  |
|:-----|:-----|
| f32  | f32  |
| bf16 | bf16 |
| f16  | f16  |
