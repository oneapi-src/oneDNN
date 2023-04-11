DynamicTranspose {#dev_guide_op_dynamictranspose}
=================================================

## General

DynamicTranspose operation rearranges the dimensions of \src. \dst may have a
different memory layout from \src. DynamicTranspose operation is not guaranteed
to return a view or a copy of \src when \dst is in-placed with the \src.

\f[

dst[src(order[0]), src(order[1]),\cdots, src(order[N-1])]\ =src[src(0), src(1),\cdots, src(N-1)]

\f]

## Operation attributes

DynamicTranspose does not support any attribute.

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
| ----- | ------------- | -------------------- |
| 0     | `src`         | Required             |
| 1     | `order`       | Required             |

@note `order` is a s32 list containing the element in the range of [-N, N-1], negative value means counting from last axis

### Outputs

| Index | Argument Name | Required or Optional |
| ----- | ------------- | -------------------- |
| 0     | `dst`         | Required             |

## Supported data types

DynamicTranspose operation supports the following data type combinations.

| Src  | Dst     | Order
| ---- | ------- | ---
| f32  | f32     | s32
| bf16 | bf16    | s32
| f16  | f16     | s32
