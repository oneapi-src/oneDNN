Reorder {#dev_guide_op_reorder}
===============================

## General

Reorder operation converts \src tensor to \dst tensor with different layout. It
supports the conversion between:

- Two different opaque layouts.

- Two different strided layouts.

- One strided layout and another opaque layout.

Reorder operation does not support layout conversion cross different backends or
different engines. Unlike [reorder primitive](@ref dev_guide_reorder), Reorder
operation cannot be used to cast the data type from \src to \dst. Please check
the usage of [TypeCast](@ref dev_guide_op_typecast) and
[Dequantize](@ref dev_guide_op_dequantize) operation.

## Operation attributes

Reorder operation does not support any attribute.

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

Reorder operation supports the following data type combinations.

| Src  | Dst  |
|:---- |:---- |
| f32  | f32  |
| bf16 | bf16 |
| f16  | f16  |
