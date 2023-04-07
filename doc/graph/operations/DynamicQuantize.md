DynamicQuantize {#dev_guide_op_dynamicquantize}
===============================================

## General

DynamicQuantize operation converts a f32 tensor to a quantized (s8 or u8)
tensor. It supports both per-tensor and per-channel asymmetric linear
quantization. The target quantized data type is specified via the data type of
dst logical tensor. Rounding mode is library-implementation defined.

For per-tensor quantization

  \f[ dst = round(src/scales + zps) \f]

For per-channel quantization, taking channel axis = 1 as an example:
  \f[ {dst}_{\cdots,i,\cdots,\cdots} =
  round(src_{\cdots,i,\cdots,\cdots}/scales_i + zps_i),i\in [0,channelNum-1] \f]

## Operation attributes

| Attribute Name                             | Description                                                          | Value Type | Supported Values                                                                                                                                | Required or Optional |
|:-------------------------------------------|:---------------------------------------------------------------------|:-----------|:------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------|
| [qtype](@ref dnnl::graph::op::attr::qtype) | Specifies which de-quantization type is used.                        | string     | `per_tensor` (default), `per_channel`                                                                                                           | Optional             |
| [axis](@ref dnnl::graph::op::attr::axis)   | Specifies dimension on which per-channel de-quantization is applied. | s64        | A s64 value in the range of [-r, r-1] where r = rank(src), `1` by default. Negative value means counting the dimension backwards from the end.  | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `scales`      | Required             |
| 2     | `zps`         | Optional             |

@note `scales` is a f32 1D tensor to be applied to the quantization formula. For
`qtype` = `per-tensor`, there should be only one element in the scales tensor.
For `qtype` = `per-channel`, the element number should be equal to the element
number of src tensor along the dimension axis.

@note `zps` is a 1D tensor with offset values that map to zero. For `qtype` = `per-tensor`, there should be only one
element in the zps tensor. For `qtype` = `per-channel`, the element number should be
equal to the element number of input tensor along the dimension axis. If not
specified, the library can assume the operator is symmetric quantization and
perform kernel optimization accordingly.

### Outputs

Index | Argument Name | Required or Optional
----- | ------------- | --------------------
0     | `dst`         | Required

## Supported data types

DynamicQuantize operation supports the following data type combinations.

| Src | Scales | Zps         | Dst |
|:----|:-------|:------------|:----|
| f32 | f32    | s8, u8, s32 | s8  |
| f32 | f32    | s8, u8, s32 | u8  |
