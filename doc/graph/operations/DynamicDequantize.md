DynamicDequantize {#dev_guide_op_dynamicdequantize}
===================================================

## General

DynamicDequantize operation converts a quantized (s8 or u8) tensor to a f32
tensor. It supports both per-tensor and per-channel asymmetric linear
de-quantization. Rounding mode is library-implementation defined. Unlike the
@ref dev_guide_op_dequantize, DynamicDequantize takes scales and zero-points as
operator src tensors.

For per-tensor de-quantization

  \f[ dst = (src - zps)*scales \f]

For per-channel de-quantization, taking channel axis = 1 as an example:
  \f[ {dst}_{\cdots,i,\cdots,\cdots} = (src_{\cdots,i,\cdots,\cdots} - zps_i)*scales_i,i\in [0,channelNum-1] \f]

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

@note `scales` is a f32 1D tensor to be applied to the de-quantization formula.
For `qtype` = `per-tensor`, there should be only one element in the scales
tensor. For `qtype` = `per-channel`, the element number should be equal to the
element number of src tensor along the dimension axis.

@note `zps` is a 1D tensor with offset values that map to zero. For `qtype` =
`per-tensor`, there should be only one element in the zps tensor. For `qtype` =
`per-channel`, the element number should be equal to the element number of input
tensor along the dimension axis. If not specified, the library can assume the
operator is symmetric de-quantization and perform kernel optimization accordingly.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

DynamicDequantize operation supports the following data type combinations.

| Src | Dst | Scales | Zps         |
|:-- -|:----|:-------|:------------|
| s8  | f32 | f32    | s8, u8, s32 |
| u8  | f32 | f32    | s8, u8, s32 |
