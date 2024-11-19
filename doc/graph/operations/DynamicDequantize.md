DynamicDequantize {#dev_guide_op_dynamicdequantize}
===================================================

## General

The Dynamic Dequantize operation converts a quantized (s4, u4, s8, or u8) tensor
to an bf16, f16 or f32 tensor. It supports per-tensor, per-channel, and per-group asymmetric
linear de-quantization. The rounding mode is defined by the library
implementation. Unlike the @ref dev_guide_op_dequantize, Dynamic Dequantize takes
scales and zero-points as operator src tensors.

For per-tensor de-quantization

  \f[ dst = (src - zps)*scales \f]

For per-channel de-quantization, taking channel axis = 1 as an example:
  \f[ {dst}_{\cdots,i,\cdots,\cdots} = (src_{\cdots,i,\cdots,\cdots} - zps_i)*scales_i,i\in [0,channelNum-1] \f]

For per-group de-quantization, let's take group shape = Gx1 as an example. It
indicates that one scaling factor will de adopted for G elements in the src
tensor. On the dimensions where group quantization is adopted, make channelNum
equal to the dimension of src and groupNum equal to channelNum/group size:
  \f[ {dst}_{i,\cdots} = (src_{i,\cdots} - zps_j)*scales_j,i\in [0,channelNum-1],j\in [0,groupNum-1] \f]
Where:
  \f[ i = j*groupSize+k,k\in [0,groupSize-1] \f]
On other dimensions:
  \f[ {dst}_{i,\cdots} = (src_{i,\cdots} - zps_i)*scales_i,i\in [0,channelNum-1] \f]

## Operation attributes

| Attribute Name                             | Description                                                          | Value Type | Supported Values                                                                                                                                | Required or Optional |
|:-------------------------------------------|:---------------------------------------------------------------------|:-----------|:------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------|
| [qtype](@ref dnnl::graph::op::attr::qtype) | Specifies which de-quantization type is used.                        | string     | `per_tensor` (default), `per_channel`                                                                                                           | Optional             |
| [axis](@ref dnnl::graph::op::attr::axis)   | Specifies dimension on which per-channel de-quantization is applied. | s64        | An s64 value in the range of [-r, r-1] where r = rank(src), `1` by default. Negative values mean counting the dimension backwards from the end.  | Optional             |
| [group_shape](@ref dnnl::graph::op::attr::group_shape)   | Specifies the group shape of an operation. | s64        | An s64 list indicates the group size on the dimensions where grouped quantization is adopted.  | Optional             |

## Execution arguments

The inputs and outputs must be provided according to below index order when
constructing an operation.

### Inputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `src`         | Required             |
| 1     | `scales`      | Required             |
| 2     | `zps`         | Optional             |

@note `scales` is a bf16/f16/f32 tensor to be applied to the de-quantization
formula. For `qtype` = `per-tensor`, there should be only one element in the
`scales` tensor. For `qtype` = `per-channel`, the element number should be equal
to the element number of the src tensor along the dimension axis. For
`qtype` = `per-gropup`, the `scale` tensor should have the same number of 
dimension as the `src` tensor. On the dimensions where grouped quantization is
applied, the dimension should be the number of groups, which equals to
`src_dim` / `group_size`, while other dimensions should match the `src` tensor.

@note `zps` is a tensor with offset values that map to zero. For `qtype` =
`per-tensor`, there should be only one element in the `zps` tensor. For `qtype` =
`per-channel`, the element number should be equal to the element number of input
tensor along the dimension axis. For `qtype` = `per-group`, the `zps` tensor
should have the same number of dimensions as the `src` tensor. On the dimensions
where grouped quantization is applied, the dimension should be the number of
groups, which equals to `src_dim` / `group_size`, while other dimensions should
match the `src` tensor. If omitted, the `zps` values are assumed to be zero.

### Outputs

| Index | Argument Name | Required or Optional |
|:------|:--------------|:---------------------|
| 0     | `dst`         | Required             |

## Supported data types

DynamicDequantize operation supports the following data type combinations.

| Src | Dst | Scales | Zps         |
|:-- -|:----|:-------|:------------|
| s8  | f16, bf16, f32 | f16, bf16, f32 | s8, u8, s32 |
| u8  | f16, bf16, f32 | f16, bf16, f32 | s8, u8, s32 |
| s4  | f16, bf16, f32 | f16, bf16, f32 | s4, u4, s32 |
| u4  | f16, bf16, f32 | f16, bf16, f32 | s4, u4, s32 |

It's expected that the data types of scales and dst should be the same.
