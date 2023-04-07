Quantize {#dev_guide_op_quantize}
=================================

## General

Quantize operation converts a f32 tensor to a quantized (u8/s8) tensor. It
supports both per-tensor and per-channel asymmetric linear quantization. Output
data type is specified in output tensor data type. Rounding mode is
library-implementation defined.

For per-tensor quantization:

  \f[ \dst_{i} = round(\src_{i} / scale + zp) \f]

For per-channel quantization, taking channel axis = 1 as an example:

   \f[ dst_{\cdots,i,\cdots,\cdots} = round(\src_{\cdots,i,\cdots,\cdots} / scale_i + zp_i), i \in {[0, ic-1]} \f]

where \f$ic\f$ is the number of channels.

## Operation attributes

| Attribute Name                               | Description                                                       | Value Type | Supported Values                                                          | Required or Optional |
|:---------------------------------------------|:------------------------------------------------------------------|:-----------|:--------------------------------------------------------------------------|:---------------------|
| [qtype](@ref dnnl::graph::op::attr::qtype)   | Specifies which quantization type is used.                        | string     | `per_tensor` (default), `per_channel`                                     | Optional             |
| [axis](@ref dnnl::graph::op::attr::axis)     | Specifies dimension on which per-channel quantization is applied. | s64        | A s64 value in the range of [-r, r-1] where r = rank(src), `1` by default | Optional             |
| [scales](@ref dnnl::graph::op::attr::scales) | Scalings applied on the src data.                                 | f32        | A f32 list (only contain one element if qtype is `per_tensor`)            | Required             |
| [zps](@ref dnnl::graph::op::attr::zps)       | Offset values that maps to float zero.                            | s64        | A s64 list (only contain one element if qtype is `per_tensor`)            | Required             |

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

Quantize operation supports the following data type combinations.

| Src | Dst    |
|:----|:-------|
| f32 | s8, u8 |

@note This operation is to support
[int8 quantization](@ref dev_guide_graph_int8_quantization_model) model.
