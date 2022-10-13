# DynamicQuantize {#dev_guide_op_dynamicquantize}

**Versioned name**: *DynamicQuantize-1*

**Category**: low_precision

**Short description**: *DynamicQuantize* converts a f32 tensor to a quantized
(s8 or u8) tensor. It supports both per-tensor and per-channel asymmetric linear
quantization. The target quantized data type is specified via the data type of
output logical tensor. Rounding mode is library-implementation defined.

For per-tensor quantization:

  \f$ output_{i} = round(input_{i} / scale + zp) \f$

For per-channel quantization, taking channel axis = 1 as an example:

   \f$ output_{...,i,...,...} = round(input_{...,i,...,...} / scale_i + zp_i),
    i \in {[0, channelNum-1]} \f$

Unlike the static version of *Quantize*, *DynamicQuantize* takes scales and
zero-points as operator input tensors.

## Attributes

* *qtype*

  * **Description**: specifies which quantization type is used.
  * **Range of values**: "per_tensor" or "per_channel".
  * **Type**: string.
  * **Default value**: "per_tensor".
  * **Required**: *no*.

* *axis*

  * **Description**: specifies the dimension on which "per-channel" quantization
    is applied. The attributes is valid only when *qtype* is "per_channel".
  * **Range of values**: integers in [-r, r-1] where r = rank(input). Negative
    value means counting the dimension backwards from the end.
  * **Type**: s64.
  * **Default value**: 1.
  * **Required**: *no*.

## Inputs

* **1**: ``input`` - f32 tensor to be quantized. **Required**.

  * **Type**: T1

* **2**: ``scales`` - f32 1D tensor to be applied to the quantization formula.
  For qtype = per-tensor, there should be only one element in the scales tensor.
  For qtype = per-channel, the element number should be equal to the element
  number of input tensor along the dimension *axis*. **Required**.

  * **Type**: T1

* **3**: ``zps`` - u8/s8/s32 1D tensor with offset values that map to zero. For
  qtype = per-tensor, there should be only one element in the zps tensor. For
  qtype = per-channel, the element number should be equal to the element number
  of input tensor along the dimension *axis*. If not specified, the library can
  assume the operator is symmetric quantization and perform kernel optimization
  accordingly. **Optional**.

  * **Type**: T2

## Outputs**

* **1**: ``output`` - quantized tensor.

  * **Type**: T3

**Types**:

* **T1**: f32.
* **T2**: s8, u8, s32.
* **T3**: s8, u8.
