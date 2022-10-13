# DynamicDequantize {#dev_guide_op_dynamicdequantize}

**Versioned name**: *DynamicDequantize-1*

**Category**: low_precision

**Short description**: *DynamicDequantize* converts a quantized (s8 or u8)
tensor to a f32 tensor. It supports both per-tensor and per-channel asymmetric
linear de-quantization. Rounding mode is library-implementation defined. Unlike
the static version of *Dequantize*, *DynamicDequantize* takes scales and
zero-points as operator input tensors.

For per-tensor de-quantization:

  \f$ output_{i} = (input_{i} - zp) / scale \f$

For per-channel de-quantization, taking channel axis = 1 as an example:

   \f$ output_{...,i,...,...} = (input_{...,i,...,...}  - zp_i) / scale,
   i \in {[0, channelNum-1]} \f$

## Attributes

* *qtype*

  * **Description**: specifies which de-quantization type is used.
  * **Range of values**: "per_tensor" or "per_channel"
  * **Type**: string
  * **Default value**: "per_tensor"
  * **Required**: *no*

* *axis*

  * **Description**: specifies the dimension on which "per-channel"
    de-quantization is applied. The attributes is valid only when *qtype* is
    "per_channel".
  * **Range of values**: integers in [-r, r-1] where r = rank(input). Negative
    value means counting the dimension backwards from the end.
  * **Type**: s64.
  * **Default value**: 1.
  * **Required**: *no*.

## Inputs

* **1**: ``input`` - s8/u8 tensor to be de-quantized. **Required.**

  * **Type**: T1

* **2**: ``scales`` - f32 1D tensor to be applied to the quantization formula.
  For qtype = per-tensor, there should be only one element in the scales tensor.
  For qtype = per-channel, the element number should be equal to the element
  number of input tensor along the dimension *axis*. **Required**.

  * **Type**: T2

* **3**: ``zps`` - u8/s8/s32 1D tensor with offset values that map to zero. For
  qtype = per-tensor, there should be only one element in the zps tensor. For
  qtype = per-channel, the element number should be equal to the element number
  of input tensor along the dimension *axis*. If not specified, the library can
  assume the operator is symmetric de-quantization and perform kernel
  optimization accordingly. **Optional**.

  * **Type**: T3

## Outputs

* **1**: ``output`` - f32 de-quantized tensor.

  * **Type**: T2

**Types**:

* **T1**: s8, u8.
* **T2**: f32.
* **T3**: s8, u8, s32.
