# Dequantize {#dev_guide_op_dequantize}

**Versioned name**: *Dequantize-1*

**Category**: low_precision

**Short description**: *Dequantize* converts a quantized (u8 or s8) tensor
to a f32 tensor. It supports  both per-tensor and per-channel asymmetric
linear de-quantization. Nearest round is used in this OP.

For per-tensor de-quantization:

  \f$ output_{i} = (input_{i} - zp) / scale \f$

For per-channel de-quantization, taking channel axis = 1 as an example:

   \f[ output_{...,i,...,...} = (input_{...,i,...,...}  - zp_i) / scale,
   i \in {[0, channelNum-1]} \f]

## Attributes

* *qtype*

  * **Description**: specifies which de-quantization type is used.
  * **Range of values**: "per_tensor" or "per_channel"
  * **Type**: string
  * **Default value**: "per_tensor"
  * **Required**: *no*

* *axis*

  * **Description**: specifies dimension on which apply per-channel
    de-quantization. Only valid if *qtype* is "per_channel".
  * **Range of values**: integers in [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

* *scales*

  * **Description**: apply in quantization formula.
  * **Range of values**: arbitrary f32 values
  * **Type**: f32[]
  * **Required**: *yes*

* *zps*

  * **Description**: offset value that maps to float zero.
  * **Range of values**: arbitrary s64 values
  * **Type**: s64[]
  * **Required**: *yes*

## Inputs

* **1**: ``input`` - quantized tensor to be de-quantized. **Required.**

  * **Type**: T1

## Outputs

* **1**: ``output`` - de-quantized tensor.

  * **Type**: T2

**Types**:

* **T1**: s8, u8.
* **T2**: f32.
