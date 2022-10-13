# Quantize {#dev_guide_op_quantize}

**Versioned name**: *Quantize-1*

**Category**: low_precision

**Short description**: *Quantize* converts a f32 tensor to a quantized(u8/s8)
tensor. It supports both per-tensor and per-channel asymmetric linear
quantization. Output data type is specified in output tensor data_type. Nearest
round is used in this OP.

## Mathematical Formulation

For per-tensor quantization:

  \f$ output_{i} = round(input_{i} / scale + zp) \f$

For per-channel quantization, taking channel axis = 1 as an example:

   \f$ output_{...,i,...,...} = round(input_{...,i,...,...} / scale_i + zp_i),
   i \in {[0, channelNum-1]} \f$

## Attributes

* *qtype*

  * **Description**: specifies which quantization type is used.
  * **Range of values**: "per_tensor" or "per_channel"
  * **Type**: string
  * **Default value**: "per_tensor"
  * **Required**: *no*

* *axis*

  * **Description**: specifies dimension on which apply per-channel quantization.
    Only valid when *qtype* is "per_channel".
  * **Range of values**: integers in [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

* *scales*

  * **Description**: apply in quantization formula.
  * **Range of values**: arbitrary valid f32 value
  * **Type**: f32[]
  * **Required**: *yes*

* *zps*

  * **Description**: offset value that maps to float zero.
  * **Range of values**: arbitrary valid s64 value
  * **Type**: s64[]
  * **Required**: *yes*

## Inputs

* **1**: ``input`` - f32 tensor to be quantized. **Required.**

  * **Type**: T1

## Outputs

* **1**: ``output`` - quantized tensor.

  * **Type**: T2

**Types**:

* **T1**: f32.
* **T2**: s8, u8.
