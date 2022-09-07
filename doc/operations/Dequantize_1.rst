----------
Dequantize
----------

**Versioned name**: *Dequantize-1*

**Category**: lower_precision

**Short description**: *Dequantize* converts a quantized(u8/s8) tensor
  to a f32 tensor. It supports  both per tensor and per channel asymmetric
  linear dequantization. Nearest round is used in this OP.

**Attributes**

* *qtype*

  * **Description**: specifies which dequantization type is used.
  * **Range of values**: "per_tensor" or "per_channel"
  * **Type**: string
  * **Default value**: "per_tensor"
  * **Required**: *no*

* *axis*

  * **Description**: specifies dimension on which apply per-channel
    dequantization. Only valid if *qtype* is "per_channel". 
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

**Inputs**:

* **1**: ``input`` - quantized tensor to be dequantized. **Required.**
  
  * **Type**: T1

**Outputs**:

* **1**: ``output`` -- dequantized tensor.
  
  * **Type**: T2

**Types**:

* **T1**: s8, u8.
* **T2**: f32.