.. SPDX-FileCopyrightText: 2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----------
Dequantize
----------

**Versioned name**: *Dequantize-1*

**Category**: lower_precision

**Short description**: *Dequantize* converts a quantized(int8 or uint8) tensor to a fp32 tensor.  
It supports  both per tensor and per channel asymmetric linear dequantization.
Nearest round is used in this OP.

**Attributes**

* *qtype*

  * **Description**: specifies which dequantization type is used.
  * **Range of values**: "per_tensor" or "per_channel"
  * **Type**: string
  * **Default value**: "per_tensor"
  * **Required**: *no*

* *axis*

  * **Description**: specifies dimension on which apply per-channel dequantization. Only valid if *qtype* is "per_channel". 
  * **Range of values**: integers in [-d, d-1] where d = input_tensor.shape().size()
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

* *scales*

  * **Description**: apply in quantization formula.
  * **Range of values**: float values
  * **Type**: float[]
  * **Default value**: None
  * **Required**: *yes*

* *zps*

  * **Description**: offset value that maps to float zero.
  * **Range of values**: integer values
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: ``input`` - quantized tensor to be dequantized. **Required.**

**Outputs**:

* **1**: ``output`` -- dequantized tensor. Data type is fp32.
