.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

--------
HardTanh
--------

**Versioned name**: *HardTanh-1*

**Category**: *Activation*

**Short description**: *HardTanh* element-wise activation function.

**Attributes**:

* *min*

  * **Description**: *min* is the lower bound of values in the output. 
  * **Range of values**: arbitrary valid f32 value
  * **Type**: f32
  * **Default value**: None
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output. 
  * **Range of values**: arbitrary valid f32 value
  * **Type**: f32
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: Input tensor x. **Required.**

  * **Type**: T

**Outputs**

* **1**: Result of HardTanh function applied to the input tensor x.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
