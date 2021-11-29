.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----------------
HardTanhBackprop
----------------

**Versioned name**: *HardTanhBackprop-1*

**Category**: *Activation*

**Short description**: *HardTanhBackprop* computes gradient for HardTanh.

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

* **1**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

* **2**: ``input_forward`` - input of forward. **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  HardTanh.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
