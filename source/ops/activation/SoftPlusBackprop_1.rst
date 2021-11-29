.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----------------
SoftPlusBackprop
----------------

**Versioned name**: *SoftPlusBackprop-1*

**Category**: *Activation*

**Short description**: *SoftPlusBackprop* computes gradient for SoftPlus

**Attributes**:

* *beta*

  * **Description**: *beta* is value for the Softplus formulation. 
  * **Range of values**: A positive s64 value
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

* **1**: ``input_forward`` - input of forward. **Required.**

  * **Type**: T

* **2**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  SoftPlus.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
