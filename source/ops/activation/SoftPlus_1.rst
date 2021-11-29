.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

--------
SoftPlus
--------

**Versioned name**: *SoftPlus-1*

**Category**: *Activation*

**Short description**: SoftPlus takes one input tensor and produces output
tensor where the SoftPlus function is applied to the tensor elementwise.

**Detailed description**: For each element from the input tensor calculates
corresponding element in the output tensor with the following formula:

.. math::
  SoftPlus(x) = 1/beta*ln(e^{beta*x} + 1.0)

**Attributes**:

* *beta*

  * **Description**: *beta* is value for the Softplus formulation. 
  * **Range of values**: A positive s64 value
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

* **1**:  Multidimensional input tensor of type T. **Required.**

  * **Type**: T

**Outputs**

* **1**:  The resulting tensor of the same shape as input tensor.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
