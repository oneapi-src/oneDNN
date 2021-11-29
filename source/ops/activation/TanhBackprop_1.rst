.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

------------
TanhBackprop
------------

**Versioned name**: *TanhBackprop-1*

**Category**: *Activation*

**Short description**: *TanhBackprop* computes gradient for Tanh

**Attributes**:

* *use_dst*

  * **Description**: If true, use *dst* to calculate gradient; else use *src*.
  * **Range of values**: True or False
  * **Type**: bool
  * **Default value**: True
  * **Required**: *no*

**Inputs**:

* **1**:  ``input`` - If *use_dst* is true, input is result of forward. Else,
  input is *src* of forward. **Required.**

  * **Type**: T

* **2**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  Tanh.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
