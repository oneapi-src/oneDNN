.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

------------
GELUBackprop
------------

**Versioned name**: *GELUBackprop-1*

**Category**: *Activation*

**Short description**: *GELUBackprop* computes gradient for GELU

**Inputs**:

* **1**: ``input_forward`` - input of forward. **Required.**

  * **Type**: T

* **2**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  GELU.
 
  * **Type**: T


**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
