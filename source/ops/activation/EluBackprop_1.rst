.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-----------
EluBackprop
-----------

**Versioned name**: *EluBackprop-1*

**Category**: *Activation*

**Short description**: *EluBackprop* computes gradient for ELU

**Attributes**:

* *alpha*

  * **Description**: *alpha* is scale for the negative factor.
  * **Range of values**: arbitrary non-negative f32 value
  * **Type**: f32
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: ``result_forward`` - result of forward. **Required.**

  * **Type**: T

* **2**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of ELU.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

