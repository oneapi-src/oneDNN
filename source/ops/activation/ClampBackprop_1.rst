.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-------------
ClampBackprop
-------------

**Versioned name**: *ClampBackprop-1*

**Category**: *Activation*

**Short description**: *ClampBackprop* computes gradient for Clamp

**Attributes**:

* *min*

  * **Description**: *min* is the lower bound of values in the output. Any value
    in the input that is smaller than the bound, is replaced with the min value.
    For example, min equal 10 means that any value in the input that is smaller
    than the bound, is replaced by 10.
  * **Range of values**: arbitrary valid f32 value
  * **Type**: f32
  * **Default value**: None
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output. Any value
    in the input that is greater than the bound, is replaced with the max value.
    For example, max equals 50 means that any value in the input that is greater
    than the bound, is replaced by 50.
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
  Clamp.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.

