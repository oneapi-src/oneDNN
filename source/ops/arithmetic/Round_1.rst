.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-----
Round
-----

**Versioned name**: *Round-1*

**Category**: *Arithmetic*

**Short description**: *Round* rounds the values of a tensor to the nearest
integer, element-wise.

**Inputs**

* **1**: A tensor of type T. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result of element-wise round operation. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.