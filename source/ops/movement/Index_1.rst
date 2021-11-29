.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-----
Index
-----

**Versioned name**: *Index-1*

**Category**: *Movement*

**Short description**: *Index* Select input tensor according to indices.

**Inputs**:

* **1**:  input tensor. **Required.**
  
  * **Type**: T

* **2**:  indices tensor. **Required.**
  
  * **Type**: s32

**Outputs**

* **1**:  A tensor with selected data from input tensor.
  
  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.