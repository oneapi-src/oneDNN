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
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
