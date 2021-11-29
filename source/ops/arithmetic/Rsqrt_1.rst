.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-----
Rsqrt
-----

**Versioned name**: *Rsqrt-1*

**Category**: *Arithmetic*

**Short description**: *Rsqrt* performs element-wise reciprocal of square root
operation with given tensor.

**Inputs**:

* **1**: A tensor of type T. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result of element-wise Rsqrt operation. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.

*Rsqrt* does the following with the input tensor *a*:

.. math::
   a_{i} = 1 / sqrt(a_{i})

