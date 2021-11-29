.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----
Sqrt
----

**Versioned name**: *Sqrt-1*

**Category**: *Arithmetic*

**Short description**: *Sqrt* performs element-wise square root operation with
given tensor.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_arithmetic_Sqrt_1.html>`__

**Inputs**:

* **1**: An tensor of type T. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result of element-wise sqrt operation. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.

*Sqrt* does the following with the input tensor *a*:

.. math::
   a_{i} = sqrt(a_{i})

