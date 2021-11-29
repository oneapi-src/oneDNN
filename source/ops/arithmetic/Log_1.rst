.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

---
Log
---

**Versioned name**: *Log-1*

**Category**: *Arithmetic*

**Short description**: *Log* performs element-wise natural logarithm operation
with given tensor.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_arithmetic_Log_1.html>`__

**Attributes**:

No attributes available.

**Inputs**:

* **1**: Input tensor. **Required.**
 
  * **Type**: T
  
**Outputs**

* **1**: The result of element-wise log operation. **Required.**
 
  * **Type**: T

**Types**

* *T*: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.

*Log* does the following with the input tensor *a*:

.. math::
   a_{i} = log(a_{i})
