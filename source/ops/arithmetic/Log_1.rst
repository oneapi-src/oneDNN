.. SPDX-FileCopyrightText: 2020 Intel Corporation
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

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise log operation. A tensor of type T.
  **Required.**

**Types**

* **T**: any numeric type.

*Log* does the following with the input tensor *a*:

.. math::
   a_{i} = log(a_{i})
