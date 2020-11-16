.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

---
Erf
---

**Versioned name**: *Erf-1*

**Category**: *Arithmetic*

**Short description**: *Erf* calculates the Gauss error function element-wise
with given tensor.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_arithmetic_Erf_1.html>`__

**Detailed description:**

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

.. math::
   erf(x) = \pi^{-1} \int_{-x}^{x} e^{-t^2} dt

**Attributes**:

No attributes available.

**Inputs**

* **1**: A tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise operation. A tensor of type T. **Required.**

**Types**

* *T*: any supported floating point type.



