.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-----
Rsqrt
-----

**Versioned name**: *Rsqrt-1*

**Category**: *Arithmetic*

**Short description**: *Rsqrt* performs element-wise reciprocal of square root operation with
given tensor.

**Inputs**:

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise Rsqrt operation. A tensor of type T.
  **Required.**

**Types**

* **T**: any numeric type.

*Rsqrt* does the following with the input tensor *a*:

.. math::
   a_{i} = 1 / sqrt(a_{i})

