.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----
Tanh
----

**Versioned name**: *Tanh-1*

**Category**: *Activation*

**Short description**: *Tanh* element-wise activation function.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_arithmetic_Tanh_1.html>`__

**Attributes**: has no attributes

**Inputs**:

* **1**: Input tensor x of any floating point type. **Required.**

**Outputs**

* **1**: Result of Tanh function applied to the input tensor *x*. Floating point
  tensor with shape and type matching the input tensor. **Required.**

**Detailed description**

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

.. math::
   tanh ( x ) = \frac{2}{1+e^{-2x}} - 1 = 2sigmoid(2x) - 1
