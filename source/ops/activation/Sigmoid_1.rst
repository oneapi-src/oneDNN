.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-------
Sigmoid
-------

**Versioned name**: *Sigmoid-1*

**Category**: *Activation*

**Short description**: Sigmoid element-wise activation function.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_activation_Sigmoid_1.html>`__

**Attributes**: operations has no attributes.

**Inputs**:

* **1**: Input tensor *x*. **Required.**

  * **Type**: T

**Outputs**

* **1**: Result of Sigmoid function applied to the input tensor *x*.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.

**Mathematical Formulation**

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

.. math::
   sigmoid( x ) = \frac{1}{1+e^{-x}}
