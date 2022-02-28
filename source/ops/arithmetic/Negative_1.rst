.. SPDX-FileCopyrightText: 2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

--------
Negative
--------

**Versioned name**: *Negative-1*

**Category**: *Arithmetic*

**Short description**: *Negative* performs element-wise negative operation on a given input tensor.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.4/openvino_docs_ops_arithmetic_Negative_1.html>`__

**Detailed description**

*Negative* performs element-wise negative operation on a given input tensor, based on the following mathematical formula:

**Mathematical Formulation**

.. math::
    output_{i} = -input_{i}

**Attributes**: *Negative* operation has no attributes.

**Inputs**

* **1**: A tensor of type T. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result of element-wise *Negative* operation applied to the input tensor. A tensor of type T and the same shape as input tensor.

  * **Type**: T



**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
