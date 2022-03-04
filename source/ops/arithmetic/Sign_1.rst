.. SPDX-FileCopyrightText: 2022 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----
Sign
----

**Versioned name**: *Sign-1*

**Category**: *Arithmetic*

**Short description**: *Sign* performs element-wise Sign operation with given
tensor.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.4/openvino_docs_ops_arithmetic_Sign_1.html>`__


**Mathematical Formulation**

.. math::
    output_{i} = sign( input_{i} )

**Attributes**: *Sign* operation has no attributes.

**Inputs**

* **1**: A tensor of type T. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result of element-wise sign operation. A tensor of type T with
mapped elements of the input tensor to -1 (if it is negative),
0 (if it is zero), or 1 (if it is positive).

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
