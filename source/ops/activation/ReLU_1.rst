.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----
ReLU
----

**Versioned name**: *ReLU-1*

**Category**: *Activation*

**Short description**:
`Reference <http://caffe.berkeleyvision.org/tutorial/layers/relu.html>`__

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_activation_ReLU_1.html>`__

**Detailed description**:
`Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#rectified-linear-units>`__

**Attributes**: *ReLU* operation has no attributes.

**Mathematical Formulation**

.. math::
   Y_{i}^{( l )} = max(0, Y_{i}^{( l - 1 )})

**Inputs**:

* **1**: Multidimensional input tensor. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
