.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-------
SoftMax
-------

**Versioned name**: *SoftMax-1*

**Category**: *Activation*

**Short description**:
`Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#softmax>`__

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_activation_SoftMax_1.html>`__

**Detailed description**:
`Reference <http://cs231n.github.io/linear-classify/#softmax>`__

**Attributes**:

* *axis*

  * **Description**: *axis* represents the axis of which the *SoftMax* is
    calculated. *axis* equal 1 is a default value.
  * **Range of values**: [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

* **1**: Input tensor with enough number of dimension to be compatible with
  *axis* attribute. **Required.**

  * **Type**: T

**Outputs**

* **1**: The resulting tensor of the same shape as input tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.

**Detailed description**

.. math::
   y_{c} = \frac{e^{Z_{c}}}{\sum_{d=1}^{C}e^{Z_{d}}}

where :math:`C` is a size of tensor along *axis* dimension.