.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

------------
ReLUBackprop
------------

**Versioned name**: *ReLUBackprop-1*

**Category**: *Activation*

**Short description**:
`Reference <http://caffe.berkeleyvision.org/tutorial/layers/relu.html>`__

**Detailed description**:
`Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLUBackprop-and-Softmax-Activation-Functions#rectified-linear-units>`__

**Attributes**: *ReLUBackprop* operation has no attributes.

**Inputs**:

* **1**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

* **2**: ``arg`` - either forward input or output tensor of ReLU. **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  ReLU.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
