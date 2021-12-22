.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

------------------
LogSoftmaxBackprop
------------------

**Versioned name**: *LogSoftmaxBackprop-1*

**Category**: *Activation*

**Short description**:
`Reference <http://caffe.berkeleyvision.org/tutorial/layers/softmax.html>`__

**Detailed description**:
`Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#softmax>`__

**Attributes**: 

* *axis*

  * **Description**: *axis* represents the axis of which the Softmax is
    calculated. Negative value means counting dimensions from the back.
  * **Range of values**: [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: -1
  * **Required**: *no*

**Inputs**:

* **1**: output_delta **Required.**

  * **Type**: T

* **2**: forward_result **Required.**

  * **Type**: T

**Outputs**

* **1**: input_delta

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
