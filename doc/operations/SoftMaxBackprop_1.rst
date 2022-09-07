---------------
SoftMaxBackprop
---------------

**Versioned name**: *SoftMaxBackprop-1*

**Category**: *Activation*

**Short description**:
`Reference <http://caffe.berkeleyvision.org/tutorial/layers/softmax.html>`__

**Detailed description**:
`Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#softmax>`__

**Attributes**:

* *axis*

  * **Description**: *axis* represents the axis of which the Softmax is
    calculated. 
  * **Range of values**: [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

* **1**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

* **2**: ``result_forward`` **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  SoftMax.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

