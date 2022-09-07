-------
SoftMax
-------

**Versioned name**: *SoftMax-1*

**Category**: *Activation*

**Short description**:
`Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#softmax>`__

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.1/openvino_docs_ops_activation_SoftMax_1.html>`__

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
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

**Detailed description**

.. math::
   y_{c} = \frac{e^{Z_{c}}}{\sum_{d=1}^{C}e^{Z_{d}}}

where :math:`C` is a size of tensor along *axis* dimension.