----
ReLU
----

**Versioned name**: *ReLU-1*

**Category**: *Activation*

**Short description**:
`Reference <http://caffe.berkeleyvision.org/tutorial/layers/relu.html>`__

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.1/openvino_docs_ops_activation_ReLU_1.html>`__

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
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
