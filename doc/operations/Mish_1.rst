-----
Mish
-----

**Versioned name**: *Mish-1*

**Category**: *Activation*

**Short description**: *Mish* is a Self Regularized Non-Monotonic Neural Activation Function.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.4/openvino_docs_ops_activation_Mish_4.html>`__

**Detailed Description**

*Mish* is a self regularized non-monotonic neural activation function proposed in this [article](https://arxiv.org/abs/1908.08681v2).

*Mish* performs element-wise activation function on a given input tensor, based on the following mathematical formula:

.. math::
   Mish(x) = x \cdot tanh(SoftPlus(x)) = x \cdot tanh(ln(1 + e^x)).

**Attributes**: *Mish* operation has no attributes.

**Inputs**:

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

  * **Type**: T
  
**Outputs**

* **1**: The result of element-wise *Mish* function applied to the input tensor. A tensor of type *T* and the same shape as input tensor. **Required.**

  * **Type**: T
  
**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

