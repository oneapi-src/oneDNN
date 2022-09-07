----
Tanh
----

**Versioned name**: *Tanh-1*

**Category**: *Activation*

**Short description**: *Tanh* element-wise activation function.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.1/openvino_docs_ops_arithmetic_Tanh_1.html>`__

**Attributes**: has no attributes

**Inputs**:

* **1**: Input tensor x. **Required.**

  * **Type**: T
  
**Outputs**

* **1**: Result of Tanh function applied to the input tensor *x*. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

**Detailed description**

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

.. math::
   tanh ( x ) = \frac{2}{1+e^{-2x}} - 1 = 2sigmoid(2x) - 1
