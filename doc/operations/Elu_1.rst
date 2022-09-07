---
Elu
---

**Versioned name**: *Elu-1*

**Category**: *Activation*

**Short description**: Exponential linear unit element-wise activation function.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.1/openvino_docs_ops_activation_Elu_1.html>`__

**Detailed Description**

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

.. math::
   elu(x) = \left\{\begin{array}{ll}
       alpha(e^{x} - 1) \quad \mbox{if } x < 0 \\
       x \quad \mbox{if } x \geq  0
   \end{array}\right.

**Attributes**:

* *alpha*

  * **Description**: *alpha* is scale for the negative factor.
  * **Range of values**: arbitrary non-negative f32 value
  * **Type**: f32
  * **Required**: *yes*

**Inputs**:

* **1**: Input tensor x. **Required.**

  * **Type**: T
  
**Outputs**

* **1**: Result of ELU function applied to the input tensor x. **Required.**

  * **Type**: T
  
**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

