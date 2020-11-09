---
Elu
---

**Versioned name**: *Elu-1*

**Category**: *Activation*

**Short description**: Exponential linear unit element-wise activation function.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_activation_Elu_1.html>`__

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
  * **Range of values**: arbitrary floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: Input tensor x of any floating point type. **Required.**

**Outputs**

* **1**: Result of ELU function applied to the input tensor x. Floating point
  tensor with shape and type matching the input tensor. **Required.**

