---
Elu
---

**Versioned name**: *Elu-1*

**Category**: *Activation*

**Short description**: Exponential linear unit element-wise activation function.

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

* **1**: Result of Elu function applied to the input tensor x. Floating point
  tensor with shape and type matching the input tensor. **Required.**

