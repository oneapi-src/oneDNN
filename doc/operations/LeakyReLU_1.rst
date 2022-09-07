---------
LeakyReLU
---------

**Versioned name**: *LeakyReLU-1*

**Category**: *Activation*

**Short description**: Leaky Rectified Linear Unit.

**Detailed description**:  LeakyReLU is a type of activation function based on
ReLU. It has a small slope for negative values with which LeakyReLU can produce
small, non-zero, and constant gradients with respect to the negative values. The
slope is also called the coefficient of leakage.

Unlike PReLU, the coefficient alpha is constant and defined before training.

.. math::
    LeakyReLU(x) = \left\{\begin{array}{r}
    x \quad \mbox{if } x \geq  0 \\
    \alpha x \quad \mbox{if } x < 0
    \end{array}\right.


**Attributes**:

* *alpha*

  * **Description**: alpha is the coefficient of leakage.
  * **Range of values**: arbitrary f32 value but usually a small positive value.
  * **Type**: f32
  * **Required**: *yes*

**Inputs**:

* **1**: Multi-dimensional input tensor. **Required.**

  * **Type**: T

**Outputs**

* **1**: Multi-dimensional output tensor which has the same data type and shape
  as the input tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
