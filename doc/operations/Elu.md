# Elu {#dev_guide_op_elu}

**Versioned name**: *Elu-1*

**Category**: *Activation*

**Short description**: Exponential linear unit element-wise activation function.

**Detailed Description**:

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

  \f$ elu(x) = \left\{\begin{array}{ll}
       alpha(e^{x} - 1) \quad \mbox{if } x < 0 \\
       x \quad \mbox{if } x \geq  0
   \end{array}\right \f$

## Attributes

* *alpha*

  * **Description**: *alpha* is scale for the negative factor.
  * **Range of values**: arbitrary non-negative f32 value
  * **Type**: f32
  * **Required**: *yes*

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - result of ELU function applied to the input tensor.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
