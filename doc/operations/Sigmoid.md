# Sigmoid {#dev_guide_op_sigmoid}

**Versioned name**: *Sigmoid-1*

**Category**: *Activation*

**Short description**: Sigmoid element-wise activation function.

## Mathematical Formulation

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

  \f$ sigmoid( x ) = \frac{1}{1+e^{-x}} \f$

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of Sigmoid function applied to the input
  tensor. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
