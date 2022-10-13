# Tanh {#dev_guide_op_tanh}

**Versioned name**: *Tanh-1*

**Category**: *Activation*

**Short description**: *Tanh* element-wise activation function.

## Detailed description

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

   \f$tanh ( x ) = \frac{2}{1+e^{-2x}} - 1 = 2sigmoid(2x) - 1 \f$

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of Tanh function applied to the input
  tensor. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
