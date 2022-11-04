# Mish {#dev_guide_op_mish}

**Versioned name**: *Mish-1*

**Category**: *Activation*

**Short description**: *Mish* is a Self Regularized Non-Monotonic Neural
Activation Function.

## Detailed Description

*Mish* is a self regularized non-monotonic neural activation function proposed
in this [article](https://arxiv.org/abs/1908.08681v2).

*Mish* performs element-wise activation function on a given input tensor, based
on the following mathematical formula:

  \f$  Mish(x) = x \cdot tanh(SoftPlus(x)) = x \cdot tanh(ln(1 + e^x)) \f$

## Inputs

* **1**: ``input`` - multidimensional input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - multidimensional output tensor with shape and type
  matching the input tensor. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
