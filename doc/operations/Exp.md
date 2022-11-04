# Exp {#dev_guide_op_exp}

**Versioned name**: *Exp-1*

**Category**: *Activation*

**Short description**: Exponential element-wise activation function.

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - result of Exp function applied to the input tensor.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
