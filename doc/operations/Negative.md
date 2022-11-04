# Negative {#dev_guide_op_negative}

**Versioned name**: *Negative-1*

**Category**: *Arithmetic*

**Short description**: *Negative* performs element-wise negative operation on a
given input tensor.

## Detailed description

*Negative* performs element-wise negative operation on a given input tensor,
based on the following mathematical formula:

   \f$ output_{i} = -input_{i} \f$

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of element-wise *Negative* operation
  applied to the input tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
