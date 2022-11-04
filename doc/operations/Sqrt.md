# Sqrt {#dev_guide_op_sqrt}

**Versioned name**: *Sqrt-1*

**Category**: *Arithmetic*

**Short description**: *Sqrt* performs element-wise square root operation with
given tensor.

*Sqrt* does the following with the *input* tensor:

## Mathematical Formula

  \f$ output_{i} = sqrt(input_{i}) \f$

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of element-wise sqrt operation.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
