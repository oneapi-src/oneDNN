# Rsqrt {#dev_guide_op_rsqrt}

**Versioned name**: *Rsqrt-1*

**Category**: *Arithmetic*

**Short description**: *Rsqrt* performs element-wise reciprocal of square root
operation with given tensor.

## Mathematical Formula

*Rsqrt* does the following with the *input* tensor:

  \f$ output_{i} = 1 / sqrt(input_{i}) \f$

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of element-wise Rsqrt operation.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
