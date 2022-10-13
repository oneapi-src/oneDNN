# Square {#dev_guide_op_square}

**Versioned name**: *Square-1*

**Category**: *Arithmetic*

**Short description**: *Square* performs element-wise square operation with
given tensor.

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of element-wise square operation.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
