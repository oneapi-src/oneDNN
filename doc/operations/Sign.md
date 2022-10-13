# Sign {#dev_guide_op_sign}

**Versioned name**: *Sign-1*

**Category**: *Arithmetic*

**Short description**: *Sign* performs element-wise Sign operation with given
tensor.

## Mathematical Formulation

  \f$  sign(x) = \left\{\begin{array}{r}
    -1 \quad \mbox{if } x <  0 \\
     0 \quad \mbox{if } x == 0 \\
     1 \quad \mbox{if } x > 0
    \end{array}\right. \f$

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of element-wise sign operation.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
