# Reciprocal {#dev_guide_op_reciprocal}

**Versioned name**: *Reciprocal-1*

**Category**: *Arithmetic*

**Short description**:Reciprocal is element-wise Power operation where exponent
(power) equals to -1. Reciprocal of 0 is infinity.

## Mathematical Formulation

  \f$ Reciprocal(x) = x^{-1}\f$

  \f$  Reciprocal(0) = inf\f$

## Inputs

* **1**: ``input`` - multidimensional input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
