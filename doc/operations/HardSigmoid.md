# HardSigmoid {#dev_guide_op_hardsigmoid}

**Versioned name**: *HardSigmoid-1*

**Category**: *Activation*

**Short description**: *HardSigmoid* element-wise activation function.

**Detailed description**: For each element from the input tensor, calculates
corresponding element in the output tensor with the following formula:

  \f$ HardSigmoid(x) = max(0, min(1, alpha * x + beta)) \f$

## Attributes

* *alpha*

  * **Description**: Value of alpha in the formula.
  * **Range of values**: Arbitrary f32 value.
  * **Type**: f32
  * **Required**: Yes

* *beta*

  * **Description**: Value of beta in the formula.
  * **Range of values**: Arbitrary f32 value.
  * **Type**: f32
  * **Required**: Yes

## Inputs

* **1**: ``input`` - multidimensional input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - result of HardSigmoid function applied to the input tensor.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
