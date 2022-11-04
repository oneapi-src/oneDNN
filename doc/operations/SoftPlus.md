# SoftPlus {#dev_guide_op_softplus}

**Versioned name**: *SoftPlus-1*

**Category**: *Activation*

**Short description**: SoftPlus takes one input tensor and produces output
tensor where the SoftPlus function is applied to the tensor elementwise.

## Detailed description

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

 \f$ SoftPlus(x) = 1/beta*ln(e^{beta*x} + 1.0) \f$

## Attributes

* *beta*

  * **Description**: *beta* is value for the Softplus formulation.
  * **Range of values**: a s64 value
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

## Inputs

* **1**:  ``input`` - multidimensional input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**:  ``output`` - the output tensor of the same shape as input tensor.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
