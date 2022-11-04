# Erf {#dev_guide_op_erf}

**Versioned name**: *Erf-1*

**Category**: *Arithmetic*

**Short description**: *Erf* calculates the Gauss error function element-wise
with given tensor.

**Detailed description**:

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

  \f$ erf(x) = \pi^{-1} \int_{-x}^{x} e^{-t^2} dt \f$

## Attributes

No attributes available.

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of Erf operation.

  * **Type**: T

**Types**:

* *T*: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
