# LogicalNot {#dev_guide_op_logicalnot}

**Versioned name**: *LogicalNot-1*

**Category**: *Logical unary*

**Short description**: *LogicalNot* performs element-wise logical negation
operation with given tensor.

## Mathematical Formulation

  \f$ output_{i} = \lnot input_{i} \f$

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of element-wise *LogicalNot* operation
  applied to the input tensors.

  * **Type**: T

**Types**:

* **T**: `boolean`.
