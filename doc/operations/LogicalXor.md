# LogicalXor {#dev_guide_op_logicalxor}

**Versioned name**: *LogicalXor-1*

**Category**: *Logical binary*

**Short description**: *LogicalXor* performs element-wise logical XOR operation
with two given tensors applying multi-directional broadcast rules.

## Detailed description

Before performing logical operation, *input_1* and *input_2* are broadcasted if
their shapes are different and `auto_broadcast` attributes is not `none`.
Broadcasting is performed according to `auto_broadcast` value.

   \f$output_{i} = input\_1_{i} \lplus input\_2_{i} \f$

## Attributes

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:
    * *none* - no auto-broadcasting is allowed, all input shapes must match.
    * *numpy* - numpy broadcasting rules, description is available in
      [ONNX docs](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md#broadcasting-in-onnx).
  * **Type**: string
  * **Default value**: *numpy*
  * **Required**: *no*

## Inputs

* **1**: ``input_1`` - the first input tensor. **Required.**

  * **Type**: T

* **2**: ``input_2`` - the second input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of element-wise *LogicalAnd* operation
  applied to the input tensors.

  * **Type**: T

**Types**:

* **T**: `boolean`.
