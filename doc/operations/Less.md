# Less {#dev_guide_op_less}

**Versioned name**: *Less-1*

**Category**: *Comparison binary*

**Short description**: *Less* performs element-wise comparison operation with
two given tensors applying broadcast rules specified in the `auto_broadcast`
attribute.

## Detailed description

Before performing arithmetic operation, *input_1* and *input_2* are broadcasted
if their shapes are different and `auto_broadcast` attribute is not `none`.
Broadcasting is performed according to `auto_broadcast` value.

After broadcasting, *Less* does the following with the input tensors:

 \f$ output_{i} = input\_1_{i} < input\_2_{i} \f$

## Attributes

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:
    * *none* - no auto-broadcasting is allowed, all input shapes should match.
    * *numpy* - numpy broadcasting rules, description is available in
      [ONNX docs](https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md).
  * **Type**: string
  * **Default value**: *numpy*
  * **Required**: *no*

## Inputs

* **1**: ``input_1`` - the first input tensor. **Required.**

  * **Type**: T1

* **2**: ``input_2`` - the second input tensor. **Required.**

  * **Type**: T1

## Outputs

* **1**: ``output`` - the output tensor of element-wise *Less* operation
  applied to the input tensors.

  * **Type**: T2

**Types**:

* **T1**: f32, f16, bf16.
* **T2**: `boolean`.
