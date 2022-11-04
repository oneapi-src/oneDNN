# Add {#dev_guide_op_add}

**Versioned name**: *Add-1*

**Category**: *Arithmetic*

**Short description**: *Add* performs element-wise addition operation with two
given tensors applying multi-directional broadcast rules.

## Detailed description

Before performing arithmetic operation, *input_1* and *input_2* are broadcasted
if their shapes are different and ``auto_broadcast`` attributes is
not ``none``. Broadcasting is performed according to ``auto_broadcast`` value.

After broadcasting *Add* does the following with *input_1* and *input_2*:

  \f$output_i = input\_1_{i}+input\_2_{i}\f$

## Attributes

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input
    tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes should match
    * *numpy* - numpy broadcasting rules, aligned with ONNX Broadcasting.
      Description is available in
      [ONNX docs](https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md).

  * **Type**: string
  * **Default value**: *numpy*
  * **Required**: *no*

## Inputs

* **1**: ``input_1`` - the first input tensor. **Required.**

  * **Type**: T

* **2**: ``input_2`` - the second input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of element-wise addition operation.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
