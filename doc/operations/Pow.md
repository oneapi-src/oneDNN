# Pow {#dev_guide_op_pow}

**Versioned name**: *Pow-1*

**Category**: *Arithmetic*

**Short description**: *Pow* performs element-wise power operation with two
given tensors applying multi-directional broadcast rules.

## Detailed description

Before performing arithmetic operation, *input* and *exponent* tensors are
broadcasted if their shapes are different and ``auto_broadcast`` attribute is
not ``none``. Broadcasting is performed according to ``auto_broadcast`` value.

After broadcasting *Pow* does the following with the input tensors *input* and
*exponent*:

  \f$ output_{i} = {input_{i} ^ {exponent_{i}}} \f$

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

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

* **2**: ``exponent`` - exponent tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of element-wise power operation.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
