# ClampBackprop {#dev_guide_op_clampbackprop}

**Versioned name**: *ClampBackprop-1*

**Category**: *Activation*

**Short description**: *ClampBackprop* computes gradient for Clamp

## Attributes

* *min*

  * **Description**: *min* is the lower bound of values in the output. Any value
    in the input that is smaller than the bound, is replaced with the min value.
    For example, min equal 10 means that any value in the input that is smaller
    than the bound, is replaced by 10.
  * **Range of values**: arbitrary valid f32 value
  * **Type**: f32
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output. Any value
    in the input that is greater than the bound, is replaced with the max value.
    For example, max equals 50 means that any value in the input that is greater
    than the bound, is replaced by 50.
  * **Range of values**: arbitrary valid f32 value
  * **Type**: f32
  * **Required**: *yes*

* *use_dst*

  * **Description**: If true, use *dst* to calculate gradient; else use *src*.
  * **Range of values**: True or False
  * **Type**: bool
  * **Default value**: True
  * **Required**: *no*

## Inputs

* **1**:  ``result_forward``/ ``input_forward`` - if *use_dst* is true,
  ``result_forward`` is used, else ``input_forward`` is used. **Required.**

  * **Type**: T

* **2**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

## Outputs

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  Clamp.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
