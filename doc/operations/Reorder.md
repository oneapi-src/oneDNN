# Reorder {#dev_guide_op_reorder}

**Versioned name**: *Reorder-1*

**Category**: *Movement*

**Short description**: *Reorder* converts the input tensor to output tensor with
different layouts. It supports the conversion between 1) two different opaque
layouts 2) two different public layouts 3) one opaque layout and another public
layout.

## Detailed description

*Reorder* also requires that the input tensor and output tensor should have the
same data type (f32/bf16/f16) and shape

For example, if the input tensor has public layout with strides, and users want
to convert it to an output tensor with opaque layout (specified by layout id),
then reorder can be used for this case.

Currently, *reorder* operator doesn't support layout conversion cross backends
or cross engines.

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**:  ``output`` - the output tensor with different layout from input tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
