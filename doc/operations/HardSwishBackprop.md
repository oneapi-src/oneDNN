# HardSwishBackprop {#dev_guide_op_hardswishbackprop}

**Versioned name**: *HardSwishBackprop-1*

**Category**: *Activation*

**Short description**: *HardSwishBackprop* computes gradient for HardSwish.

## Inputs

* **1**: ``input_forward`` - original input tensor of HardSwish op.
  **Required.**

  * **Type**: T

* **2**: ``output_delta`` - the gradient tensor with respect to the output.
  **Required.**

  * **Type**: T

## Outputs

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  HardSwish.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
