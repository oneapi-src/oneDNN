# PowBackprop {#dev_guide_op_powbackprop}

**Versioned name**: *PowBackprop-1*

**Category**: *Arithmetic*

**Short description**: *PowBackprop* computes gradient of variable base for Pow

## Inputs

* **1**: ``input_forward`` - original input tensor of Pow op. **Required.**

  * **Type**: T

* **2**: ``output_delta`` - the gradient tensor with respect to the output.
  **Required.**

  * **Type**: T

* **3**: ``exponent`` - exponent tensor of input. **Required.**

  * **Type**: T

## Outputs

* **1**: ``input_delta`` - the gradient tensor with respect to the input of Pow.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
