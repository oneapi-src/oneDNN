# PowBackpropExponent {#dev_guide_op_powbackpropexponent}

**Versioned name**: *PowBackpropExponent-1*

**Category**: *Arithmetic*

**Short description**: *PowBackprop* computes gradient of exponent for Pow

## Inputs

* **1**: ``input_forward`` - original input tensor of Pow op. **Required.**

  * **Type**: T

* **2**: ``output_delta`` - the gradient tensor with respect to the output.
  **Required.**

  * **Type**: T

* **3**: ``result_forward`` - original output tensor of Pow op. **Required.**

  * **Type**: T

* **4**: ``exponent`` - exponent tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``exponent_delta`` - the gradient tensor with respect to the exponent
  of Pow.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
