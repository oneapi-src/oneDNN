----------------
SoftPlusBackprop
----------------

**Versioned name**: *SoftPlusBackprop-1*

**Category**: *Activation*

**Short description**: *SoftPlusBackprop* computes gradient for SoftPlus

**Attributes**:

* *beta*

  * **Description**: *beta* is value for the Softplus formulation. 
  * **Range of values**: a s64 value
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

* **1**: ``input_forward`` - input of forward. **Required.**

  * **Type**: T

* **2**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  SoftPlus.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
