-----
Round
-----

**Versioned name**: *Round-1*

**Category**: *Arithmetic*

**Short description**: *Round* rounds the values of a tensor to the nearest
integer, element-wise.

**Inputs**

* **1**: A tensor of type T. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result of element-wise round operation. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
