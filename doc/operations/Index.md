# Index {#dev_guide_op_index}

**Versioned name**: *Index-1*

**Category**: *Movement*

**Short description**: *Index* selects input tensor according to indices.

## Inputs

* **1**:  ``input`` - input tensor. **Required.**

  * **Type**: T

* **2**:  ``indices`` - indices tensor. **Required.**

  * **Type**: s32

## Outputs

* **1**: ``output`` - the output tensor with selected data from input tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
