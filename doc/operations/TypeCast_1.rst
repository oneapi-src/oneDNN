--------
TypeCast
--------

**Versioned name**: *TypeCast-1*

**Category**: *Lower_precision*

**Short description**: *TypeCast* performs element-wise cast from input data
type to the data type given by output tensor. It supports the cast between 1)f32
to bf16/f16 2) bf16/f16 to f32.

**Detailed description**: *TypeCast* requires that input tensor and output
tensor should have the same layout and shape. User must be aware of precision
loss and value change caused by range difference between two types. Rounding to 
nearest even will be used during cast. If input and output tensors are same data
type or one date type is bf16 the other is f16, cast won't be executed.

**Inputs**:

* **1**:  input tensor. **Required.**

  * **Type**: T1
  
**Outputs**

* **1**:  A tensor casted from input tensor. **Required.**

  * **Type**: T2

**Types**:

  * **T1**: f32, f16, bf16.
  * **T2**: f32, f16, bf16.

  Constraints: When T1 is f32, T2 should be f16 or bf16; when T1 is f16 or bf16,
  T2 should be f32.