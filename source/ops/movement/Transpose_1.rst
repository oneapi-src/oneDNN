---------
Transpose
---------

**Versioned name**: *Transpose-1*

**Category**: *Movement*

**Short description**: *Transpose* operation reorders the input tensor
dimensions.

**Inputs**:

* **1**:  ``arg`` - the tensor to be transposed. A tensor of type T1.
  **Required.**
* **2**:  ``input_order`` - the permutation to apply to the axes of the input
  shape. Must be a vector of element T2 type, with shape *[n]*, where n is
  the rank of ``arg``. The tensor's value must contain every integer in the
  range *[0,n-1]*. If an empty list is specified *[]* then the axes will be
  inverted. A tensor of type T2. **Required.**

**Outputs**

* **1**:  A tensor with shape and type matching 1st tensor.

**Types**

* *T1*: arbitrary supported type.
* *T2*: any integer type..
