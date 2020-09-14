------
Concat
------

**Versioned name**: *Concat-1*

**Category**: *Movement*

**Short description**: `Reference <http://caffe.berkeleyvision.org/tutorial/layers/concat.html>`__

**Attributes**:

* *axis*

  * **Description**: *axis* specifies dimension to concatenate along. 
  * **Range of values**: integer values
  * **Type**: int
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: 1..N: Arbitrary number of input tensors of type T. Types of all tensors should match. Rank of all tensors should match. The rank is positive, so scalars as inputs are not allowed. Shapes for all inputs should match at every position except axis position. At least one input is required. **Required.**

**Outputs**

* **1**:  Tensor of the same type T as input tensor and shape [d1, d2, ..., d_axis, ...], where d_axis is a sum of sizes of input tensors along axis dimension.
