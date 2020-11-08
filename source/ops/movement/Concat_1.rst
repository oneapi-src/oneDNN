------
Concat
------

**Versioned name**: *Concat-1*

**Category**: *Movement*

**Short description**: `Reference
<http://caffe.berkeleyvision.org/tutorial/layers/concat.html>`__

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_movement_Concat_1.html>`__

**Attributes**:

* *axis*

  * **Description**: *axis* specifies dimension to concatenate along. 
  * **Range of values**: integer number. Negative value means counting dimension
    from the end.
  * **Type**: int
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1..N**: Arbitrary number of input tensors of type *T*. Types of all tensors
  should match. Rank of all tensors should match. The rank is positive, so
  scalars as inputs are not allowed. Shapes for all inputs should match at every
  position except ``axis`` position. At least one input is required.
  **Required.**

**Outputs**

* **1**: Tensor of the same type *T* as input tensor and shape
  ``[d1, d2, ..., d_axis, ...]``, where ``d_axis`` is a sum of sizes of input
  tensors along ``axis`` dimension.

**Types**

* *T*: any numeric type.
