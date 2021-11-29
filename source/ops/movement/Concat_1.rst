.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

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

  * **Description**: *axis* specifies dimension to concatenate along. Negative
    value means counting dimension from the end.
  * **Range of values**: [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1..N**: Arbitrary number of input tensors of type *T*. Types of all tensors
  should match. Rank of all tensors should match. The rank is positive, so
  scalars as inputs are not allowed. Shapes for all inputs should match at every
  position except ``axis`` position. At least one input is required.
  **Required.**
  
  * **Type**: T

**Outputs**

* **1**: Tensor of the same type *T* as input tensor and shape
  ``[d1, d2, ..., d_axis, ...]``, where ``d_axis`` is a sum of sizes of input
  tensors along ``axis`` dimension.
  
  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
