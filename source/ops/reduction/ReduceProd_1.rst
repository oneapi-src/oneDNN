.. SPDX-FileCopyrightText: 2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----------
ReduceProd
----------

**Versioned name**: *ReduceProd-1*

**Category**: *Reduction*

**Short description**: *ReduceProd* operation performs the reduction with
multiplication on a given input data along dimensions specified by
axes.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.4/openvino_docs_ops_reduction_ReduceProd_1.html>`__

**Attributes**

* *axes*

  * **Description**: specify indices of input data, along which the reduction is
    performed. If axes is a list, reduce over all of them. If axes is empty,
    corresponds to the identity operation. If axes contains all dimensions of
    input **data**, a single reduction value is calculated for the entire input
    tensor. Exactly one of attribute *axes* and the second input tensor *axes*
    should be available.
  * **Range of values**: ``[-r, r-1]`` where ``r`` = rank(``input``)
  * **Type**: s64[]
  * **Default value**: None
  * **Required**: *no*

* *keep_dims*

  * **Description**: If set to ``True`` it holds axes that are used for
    reduction. For each such axes, output dimension is equal to 1.
  * **Range of values**: True or False
  * **Type**: bool
  * **Default value**: False
  * **Required**: *no*

**Inputs**

* **1**: Input tensor x of type *T1*. **Required.**

  * **Type**: T1

* **2**: Axis indices of data input tensor, along which the reduction is
  performed. 1D tensor of unique elements. The range of elements is
  ``[-r, r-1]``, where ``r`` is the rank of data input tensor. Exactly one of
  attribute *axes* and the second input tensor *axes* should be available.
  **Optional.**.

  * **Type**: T2

**Outputs**

* **1**: The result of ReduceProd function applied to data input tensor.
  ``shape[i] = shapeOf(data)[i]`` for all ``i`` that is not in the list of
  axes from the second input. For dimensions from ``axes``, ``shape[i] == 1``
  if ``keep_dims == True``, or ``i``-th dimension is removed from the output
  otherwise.

  * **Type**: T1

**Types**:

* **T1**: f32, f16, bf16.
* **T2**: s32.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
