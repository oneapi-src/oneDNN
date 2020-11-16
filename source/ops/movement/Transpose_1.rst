.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

---------
Transpose
---------

**Versioned name**: *Transpose-1*

**Category**: *Movement*

**Short description**: *Transpose* operation reorders the input tensor
dimensions.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_movement_Transpose_1.html>`__

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

**Detailed description**:

*Transpose* operation reorders the input tensor dimensions. Source indexes and
destination indexes are bound by the formula:

.. math::
   output[i(order[0]), i(order[1]), ..., i(order[N-1])] = input[i(0), i(1), ..., i(N-1)]
   
where:

.. math::
   i(j) in range 0..(input.shape[j]-1)
