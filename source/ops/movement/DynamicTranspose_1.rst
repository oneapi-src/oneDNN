.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----------------
DynamicTranspose
----------------

**Versioned name**: *DynamicTranspose-1*

**Category**: *Movement*

**Short description**: *DynamicTranspose* operation reorders the input tensor
dimensions. In DynamicTranspose, the target shape order is given as an input
tensor at runtime. It's useful when the target order is unknown during the
operator creation. Use DynamicTranspose if *order* is not constant or is not
available until runtime. Otherwise, use StaticTranspose.

**Detailed description**: *DynamicTranspose* operation reorders the input tensor
dimensions. Source indices and destination indices are bound by the formula:

.. math::
   output[i(order[0]),\ i(order[1]),\ ...,\ i(order[N-1])]\ =\ input[i(0),\ i(1),\ ...,\ i(N-1)]
   
where:

.. math::
   i(j) \ in\ range\ 0...(input.shape[j]-1)
   
The input shape is [input.shape(0), input.shape(1), ......, input.shape(N-1)],
the output shape is [input.shape(order[0]), input.shape(order[1]), ...,
input.shape(order[N-1])]. Output tensor may have a different memory layout with
input tensor. *DynamicTranspose* is not guaranteed to return a view or a copy
when input tensor and output tensor can be inplaced, users should not depend
on this behavior.

**Inputs**:

* **1**:  ``data`` - the tensor to be transposed.
  **Required.**

  * **Type**: T
  
* **2**:  ``order`` - the permutation applied to the axes of the input shape.
  It must be a vector of elements with T2 type and shape *[N]*, where N is the
  rank of ``data``. If an empty list *[]* is specified, then axes will be
  inverted to ([N-1,...,1,0]). A negative number means counting from last to the
  first axis.
  **Required.**

  * **Type**: s32

**Outputs**

* **1**: A tensor transposed from input ``data`` tensor.

  * **Type**: T

**Types**

* *T*: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
