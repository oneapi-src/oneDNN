---------------
StaticTranspose
---------------

**Versioned name**: *StaticTranspose-1*

**Category**: *Movement*

**Short description**: *StaticTranspose* operation reorders the input tensor
dimensions. In StaticTranspose, order is given as an attribute. Use
StaticTranspose if *order* is constant and available before runtime. Otherwise,
use DynamicTranspose.

**Detailed description**: *StaticTranspose* operation reorders the input tensor
dimensions. Source indices and destination indices are bound by the formula:

.. math::
   output[i(order[0]),\ i(order[1]),\ ...,\ i(order[N-1])]\ =\ input[i(0),\ i(1),\ ...,\ i(N-1)]
   
where:

.. math::
   i(j) \ in\ range\ 0...(input.shape[j]-1)

The input shape is [input.shape(0), input.shape(1), ......, input.shape(N-1)],
the output shape is [input.shape(order[0]), input.shape(order[1]), ...,
input.shape(order[N-1])]. Output tensor may have a different memory layout with
input tensor. *StaticTranspose* is not guaranteed to return a view or a copy
when input tensor and output tensor can be inplaced, framework should not depend
on this behavior.


**Attributes**:

* *order*

  * **Description**: *order* specifies the permutation to apply to the
    axes of the input shape. *order* must be a vector of integer numbers, with
    shape *[N]*, where N is the rank of ``data``. If an empty list *[]* is
    specified, then axes will be inverted to [N-1,...,1,0].
  * **Range of values**: integer in the range [-N, N-1]. Negative number means
    counting from last to the first axis.
  * **Type**: int64[]
  * **Required**: *yes*

**Inputs**:

* **1**:  ``data`` - the tensor to be Transposed.
  **Required.**

  * **Type**: T

**Outputs**

* **1**: A Transposed tensor from input tensor.

  * **Type**: T

**Types**

* *T*: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
