# ReduceProd {#dev_guide_op_reduceprod}

**Versioned name**: *ReduceProd-1*

**Category**: *Reduction*

**Short description**: *ReduceProd* operation performs the reduction with
multiplication on a given input data along dimensions specified by
axes.

## Attributes

* *axes*

  * **Description**: specify indices of input data, along which the reduction is
    performed. If axes is a list, reduce over all of them. If axes is empty,
    corresponds to the identity operation. If axes contains all dimensions of
    input tensor, a single reduction value is calculated for the entire input
    tensor. Exactly one of attribute *axes* and the second input tensor *axes*
    should be available.
  * **Range of values**: ``[-r, r-1]`` where ``r`` = rank(``input``)
  * **Type**: s64[]
  * **Default value**: empty list
  * **Required**: *no*

* *keep_dims*

  * **Description**: If set to ``True`` it holds axes that are used for
    reduction. For each such axes, output dimension is equal to 1.
  * **Range of values**: True or False
  * **Type**: bool
  * **Default value**: False
  * **Required**: *no*

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T1

* **2**: ``axes`` - 1-D tensor specifying the axis along which the reduction is
  performed. 1D tensor of unique elements. The range of elements is
  ``[-r, r-1]``, where ``r`` is the rank of input tensor. Exactly one of
  attribute *axes* and the second input tensor *axes* should be available.
  **Optional.**.

  * **Type**: T2

## Outputs

* **1**: ``output`` - the he result of ReduceProd function applied to input tensor.
  ``shape[i] = shapeOf(data)[i]`` for all ``i`` that is not in the list of
  axes from the second input. For dimensions from ``axes``, ``shape[i] == 1``
  if ``keep_dims == True``, or ``i``-th dimension is removed from the output
  otherwise.

  * **Type**: T1

**Types**:

* **T1**: f32, f16, bf16.
* **T2**: s32.
* **Note**: The input tensor and the result tensor have the same data type
  denoted by *T1*. For example, if input is f32 tensor, then result tensor has
  f32 data type.
