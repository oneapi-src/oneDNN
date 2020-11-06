---------
ReduceSum
---------

**Versioned name**: *ReduceSum-1*

**Category**: *Reduction*

**Short description**: *ReduceSum* operation performs reduction with addition of
the 1st input tensor in slices specified by the second input.

**Attributes**

* *keep_dims*

  * **Description**: If set to `True` it holds axes that are used for reduction.
    For each such axis, output dimension is equal to 1.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: False
  * **Required**: *no*

**Inputs**

* **1**: Input tensor x of any data type that has defined addition operation.
  **Required**.

* **2**: Scalar or 1D tensor with axis indices for the 1st input along which
  reduction is performed. **Required**.

**Outputs**

* **1**: Tensor of the same type as the 1st input tensor and
  :math:`shape[i] = shapeOf(input1)[i]` for all :math:`i` that is not in the
  list of axes from the second input. For dimensions from the second input tensor,
  :math:`shape[i] == 1` if :math:`keep\_dims == True`, or :math:`i`\ *th* dimension
  is removed from the output otherwise.

**Detailed Description**

Corner cases: 

1. When the second input is an empty list, then this operation does nothing, it is
an identity. 
2. When the second input contains all dimensions of the 1st input, this means that
a single reduction value is calculated for entire input tensor.
