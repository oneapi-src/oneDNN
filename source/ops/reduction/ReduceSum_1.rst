---------
ReduceSum
---------

**Versioned name**: *ReduceSum-1*

**Category**: *Reduction*

**Short description**: *ReduceSum* operation performs reduction with addition of
the 1st input tensor in slices specified by the second input.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_reduction_ReduceSum_1.html>`__

**Attributes**

* *keep_dims*

  * **Description**: If set to ``True`` it holds axes that are used for
    reduction. For each such axis, output dimension is equal to 1.
  * **Range of values**: True or False
  * **Type**: ``boolean``
  * **Default value**: False
  * **Required**: *no*

**Inputs**

* **1**: Input tensor x of type *T1*. **Required.**

* **2**: Scalar or 1D tensor of type *T_IND* with axis indices for the 1st input
  along which reduction is performed. Accepted range is ``[-r, r-1]`` where
  ``r`` is the rank of input tensor, all values must be unique, repeats are not
  allowed. **Required.**

**Outputs**

* **1**: Tensor of the same type as the 1st input tensor and
  ``shape[i] = shapeOf(input1)[i]`` for all ``i`` that is not in the list of
  axes from the second input. For dimensions from the second input tensor,
  ``shape[i] == 1`` if ``keep_dims == True``, or ``i``-th dimension is removed
  from the output otherwise.

**Types**

* *T1*: any supported numeric type.
* *T_IND*: ``int64`` or ``int32``.

**Detailed Description**

Each element in the output is the result of reduction with addition operation
along dimensions specified by the 2nd input:

*output[i0, i1, ..., iN] = sum[j0,..., jN](x[j0, ..., jN]))*

Where indices i0, ..., iN run through all valid indices for the 1st input and
summation ``sum[j0, ..., jN]`` have ``jk = ik`` for those dimensions ``k`` that
are not in the set of indices specified by the 2nd input of the operation. 
Corner cases:

1. When the second input is an empty list, then this operation does nothing, it
   is an identity. 
2. When the second input contains all dimensions of the 1st input, this means
   that a single reduction value is calculated for entire input tensor.
