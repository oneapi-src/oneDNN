# Concat {#dev_guide_op_concat}

**Versioned name**: *Concat-1*

**Category**: *Movement*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/concat.html)

## Attributes

* *axis*

  * **Description**: *axis* specifies dimension to concatenate along. Negative
    value means counting dimension from the end.
  * **Range of values**: [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Required**: *yes*

## Inputs

* **1..N**: ``input`` - arbitrary number of input tensors of type *T*. Types of
  all tensors should match. Rank of all tensors should match. The rank is
  positive, so scalars as inputs are not allowed. Shapes for all inputs should
  match at every position except ``axis`` position. At least one input is
  required. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor whose shape is
  ``[d1, d2, ..., d_axis, ...]``, where ``d_axis`` is a sum of sizes of input
  tensors along ``axis`` dimension.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
