# Select {#dev_guide_op_select}

**Versioned name**: *Select-1*

**Category**: *Condition*

**Short description**: *Select* returns a tensor filled with the
elements from the second or the third input, depending on the
condition (the first input) value.

## Detailed description

*Select* takes elements from ``then`` input tensor or the ``else`` input tensor
based on a condition mask provided in the first input ``cond``. Before
performing selection, input tensors ``then`` and ``else`` are broadcasted to
each other if their shapes are different and ``auto_broadcast`` attributes is
not ``none``. Then the cond tensor is one-way broadcasted to the resulting shape
of broadcasted ``then`` and ``else``. Broadcasting is performed according to
``auto_broadcast`` value.

\f$ o_{i} = cond_{i} ? then_{i} : else_{i} \f$

## Broadcasting rules

If auto_broadcast attribute is not none, select operation takes a
two-step broadcast before performing selection:

* **Step 1**: input tensors then and else are broadcasted to each other
  if their shapes are different

* **Step 2**: then the cond tensor will be one-way broadcasted to the
  resulting shape of broadcasted then and else. To be more specific, we
  align the two shapes to the right and compare them from right to left.
  Each dimension should be either a common length or the dimension of
  cond should be 1.

* **example**:

  * cond={4, 5}, output_shape={2, 3, 4, 5} => result = {2, 3, 4, 5}
  * cond={3, 1, 5}, output_shape={2, 3, 4, 5} => result = {2, 3, 4, 5}
  * cond={3,5}, output_shape={2, 3, 4, 5} => result = invalid_shape

## Attributes

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input
    tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes should match
    * *numpy* - numpy broadcasting rules, aligned with ONNX Broadcasting.
      Description is available in `ONNX docs
      <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`__.

      **Notion**: the broadcasting here includes two different steps,
      multi-directional and uni-directional respectively. **numpy** here
      only refers to the first multi-directional step. Please refer to
      the broadcasting rules section for more details for the second step.

  * **Type**: string
  * **Default value**: *numpy*
  * **Required**: *no*

## Inputs

* **1**: ``cond`` - tensor with selection mask of type boolean. **Required.**

  * **Type**: T1

* **2**: ``then`` - the tensor with elements to take where the corresponding
  element in ``cond`` is true. **Required.**

  * **Type**: T2

* **3**: ``else`` - the tensor with elements to take where the corresponding
  element in ``cond`` is false. **Required.**

  * **Type**: T2

## Outputs

* **1**: ``output`` - blended output tensor that is tailored from values of
  inputs tensors ``then`` and ``else`` based on ``cond`` and broadcasting rules.

  * **Type**: T2

**Types**:

* **T1**: boolean.
* **T2**: f32, f16, bf16.
* **Note**: ``else`` and ``then`` tensors should have the same data type
  denoted by T2. ``cond`` tensor should be boolean data type denoted by
  T1. The output tensor has the same data type of elements as ``then`` and
  ``else``.
