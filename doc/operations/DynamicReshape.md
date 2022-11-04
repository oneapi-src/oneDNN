# DynamicReshape {#dev_guide_op_dynamicreshape}

**Versioned name**: *DynamicReshape-1*

**Category**: Shape manipulation

**Short description**: *DynamicReshape* operation changes dimensions of the
input tensor according to the specified order. Input tensor volume is equal to
output tensor volume, where volume is the product of dimensions. Output tensor
may have a different memory layout from input tensor. *DynamicReshape* is not
guaranteed to return a view or a copy of an input tensor when output tensor is
in-placed with input tensor, user should not depend on this behavior. For
*DynamicReshape*, *shape* is given as an input tensor at runtime. It's useful
when the target shape is unknown during operator creation stage. Use
*DynamicReshape* op if *shape* is not stored in a constant node or not available
until execution stage. Otherwise, use
[StaticReshape](@ref dev_guide_op_staticreshape) op.

## Attributes

* *special_zero*

  * **Description**: *special_zero* controls how zero values in ``shape`` are
    interpreted.

    * If *special_zero* is ``false``, then ``0`` is interpreted as-is which
      means that output shape will contain a zero dimension at the specified
      location. Input and output tensors are empty in this case. ``shape``
      should not contain both ``0`` and ``-1`` since the target shape cannot be
      inferred in such a case.

    * If *special_zero* is ``true``, then all zeros in ``shape`` implies the
      copying of corresponding dimensions from the input tensor into the output
      shape.

  * **Range of values**: ``false`` or ``true``
  * **Type**: boolean
  * **Required**: *yes*

## Inputs

* **1**: ``input`` - multidimensional input tensor. **Required.**

  * **Type**: T1

* **2**: ``shape`` - specifies the output shape. The values in this tensor
  could be -1, 0 and any positive integer number. ``-1`` means that this
  dimension is calculated to keep the overall elements count the same as input
  tensor. ``0`` is interpreted by attr *special_zero*. No more than one ``-1``
  can be used in ``shape`` tensor. **Required.**

  * **Type**: T2

## Outputs

* **1**: output tensor with the same content as input ``input`` but with shape
  defined by input ``shape``.

  * **Type**: T1

**Types**:

* **T1**: f32,f16,bf16
* **T2**: s32
