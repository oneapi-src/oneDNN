# StaticReshape {#dev_guide_op_staticreshape}

**Versioned name**: *StaticReshape-1*

**Category**: Shape manipulation

## Detailed description

*StaticReshape* changes dimensions of a input tensor according to the specified
*shape*. Input tensor volume is equal to output tensor volume, where volume is
the product of dimensions. Output tensor may have a different memory layout from
input tensor. *StaticReshape* is not guaranteed to return a view or a copy of an
input tensor when output tensor is in-placed with the input tensor, user should
not depend on this behavior. For *StaticReshape* op, *shape* is given as an
attribute. Users can use *StaticReshape* if *shape* is stored in a constant node
or available during graph building stage. Otherwise, use
[DynamicReshape](@ref dev_guide_op_dynamicreshape) op.

## Attributes

* *shape*

  * **Description**: *shape* is an array specifies the output shape.
    ``shape[i]`` gives the lengths of the i-th dimension of output. ``-1`` means
    that this dimension is calculated to keep the overall elements count the
    same as the input tensor. No more than one ``-1`` can be used in a
    *StaticReshape* operation.
  * **Range of values**: s64 values where value is no less than -1
  * **Type**: s64[]
  * **Required**: *yes*

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

* **1**: ``input`` - multidimensional input tensor of type *T*. **Required.**

  * **Type**: T

## Outputs

* **1**: Output tensor with the same content as ``input`` but with shape defined
  by attribute ``shape``.

  * **Type**: T

**Types**:

* **T**: f32,f16,bf16
