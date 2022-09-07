-------------
StaticReshape
-------------

**Versioned name**: *StaticReshape-1*

**Category**: Shape manipulation

**Short description**: *StaticReshape* changes dimensions of the input tensor
according to the specified *shape*. Input tensor volume is equal to output tensor
volume, where volume is the product of dimensions. Output tensor may have a
different memory layout from input tensor. *StaticReshape* is not guaranteed to
return a view or a copy when input tensor and output tensor can be inplaced,
user should not depend on this behavior. In *StaticReshape*, *shape* is given as
an attribute. Use StaticReshape if *shape* is constant and available before
runtime. Otherwise, use DynamicReshape.

**Attributes**:

* *shape*

  * **Description**: *shape* is an array specifies the output shape.
    ``shape[i]`` gives the lengths of the i-th dimension of output.
    ``-1`` means that this dimension is calculated to keep the overall
    elements count the same as in the input tensor. No more than one ``-1`` can
    be used in a StaticReshape operation.
  * **Range of values**: values >= -1
  * **Type**: int64[]
  * **Required**: *yes*

* *special_zero*

  * **Description**: *special_zero* controls how zero values in ``shape`` are
    interpreted. If *special_zero* is ``false``, then ``0`` is interpreted as-is
    which means that output shape will contain a zero dimension at the specified
    location. Input and output tensors are empty in this case. If *special_zero*
    is ``true``, then all zeros in ``shape`` implies the copying of
    corresponding dimensions from ``data.shape`` into the output shape. If
    *special_zero* is ``false``, ``shape`` should not contain both ``0`` and
    ``-1`` in the same time. Because target shape can't be infered in this
    situation.  Shape inference begins from leftmost(i.e. shape[0]) to the
    rightmost(i.e. shape[n-1]). For example, ``data`` shape is (3, 4, 5),
    ``shape`` is (0, -1), when ``special_zero`` is true, target shape is
    inferred as (3, 20).
  * **Range of values**: ``false`` or ``true``
  * **Type**: boolean
  * **Required**: *yes*

**Inputs**:

* **1**: ``data`` -- multidimensional input tensor of type *T*. **Required.**

  * **Type**: T

**Outputs**:

* **1**: Output tensor with the same content as ``data`` but with shape defined
  by attribute ``shape``.

  * **Type**: T

**Types**

  * **T**: f32,f16,bf16
