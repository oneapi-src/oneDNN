.. SPDX-FileCopyrightText: 2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

---------------
DynamicReshape
---------------

**Versioned name**: *DynamicReshape-1*

**Category**: Shape manipulation

**Short description**: *DynamicReshape* operation changes dimensions of the
input tensor according to the specified order. Input tensor volume is equal to
output tensor volume, where volume is the product of dimensions. Output tensor
may have a different memory layout from input tensor. *DynamicReshape* is not
guaranteed to return a view or a copy when input tensor and output tensor can be
inplaced, user should not depend on this behavior. In DynamicReshape, *shape* is
given as an input at runtime. It's useful when the target shape is unknown
during the operator creation. Use DynamicReshape if *shape* is not constant or
is not available until runtime. Otherwise, use StaticReshape.

**Attributes**:

* *special_zero*

  * **Description**: *special_zero* controls how zero values in ``shape`` are
    interpreted. If *special_zero* is ``false``, then ``0`` is interpreted as-is
    which means that output shape will contain a zero dimension at the specified
    location. Input and output tensors are empty in this case. If *special_zero*
    is ``true``, then all zeros in ``shape`` implies the copying of
    corresponding dimensions from ``data.shape`` into the output shape.
  * **Range of values**: ``false`` or ``true``
  * **Type**: boolean
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: ``data`` -- multidimensional input tensor of type *T*. **Required.**

  * **Type**: T1

* **2**: ``shape`` -- specifies the output shape. The values in this tensor
  could be -1, 0 and any positive integer number. ``-1`` means that this
  dimension is calculated to keep the overall elements count the same as in the
  input tensor. ``0`` is interpreted by attr *special_zero*. No more than one
  ``-1`` can be used in ``shape`` tensor. **Required.**

  * **Type**: T2

**Outputs**:

* **1**: Output tensor with the same content as input ``data`` but with shape
  defined by input ``shape``.

  * **Type**: T1

**Types**

  * **T1**: f32,f16,bf16

  * **T2**: int64
