-------------------
InterpolateBackprop
-------------------

**Versioned name**: *InterpolateBackprop-1*

**Category**: *image processing*

**Short description**: Computes the gradients of Interpolate operation.

**Attributes**:

* *axes*

  * **Description**: *axes* specify spatial dimension indices where interpolation is applied. Other dimensions are treated as batch dimensions. The order of elements in axes attribute matters and mapped directly to elements with the same indices in the 2nd input target_spatial_shape.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int[]``
  * **Default value**: None
  * **Required**: *yes*

* *mode*

  * **Description**: *mode* specifies type of interpolation.
  * **Range of values**: one of nearest, linear, cubic, area
  * **Type**: ``string``
  * **Default value**: None
  * **Required**: *yes*

* *align_corners*

  * **Description**: *align_corners* is a flag that specifies whether to align corners or not. 1 means the alignment is applied, 0 means the alignment isn't applied.
  * **Range of values**: True or False
  * **Type**: ``boolean``
  * **Default value**: True
  * **Required**: *no*

* *antialias*

  * **Description**: *antialias* is a flag that specifies whether to perform anti-aliasing.
  * **Range of values**: True or False
  * **Type**: ``boolean``
  * **Default value**: False
  * **Required**: *no*

* *pads_begin*

  * **Description**: *pads_begin* specify the number of pixels to add to the beginning of the image being interpolated. This is a scalar that specifies padding for each spatial dimension.
  * **Range of values**: non-negative integer numbers
  * **Type**: ``int``
  * **Default value**: 0
  * **Required**: *no*

* *pads_end*

  * **Description**: *pads_end* specify the number of pixels to add to the end of the image being interpolated. This is a scalar that specifies padding for each spatial dimension.
  * **Range of values**: non-negative integer numbers
  * **Type**: ``int``
  * **Default value**: 0
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - Input tensor with data for interpolation. Type of elements is any supported floating point type. **Required.**
* **2**: ``target_spatial_shape`` - 1D tensor describing output shape for spatial axes. Number of elements matches the number of indices in axes attribute, the order matches as well. **Required.**
* **3**: ``src_spatial_shape`` - 1D tensor describing src shape for spatial axes. Number of elements matches the number of indices in axes attribute, the order matches as well. **Required.**

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of Interpolate.
