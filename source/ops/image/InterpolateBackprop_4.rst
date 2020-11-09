-------------------
InterpolateBackprop
-------------------

**Versioned name**: *InterpolateBackprop-4*

**Category**: *image processing*

**Short description**: Computes the gradients of Interpolate operation.

**Attributes**:

* *mode*

  * **Description**: *mode* specifies type of interpolation.
  * **Range of values**: one of nearest, linear, linear_onnx, cubic
  * **Type**: ``string``
  * **Default value**: None
  * **Required**: *yes*

* *shape_calculation_mode*

  * **Description**: specifies which input, sizes or scales, is used to
    calculate an forward output shape.
  * **Range of values**: name of a shape calculation mode in string format:
    sizes - an output shape is calculated as output_shape[axes[i]] = sizes[i]
            for all i in range(0, len(axes)) and output_shape[j] = 
            input_shape[j] + pads_begin[j] + pads_end[j] for j not in axes,
            j in range(0, rank(data)).
    scales - an output shape is calculated as output_shape[axes[i]] = 
            floor(scales[i] * (input_shape[axes[i]] + pads_begin[axes[i]] +
            pads_end[axes[i]])) for all i in range(0, len(axes)) and
            output_shape[j] = input_shape[j] + pads_begin[j] + pads_end[j]
            for j not in axes, j in range(0, rank(data))
  * **Type**: ``string``
  * **Default value**: None
  * **Required**: *yes*

* *coordinate_transformation_mode*

  * **Description**: specifies how to transform the coordinate in the resized
    tensor to the coordinate in the original tensor
  * **Range of values**: name of the transformation mode in string format
    (here scale[x] is output_shape[x] / input_shape[x] and x_resized is a
    coordinate in axis x, for any axis x from the input axes):
    half_pixel -  the coordinate in the original tensor axis x is calculated
                  as ((x_resized + 0.5) / scale[x]) - 0.5.
    pytorch_half_pixel -  the coordinate in the original tensor axis x is
                          calculated by (x_resized + 0.5) / scale[x] - 0.5
                          if output_shape[x] > 1 else 0.0.
    asymmetric -  the coordinate in the original tensor axis x is calculated
                  according to the formula x_resized / scale[x].
    tf_half_pixel_for_nn -  the coordinate in the original tensor axis x is
                            (x_resized + 0.5) / scale[x].
    align_corners - the coordinate in the original tensor axis x is calculated
                    as 0 if output_shape[x] == 1 else x_resized *
                    (input_shape[x] - 1) / (output_shape[x] - 1)
  * **Type**: ``string``
  * **Default value**: half_pixel
  * **Required**: *no*

* *nearest_mode*

  * **Description**: specifies round mode when mode == nearest and is used only
    when mode == nearest.
  * **Range of values**: name of the round mode in string format:
    round_prefer_floor - this mode is known as round half down.
    round_prefer_ceil - it is round half up mode.
    floor - this mode computes the largest integer value not greater than
            rounded value.
    ceil -  this mode computes the smallest integer value not less than rounded
            value.
    simple - this mode behaves as ceil mode when Interpolate is downsample, and
             as dropping the fractional part otherwise.
  * **Type**: ``string``
  * **Default value**: round_prefer_floor
  * **Required**: *no*

* *antialias*

  * **Description**: *antialias* is a flag that specifies whether to perform
    anti-aliasing.
  * **Range of values**:
    False - do not perform anti-aliasing
    True - perform anti-aliasing
  * **Type**: ``boolean``
  * **Default value**: False
  * **Required**: *no*

* *pads_begin*

  * **Description**: *pads_begin* specifies the number of pixels to add to the
    beginning of the image being interpolated. This addition of pixels is done
    before interpolation calculation.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int[]``
  * **Default value**: [0]
  * **Required**: *no*

* *pads_end*

  * **Description**: *pads_end* specifies the number of pixels to add to the end
    of the image being interpolated. This addition of pixels is done
    before interpolation calculation.
  * **Range of values**: list of non-negative integer numbers
  * **Type**: ``int[]``
  * **Default value**: [0]
  * **Required**: *no*

* *cube_coeff*

  * **Description**:  cube_coeff specifies the parameter a for cubic
    interpolation (see, e.g. article `Reference
    <https://ieeexplore.ieee.org/document/1163711/>`__). cube_coeff is used
    only when mode == cubic.
  * **Range of values**: floating point number
  * **Type**: any of supported floating point type
  * **Default value**: -0.75
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - Input tensor with data for interpolation. Type of elements
  is any supported floating point type. **Required.**
* **2**: ``output_delta`` - the gradient with respect to output. **Required.**
* **3**: ``sizes`` -  1D tensor describing output shape for spatial axes.
  Number of elements matches the number of indices in axes input, the order
  matches as well. **Required.**
* **4**: ``scales`` - 1D tensor describing scales for spatial axes. Type of
  elements is any supported floating point type. Number and order of elements
  match the number and order of indices in axes input. **Required.**
* **5**: ``axes`` - 1D tensor specifying dimension indices where interpolation
  is applied, and axes is any unordered list of indices of different dimensions
  of input tensor, e.g. [0, 4], [4, 0], [4, 2, 1], [1, 2, 3]. These indices
  should be non-negative integers from 0 to rank(data) - 1 inclusively. Other
  dimensions do not change. The order of elements in axes attribute matters,
  and mapped directly to elements in the 2nd input sizes. **Optional.** with
  default value [0,...,rank(data) - 1]. 

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of Interpolate.
* **2**: ``scales_delta`` - the gradient tensor w.r.t. the input scales of
  Interpolate. **Required.** Only when shape_calculation_mode is scales.
