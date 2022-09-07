-----------------------
ConvolutionBackpropData
-----------------------

**Versioned name**: *ConvolutionBackpropData-1*

**Category**: Convolution

**Short description**: Computes the gradients of a Convolution operation with
respect to the input. Also known as a Deconvolution or a Transposed Convolution.

**Detailed description**:

ConvolutionBackpropData takes the input tensor, weights tensor and output shape
and computes the output tensor of a given shape. The shape of the output should
either be specified as an input 1D integer tensor or be determined by the
attribute ``output_shape``.

ConvolutionBackpropData accepts the same set of attributes as a regular
Convolution operation, but they are interpreted in a "backward way", so they are
applied to the output of ConvolutionBackpropData, but not to the input. Refer to
a regular Convolution operation for detailed description of each attribute.

If ``auto_pad`` is specified, ``pads_begin`` and ``pads_end`` will be ignored,
In this case pads are determined based on the next formulas to correctly align
input and output tensors:

.. code-block:: cpp

   total_padding[i] = stride[i] * (X_i - 1) + ((K_i - 1) * dilations[i] + 1)
                    - output_shape[i] + output_padding[i]
   if auto_pads != SAME_UPPER:
       pads_begin[i] = total_padding[i] // 2
       pads_end[i] = total_padding[i] - pads_begin[i]
   else:
       pads_end[i] = total_padding[i] // 2
       pads_begin[i] = total_padding[i] - pads_end[i]

**Attributes**

* *strides*

  * **Description**: *strides* has the same definition as *strides* for a
    regular Convolution but applied in the backward way, for the output tensor.
  * **Range of values**: positive s64 values.
  * **Type**: s64[]
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* has the same definition as *pads_begin* for a
    regular Convolution but applied in the backward way, for the output tensor.
    May be omitted specified, in which case pads are calculated automatically.
  * **Range of values**: Non-negative s64 values.
  * **Type**: s64[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* has the same definition as *pads_end* for a
    regular Convolution but applied in the backward way, for the output tensor.
    May be omitted, in which case pads are calculated automatically.
  * **Range of values**: Non-negative s64 values.
  * **Type**: s64[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.
  
* *dilations*

  * **Description**: *dilations* has the same definition as *dilations* for a
    regular Convolution but applied in the backward way, for the output tensor.
  * **Range of values**: positive s64 values.
  * **Type**: s64[]
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* has the same definition as *auto_pad* for a
    regular Convolution but applied in the backward way, for the output tensor.

    * *none (not specified)*: use explicit padding values from ``pads_begin``
      and ``pads_end``.
    * *same_upper (same_lower)* the input is padded to match the output size.
      In case of odd padding value an extra padding is added at the end
      (at the beginning).
    * *valid* - do not use padding.

  * **Type**: string
  * **Default value**: *none*
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad*
    is specified.

* *output_padding*

  * **Description**: *output_padding* adds additional amount of padding per
    each spatial axis in the ``output`` tensor. It unlocks more elements in the
    output allowing them to be computed. Elements are added at the higher
    coordinate indices for the spatial dimensions. Number of elements in
    *output_padding* list matches the number of spatial dimensions in ``data``
    and ``output`` tensors.
  * **Range of values**: Non-negative s64 values.
  * **Type**: s64[]
  * **Default value**: all zeros
  * **Required**: *no*

* *groups*

  * **Description**: *groups* denotes the number of groups input channels and
    output channels are divided into. In_channels and out_channels must both be
    divisible by groups
  * **Range of values**: A positive s64 value.
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and
    output data.
  * **Range of values**: *NXC* or *NCX* (S means HW for 2D convolution, DHW for
    3D convolution)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

* *filter_format*

  * **Description**: *filter_format* denotes the data format of the filter.
  * **Range of values**: *XIO* or *OIX* (X means HW for 2D convolution, DHW for
    3D convolution)
  * **Type**: string
  * **Default value**: *XIO*
  * **Required**: *no*

* *output_shape*

  * **Description**: *output_shape* denotes the shape of the output tensor.
  * **Type**: s64[]
  * **Required**: *no*

**Inputs**:

* **1**: ``output_delta`` - gradient tensor with respect to the output.
  **Required**.

  * **Type**: T

* **2**: ``filter`` --  convolution filter tensor. The format is specified by
  *filter_format*. The shape of filter is (out_channels, in_channels / groups,
  spatial_shape) for OIX format and (spatial_shape, in_channels / groups,
  out_channels)  for XIO format. In_channels and out_channels must both be
  divisible by groups. **Required.**

  * **Type**: T

* **3**: ``output_shape`` is 1D integer tensor that specifies shape of
  the output. **Optional**. If specified, *output_shape* attribute will be
  ignored. If not specified, users should define *output_shape* through
  attribute. *padding amount* can be deduced from relation of input and output
  spatial shapes according to formulas in the description.

  * **Type**: s32

**Outputs**:

* **1**: ``input_delta`` -- gradient tensor with respect to the input of
  convolution.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
