# ConvTransposeBackpropFilters {#dev_guide_op_convtransposebackpropfilters}

**Versioned name**: *ConvTransposeBackpropFilters-1*

**Category**: Convolution

**Short description**: Computes the gradients of a ConvTranspose operation with
respect to the filters.

**Detailed description**:

ConvTransposeBackpropFilters takes the input tensor, the gradient tensor of
output and filter shape (optional) to compute the gradient of filter. The shape
of the filter should either be specified as an input 1D integer tensor or be
determined by the attribute ``filter_shape``.

## Attributes

* *strides*

  * **Description**: *strides* controls the stride along each spatial axis.
  * **Range of values**: positive s64 values.
  * **Type**: s64[]
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* controls the amount of implicit zero padding
    of each spatial axis.
  * **Range of values**: Non-negative s64 values.
  * **Type**: s64[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* controls the amount of implicit zero padding of
    each spatial axis.
  * **Range of values**: Non-negative s64 values.
  * **Type**: s64[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* controls the spacing between the kernel points.
  * **Range of values**: positive s64 values.
  * **Type**: s64[]
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* describes how the padding is calculated.

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
    divisible by groups.
  * **Range of values**: A positive s64 value.
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and
    output data.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D ConvTranspose, DHW
    for 3D ConvTranspose)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

* *filter_format*

  * **Description**: *filter_format* denotes the data format of the filter.
  * **Range of values**: *XIO* or *OIX* (X means HW for 2D ConvTranspose, DHW
    for 3D ConvTranspose)
  * **Type**: string
  * **Default value**: *XIO*
  * **Required**: *no*

* *filter_shape*

  * **Description**: *filter_shape* denotes the shape of the filter.
  * **Type**: s64[]
  * **Default value**: None
  * **Required**: *no*

## Inputs

* **1**: ``input_forward`` - original input tensor of ConvTranspose op.
  **Required**.

  * **Type**: T1

* **2**: ``output_delta`` - the gradient tensor with respect to the output of
  the ConvTranspose. **Required**.

  * **Type**: T1

* **3**: ``filter_shape`` - 1D integer tensor that specifies shape of the
  filter. The shape of filter is \f$(out\_channels / groups, in\_channels,
  spatial\_shape)\f$ for OIX format and \f$(spatial\_shape, in\_channels,
  out\_channels / groups)\f$ for XIO format. If specified, *filter_shape*
  attribute will be ignored. If not specified, users should pass *filter_shape*
  through attribute. In_channels and out_channels must both be divisible by
  *groups*.
  **Optional**.

  * **Type**: T2

## Outputs

* **1**: ``filter_delta`` - gradient tensor with respect to the filter of the
  ConvTranspose. The format is specified by *filter_format* attribute.

  * **Type**: T1

**Types**:

* **T1**: f32, f16, bf16.
* **T2**: s32
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
