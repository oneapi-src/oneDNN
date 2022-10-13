# ConvolutionBackpropFilters {#dev_guide_op_convolutionbackpropfilters}

**Versioned name**: *ConvolutionBackpropFilters-1*

**Category**: Convolution

**Short description**: Computes the gradients of a Convolution operation with
respect to the filters.

**Detailed description**:

ConvolutionBackpropFilters takes the input tensor, the gradient of output tensor,
and filter shape (optional) to compute the gradient of filter.

## Attributes

* *strides*

  * **Description**: *strides* has the same definition as *strides* for a
    regular Convolution.
  * **Range of values**: positive s64 values.
  * **Type**: s64[]
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* has the same definition as *pads_begin* for a
    regular Convolution. May be omitted specified, in which case pads are
    calculated automatically.
  * **Range of values**: non-negative s64 values.
  * **Type**: s64[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* has the same definition as *pads_end* for a
    regular Convolution. May be omitted, in which case pads are calculated
    automatically.
  * **Range of values**: non-negative s64 values.
  * **Type**: s64[]
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* has the same definition as *dilations* for a
    regular Convolution.
  * **Range of values**: positive s64 values.
  * **Type**: s64[]
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* has the same definition as *auto_pad* for a
    regular Convolution.

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

* *groups*

  * **Description**: *groups* denotes the number of groups input channels and
    output channels are divided into. In_channels and out_channels must both be
    divisible by groups
  * **Range of values**: a positive s64 value.
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input data and
    output delta.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D convolution, DHW for
    3D convolution)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

* *filter_format*

  * **Description**: *filter_format* denotes the data format of the filter
    gradient.
  * **Range of values**: *XIO* or *OIX* (X means HW for 2D convolution, DHW for
    3D convolution)
  * **Type**: string
  * **Default value**: *XIO*
  * **Required**: *no*

* *filter_shape*

  * **Description**: *filter_shape* denotes the shape of the filter.
  * **Type**: s64[]
  * **Required**: *no*

## Inputs

* **1**: ``input_forward`` - original input tensor of Convolution op.
  **Required**.

  * **Type**: T

* **2**: ``output_delta`` - the gradient tensor with respect to the output of
  the convolution. **Required**.

  * **Type**: T

* **3**: ``filter_shape`` - 1D integer tensor that specifies shape of the
  filter. The shape of filter is
  \f$(out\_channels, in\_channels / groups, spatial\_shape)\f$ for OIX format or
  \f$(spatial\_shape, in\_channels / groups, out\_channels)\f$ for XIO format.
  \f$in\_channels\f$ and \f$out\_channels\f$ must both be divisible by groups.
  If specified, *filter_shape* attribute will be ignored. If not specified,
  users should define *filter_shape* through attribute. **Optional**.

  * **Type**: s32

## Outputs

* **1**: ``filter_delta`` - gradient tensor with respect to the filter of the
  convolution. The format is specified by *filter_format* attribute.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
