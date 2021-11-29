.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

--------------------------
ConvolutionBackpropFilters
--------------------------

**Versioned name**: *ConvolutionBackpropFilters-1*

**Category**: Convolution

**Short description**: Computes the gradients of a Convolution operation with
respect to the filters.

**Detailed description**:

ConvolutionBackpropFilters takes the input tensor, filter shape and output
gradient and computes the weights gradient.

**Attributes**

* *strides*

  * **Description**: *strides* has the same definition as *strides* for a
    regular Convolution.
  * **Range of values**: positive s64 values.
  * **Type**: s64[]
  * **Default value**: None
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* has the same definition as *pads_begin* for a
    regular Convolution. May be omitted specified, in which case pads are
    calculated automatically.
  * **Range of values**: Non-negative s64 values.
  * **Type**: s64[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* has the same definition as *pads_end* for a
    regular Convolution. May be omitted, in which case pads are calculated
    automatically.
  * **Range of values**: Non-negative s64 values.
  * **Type**: s64[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.
  
* *dilations*

  * **Description**: *dilations* has the same definition as *dilations* for a
    regular Convolution.
  * **Range of values**: positive s64 values.
  * **Type**: s64[]
  * **Default value**: None
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* has the same definition as *auto_pad* for a
    regular Convolution.

    * None (not specified): use explicit padding values from ``pads_begin`` and
      ``pads_end``.
    * *same_upper (same_lower)* the input is padded to match the output size.
      In case of odd padding value an extra padding is added at the end
      (at the beginning).
    * *valid* - do not use padding.

  * **Type**: string
  * **Default value**: None
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad*
    is specified.

* *groups*

  * **Description**: *groups* denotes the number of groups input channels and
    output channels are divided into. In_channels and out_channels must both be
    divisible by groups
  * **Range of values**: A positive s64 value.
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

**Inputs**:

* **1**: ``input`` - input tensor. **Required**.

  * **Type**: T

* **2**: ``filters_shape`` - 1D integer tensor that specifies spatial shape of
  the filter. The shape of filter is (out_channels, in_channels // groups,
  spatial_shape) for OIX format and (spatial_shape, in_channels // groups,
  out_channels) for XIO format. **Required**

  * **Type**: s32

* **3**: ``output_delta`` - gradients tensor with respect to the output of the
  convolution. **Required**.

  * **Type**: T

**Outputs**:

* **1**: ``filter_delta`` - gradient tensor with respect to the filter of the
  convolution. The format is specified by *filter_format*.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.