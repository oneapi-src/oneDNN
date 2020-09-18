-----------
Convolution
-----------

**Versioned name**: *Convolution-1*

**Category**: Convolution

**Short description**: `Reference <http://caffe.berkeleyvision.org/tutorial/layers/convolution.html>`__

**Detailed description**: `Reference <http://cs231n.github.io/convolutional-networks/#conv>`__


* For the convolutional layer, the number of output features in each dimension is calculated using the formula:

    .. math::
       n_{out} = \left ( \frac{n_{in} + 2p - k}{s} \right ) + 1

* The receptive field in each layer is calculated using the formulas:

  * Jump in the output feature map:

    .. math:: 
       j_{out} = j_{in} * s

  * Size of the receptive field of output feature:

    .. math::
       r_{out} = r_{in} + ( k - 1 ) * j_{in}

  * Center position of the receptive field of the first output feature:

    .. math::
       start_{out} = start_{in} + ( \frac{k - 1}{2} - p ) * j_{in}

  * Output is calculated using the following formula:

    .. math::
       out = \sum_{i = 0}^{n}w_{i}x_{i} + b

**Attributes**

* *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the (z, y, x) axes for 3D convolutions and (y, x) axes for 2D convolutions. For example, *strides* equal *4,2,1* means sliding the filter 4 pixel at a time over depth dimension, 2 over height dimension and 1 over width dimension.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* is a number of pixels to add to the beginning along each axis. For example, *pads_begin* equal *1,2* means adding 1 pixel to the top of the input and 2 to the left of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* is a number of pixels to add to the ending along each axis. For example, *pads_end* equal *1,2* means adding 1 pixel to the bottom of the input and 2 to the right of the input.
  * **Range of values**: integer values starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *dilations*

  * **Description**: *dilations* denotes the distance in width and height between elements (weights) in the filter. For example, *dilation* equal *1,1* means that all the elements in the filter are neighbors, so it is the same as for the usual convolution. *dilation* equal *2,2* means that all the elements in the filter are matched not to adjacent elements in the input matrix, but to those that are adjacent with distance 1.
  * **Range of values**: integer value starting from 0
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* how the padding is calculated. Possible values:

    * None (not specified): use explicit padding values.
    * *same_upper (same_lower)* the input is padded to match the output size. In case of odd padding value an extra padding is added at the end (at the beginning).
    * *valid* - do not use padding.

  * **Type**: string
  * **Default value**: None
  * **Required**: *no*
  * **Note**: *pads_begin* and *pads_end* attributes are ignored when *auto_pad* is specified.

* *groups*

  * **Description**: *groups* denotes the number of groups input channels and output channels are divided into.
  * **Range of values**: integer value greater than 0
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and output data.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D convolution, DHW for 3D convolution)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

* *filter_format*

  * **Description**: *filter_format* denotes the data format of the filter.
  * **Range of values**: *XIO* or *OIX* (X means HW for 2D convolution, DHW for 3D convolution)
  * **Type**: string
  * **Default value**: *XIO*
  * **Required**: *no*

**Inputs**:

* **1**: Input tensor. The format is specified by *data_format*. The layout is determined by the value of layout in logical tensor. **Required.**
* **2**: Convolution kernel tensor. The format is specified by *filter_format*. The layout is determined by the value of layout in logical tensor. The size of the kernel is derived from the shape of this input and not specified by any attribute. **Required.**

**Outputs**:

* **1**: ``output`` -- output tensor. The dimension order is determined by the value of layout in logical tensor.
