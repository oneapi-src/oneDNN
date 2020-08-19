--------------------------
ConvolutionBackpropFilters
--------------------------

**Versioned name**: *ConvolutionBackpropFilters-1*

**Category**: Convolution

**Short description**: Computes the gradients of a Convolution operation with respect to the filters.

**Detailed description**:

ConvolutionBackpropFilters takes the input tensor, filter shape and output gradient and computes the weights gradient.

**Attributes**

* *strides*

  * **Description**: *strides* has the same definition as *strides* for a regular Convolution.
  * **Range of values**: positive integers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *pads_begin*

  * **Description**: *pads_begin* has the same definition as *pads_begin* for a regular Convolution. May be omitted specified, in which case pads are calculated automatically.
  * **Range of values**: non-negative integers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.

* *pads_end*

  * **Description**: *pads_end* has the same definition as *pads_end* for a regular Convolution. May be omitted, in which case pads are calculated automatically.
  * **Range of values**: non-negative integers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*
  * **Note**: the attribute is ignored when *auto_pad* attribute is specified.
  
* *dilations*

  * **Description**: *dilations* has the same definition as *dilations* for a regular Convolution.
  * **Range of values**: positive integers
  * **Type**: int[]
  * **Default value**: None
  * **Required**: *yes*

* *auto_pad*

  * **Description**: *auto_pad* has the same definition as *auto_pad* for a regular Convolution.

    * None (not specified): use explicit padding values from ``pads_begin`` and ``pads_end``.
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

**Inputs**:

*   **1**: ``input`` - input tensor. **Required**.

*   **2**: ``filters_shape`` - 1D integer tensor that specifies spatial shape of the filter. **Required**.

*   **3**: ``output_delta`` - gradients tensor w.r.t. the output of the convolution. **Required**.

**Outputs**:

*   **1**: ``filter_delta`` - gradient tensor w.r.t. the filter of the convolution.
