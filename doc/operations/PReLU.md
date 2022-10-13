# PReLU {#dev_guide_op_prelu}

**Versioned name**: *PReLU-1*

**Category**: *Activation*

**Short description**:
Parametric rectified linear unit element-wise activation function.

## Detailed description

*PReLU* operation is introduced in this [article](https://arxiv.org/abs/1502.01852v1).
*PReLU* performs element-wise *parametric ReLU* operation on a given input
tensor, based on the following mathematical formula:

   \f$ PReLU(x) = \left\{\begin{array}{r}
    x \quad \mbox{if } x \geq  0 \\
    \alpha x \quad \mbox{if } x < 0
    \end{array}\right \f$

## Attributes

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and
    output data.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

* *per_channel_broadcast*

  * **Description**: *per_channel_broadcast* denotes whether to apply
    per_channel broadcast when slope is 1D tensor.
  * **Range of values**: False or True
  * **Type**: bool
  * **Default value**: *True*
  * **Required**: *no*

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

* **2**: ``slope`` - slope tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - output tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

## Broadcasting rules

Only slope tensor supports broadcasting semantics. Slope tensor is
uni-directionally broadcasted to *input* if one of the following rules is met:

* **1**: slope is 1D tensor and *per_channel_broadcast* is set to True, the
  length of slope tensor is equal to the length of *input* of channel dimension.

* **2**: slope is 1D tensor and *per_channel_broadcast* is set to False, the
  length of slope tensor is equal to the length of *input* of the rightmost
  dimension.

* **3**: slope is nD tensor, starting from the rightmost dimension,
  \f$input.shape[i] == slope.shape[i]\f$ or \f$slope.shape[i] == 1\f$ or
  slope dimension i is empty.
