# SoftMax {#dev_guide_op_softmax}

**Versioned name**: *SoftMax-1*

**Category**: *Activation*

**Short description**:
[Reference](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#softmax)

**Detailed description**:
[Reference](http://cs231n.github.io/linear-classify/#softmax)

## Mathematical Formulation**

  \f$ y_{c} = \frac{e^{Z_{c}}}{\sum_{d=1}^{C}e^{Z_{d}}} \f$

where \f$C\f$ is a size of tensor along *axis* dimension.

## Attributes

* *axis*

  * **Description**: *axis* represents the axis of which the *SoftMax* is
    calculated. *axis* equal 1 is a default value.
  * **Range of values**: [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: 1
  * **Required**: *no*

## Inputs

* **1**: ``input`` - input tensor with enough number of dimension to be
  compatible with *axis* attribute. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of the same shape as input tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
