# ReLU {#dev_guide_op_relu}

**Versioned name**: *ReLU-1*

**Category**: *Activation*

**Short description**:
[Reference](http://caffe.berkeleyvision.org/tutorial/layers/relu.html)

**Detailed description**:
[Reference](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#rectified-linear-units)

## Mathematical Formulation

  \f$ output_{i} = max(0, input_{i}) \f$

## Inputs

* **1**: ``input`` - multidimensional input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
