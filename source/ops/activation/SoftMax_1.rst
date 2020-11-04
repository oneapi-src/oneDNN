-------
SoftMax
-------

**Versioned name**: *SoftMax-1*

**Category**: *Activation*

**Short description**:
`Reference <http://caffe.berkeleyvision.org/tutorial/layers/softmax.html>`__

**Detailed description**:
`Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#softmax>`__

**Attributes**:

* *axis*

  * **Description**: *axis* represents the axis of which the SoftMax is
    calculated. 
  * **Range of values**: integer values
  * **Type**: int
  * **Default value**: -1
  * **Required**: *no*

**Inputs**:

* **1**: Input tensor with enough number of dimension to be compatible with
  axis attribute. **Required.**

**Outputs**

* **1**: The resulting tensor of the same shape and type as input tensor.
