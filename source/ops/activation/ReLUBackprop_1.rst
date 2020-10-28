------------
ReLUBackprop
------------

**Versioned name**: *ReLUBackprop-1*

**Category**: *Activation*

**Short description**:
`Reference <http://caffe.berkeleyvision.org/tutorial/layers/relu.html>`__

**Detailed description**:
`Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLUBackprop-and-Softmax-Activation-Functions#rectified-linear-units>`__

**Attributes**: *ReLUBackprop* operation has no attributes.

**Inputs**:

* **1**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**
* **2**: ``arg`` - either forward input or output tensor of ReLU. **Required.**

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of ReLU.
