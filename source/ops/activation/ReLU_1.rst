----
ReLU
----

**Versioned name**: *ReLU-1*

**Category**: *Activation*

**Short description**:
`Reference <http://caffe.berkeleyvision.org/tutorial/layers/relu.html>`__

**Detailed description**:
`Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#rectified-linear-units>`__

**Attributes**: *ReLU* operation has no attributes.

**Mathematical Formulation**

.. math::
   Y_{i}^{( l )} = max(0, Y_{i}^{( l - 1 )})

**Inputs**:

* **1**: Multidimensional input tensor. **Required.**

**Outputs**

* **1**: The result tensor.
