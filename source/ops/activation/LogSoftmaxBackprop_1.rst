------------------
LogSoftmaxBackprop
------------------

**Versioned name**: *LogSoftmaxBackprop-1*

**Category**: *Activation*

**Short description**: `Reference <http://caffe.berkeleyvision.org/tutorial/layers/softmax.html>`__

**Detailed description**: `Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#softmax>`__

**Attributes**: 

* *axis*

  * **Description**: *axis* represents the axis of which the Softmax is calculated. 
  * **Range of values**: integer values
  * **Type**: int
  * **Default value**: -1
  * **Required**: *no*

**Inputs**:

* **1**: output_delta **Required.**

* **2**: forward_result **Required.**

**Outputs**

* **1**: input_delta
