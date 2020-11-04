---------------
SigmoidBackprop
---------------

**Versioned name**: *SigmoidBackprop-1*

**Category**: *Activation*

**Short description**: *SigmoidBackprop* computes gradient for Sigmoid

**Attributes**:

* *use_dst*

  * **Description**: If true, use dst to calculate gradient; else use src.
  * **Range of values**: True or False
  * **Type**: Boolen
  * **Default value**: True
  * **Required**: *no*

**Inputs**:

* **1**: ``input`` - If *use_dst* is true, input is result of forward. Else,
  input is src of forward. **Required.**
* **2**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of Sigmoid.

