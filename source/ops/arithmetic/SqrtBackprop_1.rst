------------
SqrtBackprop
------------

**Versioned name**: *SqrtBackprop-1*

**Category**: *Arithmetic*

**Short description**: *SqrtBackprop* computes gradient for Sqrt

**Attributes**: 
* *use_dst*

  * **Description**: If true, use dst to calculate gradient; else use src.
  * **Range of values**: True or False
  * **Type**: Boolen
  * **Default value**: True
  * **Required**: *no*

**Inputs**:

* **1**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**
* **2**:  ``input`` - If *use_dst* is true, input is result of forward. Else, input is src of forward. **Required.**

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of Sqrt.

