-----------
EluBackprop
-----------

**Versioned name**: *EluBackprop-1*

**Category**: *Activation*

**Short description**: *EluBackprop* computes gradient for ELU

**Attributes**:

* *alpha*

  * **Description**: *alpha* is scale for the negative factor.
  * **Range of values**: arbitrary floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: ``result_forward`` - result of forward. **Required.**
* **2**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of ELU.

