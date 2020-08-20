-------------
ClampBackprop
-------------

**Versioned name**: *ClampBackprop-1*

**Category**: *Activation*

**Short description**: *ClampBackprop* computes gradient for Clamp

**Attributes**:

* *min*

  * **Description**: *min* is the lower bound of values in the output. Any value in the input that is smaller than the bound, is replaced with the min value. For example, min equal 10 means that any value in the input that is smaller than the bound, is replaced by 10.
  * **Range of values**: non-negative positive floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output. Any value in the input that is greater than the bound, is replaced with the max value. For example, max equals 50 means that any value in the input that is greater than the bound, is replaced by 50.
  * **Range of values**: non-negative positive floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**
* **2**: ``input_forward`` - input of forward. **Required.**

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of Clamp.

