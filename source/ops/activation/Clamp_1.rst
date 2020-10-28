-----
Clamp
-----

**Versioned name**: *Clamp-1*

**Category**: *Activation*

**Short description**: *Clamp* operation represents clipping activation
function.

**Attributes**:

* *min*

  * **Description**: *min* is the lower bound of values in the output. Any
    value in the input that is smaller than the bound, is replaced with the min
    value. For example, min equal 10 means that any value in the input that is
    smaller than the bound, is replaced by 10.
  * **Range of values**: non-negative positive floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output. Any value
    in the input that is greater than the bound, is replaced with the max value.
    For example, max equals 50 means that any value in the input that is greater
    than the bound, is replaced by 50.
  * **Range of values**: non-negative positive floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: Multidimensional input tensor. **Required.**

**Outputs**

* **1**: Multidimensional output tensor with shape and type matching the input
  tensor. **Required.**

