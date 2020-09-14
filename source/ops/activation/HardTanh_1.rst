--------
HardTanh
--------

**Versioned name**: *HardTanh-1*

**Category**: *Activation*

**Short description**: *HardTanh* element-wise activation function.

**Attributes**:

* *min*

  * **Description**: *min* is the lower bound of values in the output. 
  * **Range of values**: floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output. 
  * **Range of values**: floating point number
  * **Type**: float
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

* **1**: Input tensor x of any floating point type. **Required.**

**Outputs**

* **1**: Result of HardTanh function applied to the input tensor x. Floating point tensor with shape and type matching the input tensor. **Required.**
