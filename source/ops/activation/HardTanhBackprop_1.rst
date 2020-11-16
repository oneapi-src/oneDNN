.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----------------
HardTanhBackprop
----------------

**Versioned name**: *HardTanhBackprop-1*

**Category**: *Activation*

**Short description**: *HardTanhBackprop* computes gradient for HardTanh.

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

* **1**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**
* **2**: ``input_forward`` - input of forward. **Required.**

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of HardTanh.
