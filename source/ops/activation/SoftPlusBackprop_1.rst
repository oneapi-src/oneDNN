.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----------------
SoftPlusBackprop
----------------

**Versioned name**: *SoftPlusBackprop-1*

**Category**: *Activation*

**Short description**: *SoftPlusBackprop* computes gradient for SoftPlus

**Attributes**:

* *beta*

  * **Description**: *beta* is value for the Softplus formulation. 
  * **Range of values**: positive integers
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

* **1**: ``input_forward`` - input of forward. **Required.**
* **2**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of SoftPlus.

