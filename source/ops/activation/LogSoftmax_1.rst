.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----------
LogSoftmax
----------

**Versioned name**: *LogSoftmax-1*

**Category**: *Activation*

**Attributes**:

* *axis*

  * **Description**: *axis* represents the axis of which the LogSoftmax is
    calculated. 
  * **Range of values**: integer values
  * **Type**: int
  * **Default value**: -1
  * **Required**: *no*

**Inputs**:

* **1**: Input tensor with enough number of dimension to be compatible with
  axis attribute. **Required.**

**Outputs**

* **1**: The resulting tensor of the same shape and type as input tensor.
