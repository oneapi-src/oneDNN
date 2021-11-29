.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
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
    calculated. Negative value means counting dimensions from the back.
  * **Range of values**: [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: -1
  * **Required**: *no*

**Inputs**:

* **1**: Input tensor with enough number of dimension to be compatible with
  axis attribute. **Required.**

  * **Type**: T

**Outputs**

* **1**: The resulting tensor of the same shape and type as input tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
