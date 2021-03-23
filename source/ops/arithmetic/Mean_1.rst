.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

----
Mean
----

**Versioned name**: *Mean-1*

**Category**: *Arithmetic*

**Short description**: *Mean* operation calculate mean value of each row of the input
tensor in the given axis.

**Attributes**

* *axis*

  * **Description**: Returns the mean value of each row of the input tensor in the
    given axis. If axis is a list, reduce over all of them. If axis is None, all
    dimensions are reduced, and a tensor with a single element is returned.
  * **Range of values**: True or False
  * **Type**: ``int[]``
  * **Default value**: None
  * **Required**: *no*

* *keep_dims*

  * **Description**: If set to ``True`` it holds axes that are used for mean
    operation. For each such axis, output dimension is equal to 1.
  * **Range of values**: True or False
  * **Type**: ``boolean``
  * **Default value**: False
  * **Required**: *no*

**Inputs**

* **1**: Input tensor x of type *T*. **Required.**

**Outputs**

* **1**: The result of mean operation. A tensor of type T.

**Types**

* *T*: any supported numeric type.

