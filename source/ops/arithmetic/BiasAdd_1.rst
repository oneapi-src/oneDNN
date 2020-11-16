.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

---------------
BiasAdd
---------------

**Versioned name**: *BiasAdd-1*

**Category**: *Arithmetic*

**Short description**: Adds bias to channel dimension of input.

**Detailed description**:

This is an Add with bias restricted to be 1-D. Broadcasting is supported. 

**Inputs**:

* **1**: ``input`` - data tensor. **Required.**

* **2**: ``bias`` - 1-D tensor. **Required.**

**Attributes**:

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and
    output data.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Outputs**:

* **1**: ``output`` - sum of input and bias.
