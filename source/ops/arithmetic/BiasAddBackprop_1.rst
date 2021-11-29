.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

---------------
BiasAddBackprop
---------------

**Versioned name**: *BiasAddBackprop-1*

**Category**: *Arithmetic*

**Short description**: Computes the gradients on the "bias" tensor for add bias
operator.

**Detailed description**:

This op accumulates all the values from output_delta into the channel dimension,
the axis depends on the layout of input tensor.

**Inputs**:

* **1**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**
  
  * **Type**: T

**Attributes**:

* *data_format*

  * **Description**: *data_format* denotes the data format of the input.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Outputs**:

* **1**: ``bias_delta`` - gradient tensor with respect to bias.
  
  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.