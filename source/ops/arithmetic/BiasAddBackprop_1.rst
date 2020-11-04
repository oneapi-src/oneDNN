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

* **1**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**

**Attributes**:

* *data_format*

  * **Description**: *data_format* denotes the data format of the input.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Outputs**:

* **1**: ``bias_delta`` - gradient tensor w.r.t. bias.
