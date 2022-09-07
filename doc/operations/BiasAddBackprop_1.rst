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

**Attributes**:

* *data_format*

  * **Description**: *data_format* denotes the data format of the input.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Inputs**:

* **1**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**
  
  * **Type**: T

**Outputs**:

* **1**: ``bias_delta`` - gradient tensor with respect to bias.
  
  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.