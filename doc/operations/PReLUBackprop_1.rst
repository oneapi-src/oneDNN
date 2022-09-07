-------------
PReLUBackprop
-------------

**Versioned name**: *PReLUBackprop-1*

**Category**: *Activation*

**Short description**: *PReLUBackprop* computes gradient for PReLU.

**Attributes**:

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and
    output data.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Inputs**:

* **1**: ``input_forward`` - input of forward. **Required.**

  * **Type**: T

* **2**: ``slope`` - slope tensor. **Required.**

  * **Type**: T

* **3**: ``output_delta`` - the gradient tensor with respect to the output.
  **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  PReLU.

* **2**: ``slope_delta`` - the gradient tensor with respect to the slope.

  * **Type**: T


**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

Broadcast rules:

Only slope tensor supports broadcast-semantics. Slope tensor is unidirectional
broadcastable to *data* if one of the following rules is true:

* **1**: PyTorch case: slope is 1D tensor and broadcast per channel, length of
  slope is equal to the length of *input_forward* in channel dimensions.

* **2**: PyTorch case: slope is 1D tensor and broadcast per tensor, length of
  slope is equal to 1.

* **3**: Tensorflow case: slope is nD tensor and its dimensions must be equal
  to the input of forward dimensions starting from the second element:

  .. code-block:: markdown

    slope_shape = input_forward_shape[1:]
