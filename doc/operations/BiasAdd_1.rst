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
  
  * **Type**: T

* **2**: ``bias`` - 1-D tensor. **Required.**
  
  * **Type**: T

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
  
  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.