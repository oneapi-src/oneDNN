# BiasAdd {#dev_guide_op_biasadd}

**Versioned name**: *BiasAdd-1*

**Category**: *Arithmetic*

**Short description**: Adds bias to channel dimension of input.

**Detailed description**:

This is an Add with bias restricted to be 1-D. Broadcasting is supported.

## Attributes

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and
    output data.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

* **2**: ``bias`` - 1-D tensor to be added to input tensor. The size should be
  the same as size of channel dimension of ``input`` tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
