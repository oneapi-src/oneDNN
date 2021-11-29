.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-------------------------
BatchNormTrainingBackprop
-------------------------

**Versioned name**: *BatchNormTrainingBackprop-1*

**Category**: *Normalization*

**Short description**: *BatchNormTrainingBackprop* computes gradient for batch
normalization.

**Attributes**:

* *epsilon*

  * **Description**: *epsilon* is the number to be added to the variance to
    avoid division by zero when normalizing a value. For example, *epsilon*
    equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: arbitrary positive f32 value 
  * **Type**: f32
  * **Default value**: None
  * **Required**: *yes*

* *is_training*

  * **Description**: *is_training* is used to indicate the operation is for
    training.
  * **Range of values**: true or false
  * **Type**: ``bool``
  * **Default value**: true
  * **Required**: *no*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input,
    output_delta and input_delta.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Inputs**

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

* **2**: ``output_delta`` - the gradient with respect to output. **Required.**

  * **Type**: T

* **3**: ``gamma`` - gamma scaling for normalized value. A 1D tensor with the
  same span as input's channel axis. **Optional.**

  * **Type**: f32

* **4**: ``beta`` - beta added to the scaled normalized value. A 1D tensor with
  the same span as input's channel axis. **Optional.**

  * **Type**: f32

* **5**: ``mean`` - if is_training is true, pass batch mean, otherwise running
  mean. **Required.**

  * **Type**: f32

* **6**: ``variance`` - if is_training is true, pass batch variance, otherwise
  running variance. **Required.**

  * **Type**: f32

**Outputs**

* **1**: ``input_delta`` - the the gradient tensor with respect to the output of
  the batch normalization.

  * **Type**: T

* **2**: ``gamma_delta`` - the the gradient tensor with respect to the gamma of
  the batch normalization. **Optional.**

  * **Type**: f32

* **3**: ``beta_delta`` - the the gradient tensor with respect to the beta of
  the batch normalization. **Optional.**

  * **Type**: f32

**Types**

* *T*: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
  