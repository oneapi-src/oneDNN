-------------------------
BatchNormTrainingBackprop
-------------------------

**Versioned name**: *BatchNormTrainingBackprop-1*

**Category**: *Normalization*

**Short description**: *BatchNormTrainingBackprop* computes gradient for batch normalization.

**Attributes**:

* *epsilon*

  * **Description**: *epsilon* is the number to be added to the variance to avoid division by zero when normalizing a value. For example, *epsilon* equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: None
  * **Required**: *yes*

* *is_training*

  * **Description**: *is_training* is used to indicate the operation is for training.
  * **Range of values**: true or false
  * **Type**: ``bool``
  * **Default value**: true
  * **Required**: *yes*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input, output_delta and input_delta.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Inputs**

* **1**: ``input`` - input tensor. **Required.**
* **2**: ``output_delta`` - the gradient with respect to output. **Required.**
* **2**: ``gamma`` - gamma scaling for normalized value. A 1D tensor with the same span as input's channel axis. **Optional.**
* **3**: ``beta`` - beta added to the scaled normalized value. A 1D tensor with the same span as input's channel axis. **Optional.**
* **4**: ``mean`` - if is_training is true, pass batch mean, otherwise running mean. **Required.**
* **5**: ``variance`` - if is_training is true, pass batch variance, otherwise running variance. **Required.**

**Outputs**

* **1**: ``input_delta`` - the the gradient tensor w.r.t. the output of the batch normolization.
* **2**: ``gamma_delta`` - the the gradient tensor w.r.t. the gamma of the batch normolization. **Optional.**
* **3**: ``beta_delta`` - the the gradient tensor w.r.t. the beta of the batch normolization. **Optional.**
