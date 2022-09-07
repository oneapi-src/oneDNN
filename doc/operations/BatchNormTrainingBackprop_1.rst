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
  * **Required**: *yes*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input,
    output_delta and input_delta.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Inputs**

* **1**: ``input_forward`` - input of forward. **Required.**

  * **Type**: T1

* **2**: ``output_delta`` - the gradient with respect to output. **Required.**

  * **Type**: T1

* **3**: ``mean`` - batch mean. A 1D tensor with the same span
  as input's channel axis. **Required.**

  * **Type**: T2

* **4**: ``variance`` - batch variance. A 1D tensor with the same span
  as input's channel axis. **Required.**

  * **Type**: T2

* **5**: ``gamma`` - gamma scaling for normalized value. A 1D tensor with the
  same span as input's channel axis. **Optional.**

  * **Type**: T2

**Outputs**

* **1**: ``input_delta`` - the the gradient tensor with respect to the output of
  the batch normalization.

  * **Type**: T1

* **2**: ``gamma_delta`` - the the gradient tensor with respect to the gamma of
  the batch normalization. **Optional.**

  * **Type**: T2

* **3**: ``beta_delta`` - the the gradient tensor with respect to the beta of
  the batch normalization. **Optional.**

  * **Type**: T2

**Types**

* *T1*: f32, f16, bf16.
* *T2*: f32, bf16.
* Constraints: *T2* can be bf16 only when *T1* is bf16.
