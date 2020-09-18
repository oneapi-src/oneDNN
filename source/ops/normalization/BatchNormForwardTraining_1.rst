------------------------
BatchNormForwardTraining
------------------------

**Versioned name**: *BatchNormForwardTraining-1*

**Category**: *Normalization*

**Short description**: *BatchNormForwardTraining* works on forward pass at training mode.

**Attributes**:

* *epsilon*

  * **Description**: *epsilon* is the number to be added to the variance to avoid division by zero when normalizing a value. For example, *epsilon* equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: None
  * **Required**: *yes*

* *momentum*

  * **Description**: *momentum* is used for the computation of running_mean and running_var. If it's not available, a cumulative moving average (i.e. simple average) will be computed.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Default value**: 0.1
  * **Required**: *yes*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and output data.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D convolution, DHW for 3D convolution)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Inputs**

* **1**: ``input`` - input tensor with data for normalization.  The format is specified by *data_format*. The layout is determined by the value of layout in logical tensor. **Required.**
* **2**: ``gamma`` - gamma scaling for normalized value. A 1D tensor of type T with the same span as input's channel axis. **Optional.**
* **3**: ``beta`` - beta added to the scaled normalized value. A 1D tensor of type T with the same span as input's channel axis. **Optional.**
* **4**: ``mean`` - value for mean normalization. A 1D tensor of type T with the same span as input's channel axis. **Required.**
* **5**: ``variance`` - value for variance normalization. A 1D tensor of type T with the same span as input's channel axis. **Required.**

**Outputs**

* **1**: ``output`` - the result of normalization. A tensor of the same type, shape and format with 1st input tensor.
* **2**: ``running mean`` - the computed running mean.
* **3**: ``running variance`` - the computed running variance.
* **4**: ``batch mean`` - the computed batch mean.
* **5**: ``batch variance`` - the computed batch variance.

**Mathematical Formulation**

*BatchNormForwardTraining*  normalizes the output in each hidden layer.

* **Input**: Values of :math:`x` over a mini-batch:

  .. math::
     \beta = \{ x_{1...m} \}

* **Parameters to learn**: :math:`\gamma, \beta`
* **Output**:

  .. math::
     \{ o_{i} = BN_{\gamma, \beta} ( b_{i} ) \}

* **Mini-batch mean**:

  .. math::
     \mu_{\beta} \leftarrow \frac{1}{m}\sum_{i=1}^{m}b_{i}

* **Mini-batch variance**:

  .. math::
     \sigma_{\beta }^{2}\leftarrow \frac{1}{m}\sum_{i=1}^{m} ( b_{i} - \mu_{\beta} )^{2}

* **Normalize**:

  .. math::
     \hat{b_{i}} \leftarrow \frac{b_{i} - \mu_{\beta}}{\sqrt{\sigma_{\beta }^{2} + \epsilon }}

* **Scale and shift**:

  .. math::
     o_{i} \leftarrow \gamma\hat{b_{i}} + \beta = BN_{\gamma ,\beta } ( b_{i} )

