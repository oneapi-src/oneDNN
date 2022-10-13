# BatchNormInference {#dev_guide_op_batchnorminference}

**Versioned name**: *BatchNormInference-1*

**Category**: *Normalization*

**Short description**: *BatchNormInference* layer normalizes a ``input`` tensor
by ``mean`` and ``variance``, and applies a scale (``gamma``) to it, as well as
an offset (``beta``).

## Mathematical Formulation

*BatchNormInference*  normalizes the output in each hidden layer.

* **Input**: Values of \f$x\f$ over a mini-batch:

    \f$ \beta = \{ x_{1...m} \}\f$

* **Parameters to learn**: \f$\gamma\f$, \f$\beta\f$
* **Output**:

    \f$ \{ o_{i} = BN_{\gamma, \beta} ( b_{i} ) \}\f$

* **Mini-batch mean**:

    \f$ \mu_{\beta} \leftarrow \frac{1}{m}\sum_{i=1}^{m}b_{i}\f$

* **Mini-batch variance**:

    \f$ \sigma_{\beta }^{2}\leftarrow \frac{1}{m}\sum_{i=1}^{m} ( b_{i} -
      \mu_{\beta} )^{2}\f$

* **Normalize**:

    \f$ \hat{b_{i}} \leftarrow \frac{b_{i} -
      \mu_{\beta}}{\sqrt{\sigma_{\beta }^{2} + \epsilon }}\f$

* **Scale and shift**:

    \f$ o_{i} \leftarrow \gamma\hat{b_{i}} + \beta =
      BN_{\gamma ,\beta } ( b_{i} )\f$

## Attributes

* *epsilon*

  * **Description**: *epsilon* is the number to be added to the variance to
    avoid division by zero when normalizing a value. For example, *epsilon*
    equal to `0.001` means that `0.001` is added to the variance.
  * **Range of values**: arbitrary positive f32 value
  * **Type**: f32
  * **Required**: *yes*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and
    output data.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

## Inputs

* **1**: ``input`` - input tensor with data for normalization. The format is
  specified by *data_format*. **Required.**

  * **Type**: T1

* **2**: ``gamma`` - gamma scaling for normalized value. A 1D tensor with the
  same span as input's channel axis. **Required.**

  * **Type**: T2

* **3**: ``beta`` - bias added to the scaled normalized value. A 1D tensor with
  the same span as input's channel axis.. **Required.**

  * **Type**: T2

* **4**: ``mean`` - value for mean normalization. A 1D tensor with the same span
  as input's channel axis. **Required.**

  * **Type**: T2

* **5**: ``variance`` - value for variance normalization. A 1D tensor with the
  same span as input's channel axis.. **Required.**

  * **Type**: T2

## Outputs

* **1**: ``output`` - the result of normalization. A tensor of the same shape and
  format with 1st input tensor.

  * **Type**: T1

**Types**:

* *T1*: f32, f16, bf16.
* *T2*: f32, bf16.
* Constraints: *T2* can be bf16 only when *T1* is bf16.
