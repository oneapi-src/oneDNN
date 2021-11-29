.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

------------------
BatchNormInference
------------------

**Versioned name**: *BatchNormInference-1*

**Category**: *Normalization*

**Short description**: *BatchNormInference* layer normalizes a ``input`` tensor
by ``mean`` and ``variance``, and applies a scale (``gamma``) to it, as well as
an offset (``beta``).

**Attributes**:

* *epsilon*

  * **Description**: *epsilon* is the number to be added to the variance to
    avoid division by zero when normalizing a value. For example, *epsilon*
    equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: arbitrary positive f32 value 
  * **Type**: f32
  * **Default value**: None
  * **Required**: *yes*

* *data_format*

  * **Description**: *data_format* denotes the data format of the input and
    output data.
  * **Range of values**: *NXC* or *NCX* (X means HW for 2D, DHW for 3D)
  * **Type**: string
  * **Default value**: *NXC*
  * **Required**: *no*

**Inputs**

* **1**: ``input`` - input tensor with data for normalization. The format is
  specified by *data_format*. **Required.**

  * **Type**: T

* **2**: ``gamma`` - gamma scaling for normalized value. A 1D tensor with the
  same span as input's channel axis. **Required.**

  * **Type**: f32

* **3**: ``beta`` - bias added to the scaled normalized value. A 1D tensor with
  the same span as input's channel axis.. **Required.**

  * **Type**: f32

* **4**: ``mean`` - value for mean normalization. A 1D tensor with the same span
  as input's channel axis. **Required.**

  * **Type**: f32

* **5**: ``variance`` - value for variance normalization. A 1D tensor with the
  same span as input's channel axis.. **Required.**

  * **Type**: f32

**Outputs**

* **1**: The result of normalization. A tensor of the same shape and 
  format with 1st input tensor.

  * **Type**: T

**Types**

* *T*: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.

**Mathematical Formulation**

*BatchNormInference*  normalizes the output in each hidden layer.

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
     \sigma_{\beta }^{2}\leftarrow \frac{1}{m}\sum_{i=1}^{m} ( b_{i}
     - \mu_{\beta} )^{2}

* **Normalize**:

  .. math::
     \hat{b_{i}} \leftarrow \frac{b_{i}
     - \mu_{\beta}}{\sqrt{\sigma_{\beta }^{2} + \epsilon }}

* **Scale and shift**:

  .. math::
     o_{i} \leftarrow \gamma\hat{b_{i}}
     + \beta = BN_{\gamma ,\beta } ( b_{i} )

