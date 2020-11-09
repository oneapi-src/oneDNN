-----------------
LayerNormBackprop
-----------------

**Versioned name**: *LayerNormBackprop-1*

**Category**: *Normalization*

**Short description**: `Reference
<https://arxiv.org/abs/1607.06450>`__

**Attributes**:

* *begin_norm_axis*

  * **Description**: *begin_norm_axis* is used to indicate which axis to perform
    layer normalization. The normalization is from *begin_norm_axis* to last
    dimension. Negative values means indexing from right to left. The default is
    last dimension.
  * **Range of values**: integer values
  * **Type**: int
  * **Default value**: -1
  * **Required**: *no*

* *use_affine*

  * **Description**: *use_affine* when set to True, this module has learnable
    per-element affine parameters.
  * **Range of values**: False or True
  * **Type**: boolen
  * **Default value**: True
  * **Required**: *no*

* *epsilon*

  * **Description**: *epsilon* is a constant to improve numerical stability
  * **Range of values**: a positive floating-point number
  * **Type**: float
  * **Default value**: 1e-5
  * **Required**: *no*

* *use_stats*

  * **Description**: *use_stats* is used to indicate whether to use input mean
    and variance.
  * **Range of values**: False or True
  * **Type**: ``bool``
  * **Default value**: true
  * **Required**: *no*


**Inputs**

* **1**: ``input_forward`` - input tensor. **Required.**
* **2**: ``gamma`` - gamma scaling for normalized value. A 1D tensor of type T
  with the same span as input's channel axis. Required by attributs
  ``use_affine``. **Optional.**
* **3**: ``beta`` - bias added to the scaled normalized value. A 1D tensor of
  type T with the same span as input's channel axis.Required by attributs
  ``use_affine``. **Optional.**
* **4**: ``mean`` - mean of input. Required by attributs ``use_stats``.
  **Optional.**
* **5**: ``variance`` - variance of input. Required by attributs ``use_stats``.
  **Optional.**

**Outputs**

* **1**: ``input_delta`` - the the gradient tensor w.r.t. the output of the
  layer normolization. **Required.**
* **2**: ``gamma_delta`` - the the gradient tensor w.r.t. the gamma of the layer
  normolization. **Optional.**
* **3**: ``beta_delta`` - the the gradient tensor w.r.t. the beta of the layer
  normolization. **Optional.**

**Types**

* *T*: any numeric type.
