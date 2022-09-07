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
  * **Range of values**: [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: -1
  * **Required**: *no*

* *use_affine*

  * **Description**: *use_affine* when set to True, this module has learnable
    per-element affine parameters.
  * **Range of values**: False or True
  * **Type**: bool
  * **Default value**: True
  * **Required**: *no*

* *epsilon*

  * **Description**: *epsilon* is a constant to improve numerical stability
  * **Range of values**: arbitrary positive f32 value
  * **Type**: f32
  * **Default value**: 1e-5
  * **Required**: *no*


**Inputs**

* **1**: ``input_forward`` - input of forward. **Required.**

  * **Type**: T1

* **2**: ``output_delta`` - the gradient tensor with respect to the output of
  the layer normalization. **Required.**

  * **Type**: T1

* **3**: ``mean`` - mean of input_forward. **Required.**

  * **Type**: T2

* **4**: ``variance`` - variance of input_forward. **Required.**

  * **Type**: T2

* **5**: ``gamma`` - gamma scaling for normalized value. A 1D tensor with the
  same span as input's channel axis. Required by attribute ``use_affine``.
  **Optional.**

  * **Type**: T2

* **6**: ``beta`` - bias added to the scaled normalized value. A 1D tensor with
  the same span as input's channel axis. Required by attribute ``use_affine``.
  **Optional.**

  * **Type**: T2

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of the
  layer normalization. **Required.**

  * **Type**: T1

* **2**: ``gamma_delta`` - the gradient tensor with respect to the gamma of the
  layer normalization. **Optional.**

  * **Type**: T2

* **3**: ``beta_delta`` - the gradient tensor with respect to the beta of the
  layer normalization. **Optional.**

  * **Type**: T2

**Types**

* *T1*: f32, f16, bf16.
* *T2*: f32, bf16.
* Constraints: *T2* can be bf16 only when *T1* is bf16.
