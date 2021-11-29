.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

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

* *use_stats*

  * **Description**: *use_stats* is used to indicate whether to use input mean
    and variance.
  * **Range of values**: False or True
  * **Type**: ``bool``
  * **Default value**: true
  * **Required**: *no*


**Inputs**

* **1**: ``input_forward`` - input tensor. **Required.**

  * **Type**: T

* **2**: ``gamma`` - gamma scaling for normalized value. A 1D tensor of the same
  span as input's channel axis. Required by attribute ``use_affine``.
  **Optional.**

  * **Type**: f32

* **3**: ``beta`` - bias added to the scaled normalized value. A 1D tensor with
  the same span as input's channel axis.Required by attribute ``use_affine``.
  **Optional.**

  * **Type**: f32

* **4**: ``mean`` - mean of input. Required by attribute ``use_stats``.
  **Optional.**

  * **Type**: f32

* **5**: ``variance`` - variance of input. Required by attribute ``use_stats``.
  **Optional.**

  * **Type**: f32

**Outputs**

* **1**: ``input_delta`` - the the gradient tensor with respect to the output of
  the layer normalization. **Required.**

  * **Type**: T
  
* **2**: ``gamma_delta`` - the the gradient tensor with respect to the gamma of
  the layer normalization. **Optional.**

  * **Type**: f32

* **3**: ``beta_delta`` - the the gradient tensor with respect to the beta of
  the layer normalization. **Optional.**

  * **Type**: f32

**Types**

* *T*: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
