.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

---------
LayerNorm
---------

**Versioned name**: *LayerNorm-1*

**Category**: *Normalization*

**Short description**: `Reference
<https://arxiv.org/abs/1607.06450>`__

**Attributes**:

* *keep_stats*

  * **Description**: *keep_stats* is used to indicate whether to output
    mean&&var. One typical usage is to pass mean&&var to backwords op.
  * **Range of values**: False or True
  * **Type**: bool
  * **Default value**: True
  * **Required**: *no*

* *begin_norm_axis*

  * **Description**: *begin_norm_axis* is used to indicate which axis to start
    layer normalization. The normalization is from *begin_norm_axis* to last
    dimension. Negative values means indexing from right to left. This op
    normalizes over the last dimension by default, e.g. C in TNC for 3D and
    LDNC for 4D.
  * **Range of values**: [-r, r-1] where r = rank(input)
  * **Type**: s64
  * **Default value**: -1
  * **Required**: *no*

* *use_affine*

  * **Description**: when set to True, this module has learnable per-element
    affine parameters. 
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

* **1**: ``input`` - input tensor with data for normalization. **Required.**

  * **Type**: T
  
* **2**: ``gamma`` - gamma scaling for normalized value. A 1D tensor with the
  same span as input's channel axis. Required by attribute ``use_affine``.
  **Optional.**

  * **Type**: f32

* **3**: ``beta`` - bias added to the scaled normalized value. A 1D tensor with
  the same span as input's channel axis.Required by attribute ``use_affine``.
  **Optional.**

  * **Type**: f32

**Outputs**

* **1**: ``output``  The result of normalization. A tensor of the same  shape
  with 1st input tensor. **Required.**

  * **Type**: T

* **2**: ``mean`` Output the mean calculated along the given axis. **Optional.**

  * **Type**: f32

* **3**: ``variance`` Output the std calculated along the given axis.
  **Optional.**

  * **Type**: f32

**Types**

* *T*: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
