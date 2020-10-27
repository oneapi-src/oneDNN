---------
LayerNorm
---------

**Versioned name**: *LayerNorm-1*

**Category**: *Normalization*

**Short description**: `Reference <https://caffe.berkeleyvision.org/tutorial/layers/lrn.html>`__

**Attributes**:

* *keep_stats*

  * **Description**: *keep_stats* is used to indicate whether to output mean&&var. One typical usage is to pass mean&&var to backword op.
  * **Range of values**: False or True
  * **Type**: boolen
  * **Default value**: True
  * **Required**: *no*

* *begin_norm_axis*

  * **Description**: *begin_norm_axis* is used to indicate which axis to start layer normalization. The normalization is from *begin_norm_axis* to last dimension. Negative values means indexing from right to left. This op normalizes over the last dimension by default, e.g. C in TNC for 3D and LDNC for 4D.
  * **Range of values**: integer values
  * **Type**: int
  * **Default value**: -1
  * **Required**: *no*

* *use_affine*

  * **Description**: when set to True, this module has learnable per-element affine parameters. 
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


**Inputs**

* **1**: ``input`` - input tensor with data for normalization. **Required.**
* **2**: ``gamma`` - gamma scaling for normalized value. A 1D tensor of type T with the same span as input's channel axis. Required by attributs ``use_affine``. **Optional.**
* **3**: ``beta`` - bias added to the scaled normalized value. A 1D tensor of type T with the same span as input's channel axis.Required by attributs ``use_affine``. **Optional.**


**Outputs**

* **1**: ``output``  The result of normalization. A tensor of the same type and shape with 1st input tensor. **Required.**
* **2**: ``mean`` Output the mean calculated along the given axis. **Optional.**
* **3**: ``variance`` Output the std calculated along the given axis. **Optional.**

**Types**

* *T*: any numeric type.