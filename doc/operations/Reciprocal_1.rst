----------
Reciprocal
----------

**Versioned name**: *Reciprocal-1*

**Category**: *Arithmetic*

**Short description**:Reciprocal is element-wise Power operation where exponent(power) equals to -1.
Reciprocal of 0 is infinity.

**Attributes**: *Reciprocal* operation has no attributes.

**Mathematical Formulation**

.. math::
   Reciprocal(x) = x^{-1}

   Reciprocal(0) = inf

**Inputs**:

* **1**: Multidimensional input tensor. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result tensor.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
