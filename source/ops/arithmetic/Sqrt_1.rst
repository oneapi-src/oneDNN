----
Sqrt
----

**Versioned name**: *Sqrt-1*

**Category**: *Arithmetic*

**Short description**: *Sqrt* performs element-wise square root operation with given tensor.

**Inputs**:

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise sqrt operation. A tensor of type T. **Required.**

**Types**

* **T**: any numeric type. **Required.**

Sqrt does the following with the input tensor a:

.. math::
   a_{i} = sqrt(a_{i})
