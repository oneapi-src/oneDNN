----
Sqrt
----

**Versioned name**: *Sqrt-1*

**Category**: *Arithmetic*

**Short description**: *Sqrt* performs element-wise square root operation with
given tensor.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_arithmetic_Sqrt_1.html>`__

**Inputs**:

* **1**: An tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise sqrt operation. A tensor of type T.
  **Required.**

**Types**

* **T**: any numeric type.

*Sqrt* does the following with the input tensor *a*:

.. math::
   a_{i} = sqrt(a_{i})

