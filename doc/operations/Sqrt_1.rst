----
Sqrt
----

**Versioned name**: *Sqrt-1*

**Category**: *Arithmetic*

**Short description**: *Sqrt* performs element-wise square root operation with
given tensor.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.1/openvino_docs_ops_arithmetic_Sqrt_1.html>`__

**Inputs**:

* **1**: An tensor of type T. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result of element-wise sqrt operation. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

*Sqrt* does the following with the input tensor *a*:

.. math::
   a_{i} = sqrt(a_{i})

