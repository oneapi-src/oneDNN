---
Erf
---

**Versioned name**: *Erf-1*

**Category**: *Arithmetic*

**Short description**: *Erf* calculates the Gauss error function element-wise
with given tensor.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.1/openvino_docs_ops_arithmetic_Erf_1.html>`__

**Detailed description:**

For each element from the input tensor calculates corresponding element in the
output tensor with the following formula:

.. math::
   erf(x) = \pi^{-1} \int_{-x}^{x} e^{-t^2} dt

**Attributes**:

No attributes available.

**Inputs**

* **1**: Input tensor. **Required.**
  
  * **Type**: T

**Outputs**

* **1**: The result of element-wise operation. **Required.**
  
  * **Type**: T

**Types**

* *T*: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
