---
Abs
---

**Versioned name**: *Abs-1*

**Category**: *Arithmetic*

**Short description**: *Abs* performs element-wise the absolute value with given tensor.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.4/openvino_docs_ops_arithmetic_Abs_1.html>`__

**Attributes**:

    No attributes available.

**Inputs**

* **1**: A tensor of type T. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result of element-wise abs operation.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
