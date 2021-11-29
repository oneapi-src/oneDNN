.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

---
Exp
---

**Versioned name**: *Exp-1*

**Category**: *Activation*

**Short description**: Exponential element-wise activation function.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_activation_Exp_1.html>`__

**Attributes**: has no attributes

**Inputs**:

* **1**: Input tensor x. **Required.**

  * **Type**: T

**Outputs**

* **1**: Result of Exp function applied to the input tensor x. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.
