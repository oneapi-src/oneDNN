.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

--------
Divide
--------

**Versioned name**: *Divide-1*

**Category**: *Arithmetic*

**Short description**: *Divide* performs element-wise division  operation with
two given tensors applying multi-directional broadcast rules.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvinotoolkit.org/2021.1/openvino_docs_ops_arithmetic_Divide_1.html>`__

**Attributes**:

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input
    tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes should match
    * *numpy* - numpy broadcasting rules, aligned with ONNX Broadcasting.
      Description is available in `ONNX docs
      <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`__.

  * **Type**: string
  * **Default value**: "numpy"
  * **Required**: *no*

**Inputs**

* **1**: Dividend tensor. **Required.**
  
  * **Type**: T

* **2**: Divisor tensor. **Required.**
  
  * **Type**: T

**Outputs**

* **1**: The result of element-wise division operation.
  
  * **Type**: T
  
**Types**:

* **T**: f32, f16, bf16.
* **Note**: Tensors denoted with same data type symbol(such as *T*) have same
  data type. For example, if *T* is f32, all these tensors are f32 tensor.

**Detailed description**:

Before performing arithmetic operation, input tensors *a* and *b* are
broadcast if their shapes are different and ``auto_broadcast`` attributes is
not ``none``. Broadcasting is performed according to ``auto_broadcast`` value.

After broadcasting *Divide* does the following with the input tensors *a* and
*b*:

.. math::
   o_{i} = a_{i} / b_{i}
