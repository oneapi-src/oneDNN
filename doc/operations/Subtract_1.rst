--------
Subtract
--------

**Versioned name**: *Subtract-1*

**Category**: *Arithmetic*

**Short description**: *Subtract* performs element-wise subtraction operation with two
given tensors applying multi-directional broadcast rules.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/latest/openvino_docs_ops_arithmetic_Subtract_1.html>`__

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

* **1**: Tensor A. **Required.**

  * **Type**: T
  
* **2**: Tensor B. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result of element-wise subtraction operation. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

**Detailed description**:

Before performing arithmetic operation, input tensors *a* and *b* are
broadcast if their shapes are different and ``auto_broadcast`` attributes is
not ``none``. Broadcasting is performed according to ``auto_broadcast`` value.

After broadcasting *Subtract* does the following with the input tensors *a* and
*b*:

.. math::
   o_{i} = a_{i} - b_{i}
