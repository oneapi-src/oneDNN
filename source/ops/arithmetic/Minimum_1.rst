-------
Minimum
-------

**Versioned name**: *Minimum-1*

**Category**: *Arithmetic*

**Short description**: *Minimum* performs element-wise minimum operation with two given tensors applying multi-directional broadcast rules.

**Attributes**:

* *auto_broadcast*

  * **Description**: specifies rules used for auto-broadcasting of input tensors.
  * **Range of values**:

    * *none* - no auto-broadcasting is allowed, all input shapes should match
    * *numpy* - numpy broadcasting rules, aligned with ONNX Broadcasting. Description is available in `ONNX docs <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`__.

  * **Type**: string
  * **Default value**: "numpy"
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type T. **Required.**
* **2**: A tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise minimum operation. A tensor of type T.

**Types**

* *T*: any numeric type.

**Detailed description:**

Before performing arithmetic operation, input tensors *a* and *b* are broadcasted if their shapes are different and ``auto_broadcast`` attributes is not ``none``. Broadcasting is performed according to ``auto_broadcast`` value.

After broadcasting *Add* does the following with the input tensors *a* and *b*:

.. math::
   o_{i} = min(a_{i}, b_{i})
