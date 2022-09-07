---------
HardSwish
---------

**Versioned name**: *HardSwish-1*

**Category**: *Activation*

**Short description**: *HardSwish* element-wise activation function.

**OpenVINO description**: This OP is as same as `OpenVINO OP
<https://docs.openvino.ai/2021.4/openvino_docs_ops_activation_HSwish_4.html>`__

**Detailed description**: For each element from the input tensor, calculates
corresponding element in the output tensor with the following formula:

.. math::
   HardSwish(x) = x * \frac{min(max(x + 3, 0), 6)}{6}

The HardSwish operation was introduced in the article available
`here <https://arxiv.org/pdf/1905.02244.pdf>`__.

**Attributes**: *HardSwish* operation has no attributes.

**Inputs**:

* **1**: Multidimensional input tensor. **Required.**

  * **Type**: T

**Outputs**

* **1**: The result tensor. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
