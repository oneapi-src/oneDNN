# HardSwish {#dev_guide_op_hardswish}

**Versioned name**: *HardSwish-1*

**Category**: *Activation*

**Short description**: *HardSwish* element-wise activation function.

**Detailed description**: For each element from the input tensor, calculates
corresponding element in the output tensor with the following formula:

  \f$ HardSwish(x) = x * \frac{min(max(x + 3, 0), 6)}{6} \f$

The HardSwish operation was introduced in the article available
[here](https://arxiv.org/pdf/1905.02244.pdf).

## Inputs

* **1**: ``input`` - multidimensional input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - result of HardSwish function applied to the input tensor.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
