# GELU {#dev_guide_op_gelu}

**Versioned name**: *GELU-1*

**Category**: *Activation*

**Short description**:
[Reference](https://pytorch.org/docs/stable/nn.functional.html#gelu).

**Detailed description**: [Reference](https://arxiv.org/abs/1606.08415).

## Mathematical Formulation

:math:`GELU(x)=x*Φ(x)`, where :math:`Φ(x)` is the Cumulative Distribution
Function for Gaussian Distribution.

  \f$ GELU(x) = 0.5*x*(1.0 + erf((x) / \sqrt{2})\f$

The following approximation calculation (typical for the TensorFlow*) is
implementation defined behavior.

  \f$ GELU(x) \approx 0.5x(1.0 + tanh(\sqrt{2.0/pi}*(x+0.044715*x^3))\f$

## Inputs

* **1**: ``input`` - multidimensional input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - result of GELU function applied to the input tensor.
  **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
