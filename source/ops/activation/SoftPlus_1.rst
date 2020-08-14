--------
SoftPlus
--------

**Versioned name**: *SoftPlus-4*

**Category**: *Activation*

**Short description**: SoftPlus takes one input tensor and produces output tensor where the SoftPlus function is applied to the tensor elementwise.

**Detailed description**: For each element from the input tensor calculates corresponding element in the output tensor with the following formula:

.. math::
  SoftPlus(x) = ln(e^{x} + 1.0)

**Inputs**:

* **1**:  Multidimensional input tensor of type T. **Required.**

**Outputs**

* **1**:  The resulting tensor of the same shape and type as input tensor. **Required.**

**Types**

* **T**:  arbitrary supported floating point type. **Required.**
