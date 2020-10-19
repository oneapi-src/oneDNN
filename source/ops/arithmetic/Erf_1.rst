---
Erf
---

**Versioned name**: *Erf-1*

**Category**: *Arithmetic*

**Short description**: *Erf* calculates the Gauss error function element-wise with given tensor.

**Detailed description:**

For each element from the input tensor calculates corresponding element in the output tensor with the following formula:
\f[
erf(x) = \pi^{-1} \int_{-x}^{x} e^{-t^2} dt
\f]

**Attributes**:

No attributes available.

**Inputs**

* **1**: A tensor of type T. **Required.**

**Outputs**

* **1**: The result of element-wise operation. A tensor of type T. **Required.**

**Types**

* *T*: any supported floating point type.



