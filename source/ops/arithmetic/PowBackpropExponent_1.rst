-------------------
PowBackpropExponent
-------------------

**Versioned name**: *PowBackpropExponent-1*

**Category**: *Arithmetic*

**Short description**: *PowBackprop* computes gradient of exponent for Pow

**Inputs**:

* **1**: ``input_forward`` - input of forward. **Required.**
* **2**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**
* **3**: ``result_forward`` - original output of pow. **Required.**
* **4**: ``exponent`` - exponent of input. **Optional.**

**Outputs**

* **1**: ``exponent_delta`` - the gradient tensor w.r.t. the exponent of Pow.

