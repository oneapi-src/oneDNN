.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

-----------
PowBackprop
-----------

**Versioned name**: *PowBackprop-1*

**Category**: *Arithmetic*

**Short description**: *PowBackprop* computes gradient of variable base for Pow

**Inputs**:

* **1**: ``input_forward`` - input of forward. **Required.**
* **2**: ``output_delta`` - gradients tensor w.r.t. the output. **Required.**
* **3**: ``exponent`` - exponent of input. **Required.**

**Outputs**

* **1**: ``input_delta`` - the gradient tensor w.r.t. the input of Pow.
