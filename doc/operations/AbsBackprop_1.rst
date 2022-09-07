-----------
AbsBackprop
-----------

**Versioned name**: *AbsBackprop-1*

**Category**: *Arithmetic*

**Short description**: *AbsBackprop* computes gradient for Abs

.. math::
   ds = \begin{cases}
      dd & \text{if } s>0 \\
      -dd & \text{if } s<0 \\
      0 & \text{if } s=0
    \end{cases} 

**Inputs**:

* **1**: ``input_forward`` - input of forward. **Required.**
  
  * **Type**: T

* **2**: ``output_delta`` - gradients tensor with respect to the output.
  **Required.**

  * **Type**: T

**Outputs**

* **1**: ``input_delta`` - the gradient tensor with respect to the input of
  Abs.

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
