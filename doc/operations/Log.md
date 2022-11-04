# Log {#dev_guide_op_log}

**Versioned name**: *Log-1*

**Category**: *Arithmetic*

**Short description**: *Log* performs element-wise natural logarithm operation
with given tensor.

## Inputs

* **1**: ``input`` - input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - the output tensor of log operation.

  * **Type**: T

**Types**:

* *T*: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.

*Log* does the following with the input tensor *a*:

.. math::
   a_{i} = log(a_{i})
