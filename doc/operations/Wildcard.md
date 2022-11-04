# Wildcard {#dev_guide_op_wildcard}

**Versioned name**: *Wildcard-1*

**Category**: *Misc*

**Short description**: *Wildcard* operation represents any compute logic and its
  input and output tensors contribute to the graph building.

## Inputs

* **0 - N**: ``input`` - input tensors. **Optional.**

  * **Type**: Input tensors may have different data types from the list of s32,
    f32, f16, bf16, s8, u8.

## Outputs

* **0 - N**: ``output`` - output tensors. **Optional.**

  * **Type**: Output tensors may have different data types from the list of s32,
    f32, f16, bf16, s8, u8.
