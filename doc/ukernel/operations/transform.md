Data transformation {#dev_guide_ukernel_transform}
=======================================

>
> [API Reference](@ref dnnl_api_ukernel_brgemm)
>

## General

The packB ukernel allows users to pack BRGeMM B matrices in an optimal layout
before executing the [BRGeMM ukernel](@ref dev_guide_ukernel_brgemm). This is an
out-of-place operation.

## Data Types

The packB ukernel does not allow data type conversion.

## Data Representation

| src  | dst  |
|:-----|:-----|
| f32  | f32  |
| f16  | f16  |
| bf16 | bf16 |
| s8   | s8   |
| u8   | u8   |

## Attributes

No attribute is supported for packB ukernel.

## Implementation limitations

- Source leading dimension should be greater or equal to N to return the correct
  result.
- Destination leading dimension should be one of 16, 32, 48, or 64.

## Examples

[BRGeMM ukernel example](@ref cpu_brgemm_example_cpp)

@copydetails cpu_brgemm_example_cpp
