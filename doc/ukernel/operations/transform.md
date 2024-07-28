Data transformation {#dev_guide_ukernel_transform}
=======================================

>
> [API Reference](@ref dnnl_api_ukernel_brgemm)
>

## General

The transform ukernel allows users to convert data from one format to the other,
similar to what reorder primitive provides functionally.

The only output data format supported by this routine is packed format, which is
required by B matrices in [BRGeMM ukernel](@ref dev_guide_ukernel_brgemm).
This is an out-of-place operation.

## Data Types

The transform ukernel does not allow data type conversion.

## Data Representation

| src  | dst  |
|:-----|:-----|
| f32  | f32  |
| f16  | f16  |
| bf16 | bf16 |
| s8   | s8   |
| u8   | u8   |

## Attributes

No attribute is supported for transform ukernel.

## Implementation limitations

- Destination leading dimension only supported values are: 16, 32, 48, or 64.

## Examples

[BRGeMM ukernel example](@ref cpu_brgemm_example_cpp)

@copydetails cpu_brgemm_example_cpp
