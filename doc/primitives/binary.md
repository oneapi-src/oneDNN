Binary {#dev_guide_binary}
====================

>
> API reference: [C](@ref c_api_binary), [C++](@ref cpp_api_binary)
>

The binary primitive computes an operation between source 0 and source 1
element-wise:

\f[
    dst(\overline{x}) =
        src0(\overline{x}) \mathbin{op} src1(\overline{x}),
\f]

where \f$op\f$ is addition or multiplication.

The binary primitive does not have a notion of forward or backward propagations.

## Implementation Details

### General Notes

 * The binary primitive requires all source and destination tensors to have the
   same number of dimensions.

 * The binary primitive supports implicit broadcast semantics for source 1. It
   means that if some dimension has value of one, this value will be used to
   compute an operation with each point of source 0 for this dimension.

 * The \f$dst\f$ memory format can be either specified explicitly or be
   #dnnl::memory::format_tag::any (recommended), in which case the primitive
   will derive the most appropriate memory format based on the format of the
   source 0 tensor.

 * Destination memory descriptor should completely match source 0 memory
   descriptor.

 * The binary primitive supports in-place operations, meaning that source 0
   tensor may be used as the destination, in which case its data will
   be overwritten.


### Post-ops and Attributes

The binary primitive does not support any post-ops or attributes.

### Data Types Support

The source and destination tensors may have `f32` or `bf16` data types. See @ref
dev_guide_data_types page for more details.

### Data Representation

#### Sources, Destination

The binary primitive works with arbitrary data tensors. There is no special
meaning associated with any of tensors dimensions.


## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.


## Performance Tips

1. Whenever possible, avoid specifying the destination memory format so that
   the primitive is able to choose the most appropriate one.
