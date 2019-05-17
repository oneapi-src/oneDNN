Concat {#dev_guide_concat}
==========================

>
> API reference: [C](@ref c_api_concat), [C++](@ref cpp_api_concat)
>

The concat primitive concatenates \f$N\f$ tensors over `concat_axis` (here
designated as \f$C\f$ axis) and defined as:

\f[
    dst(\overline{ou}, c, \overline{in}) =
        src_i(\overline{ou}, c', \overline{in}),
\f]

where \f$c = C_1 + .. + C_{i-1} {}_{} + c'\f$.

The concat primitive doesn't have a notion of forward or backward propagations.
The backward propagation for the concatenation operation is simply an identity
operation.

## Implementation Details

### General Notes

1. The \f$dst\f$ memory format can be either specified by a user or derived by
   the primitive. The recommended way is to allow the primitive to choose the
   most appropriate format.

2. The concat primitive requires all source and destination tensors to have the
   same shape except for the `concat_axis`. The destination dimension for the
   `concat_axis` must be equal to the sum of the `concat_axis` dimensions of
   the sources (i.e. \f$C = \sum_i C_i\f$).
   Implicit broadcasting is not supported.

### Data Types Support

The concat primitive supports arbitrary data types for source and destination
tensors according to the @ref dev_guide_data_types page. However, it is
required that all source tensors are of the same data type (but not necessarily
matching the data type of the destination tensor).

### Data Representation

The concat primitive works with arbitrary data tensors. There is no special
meaning associated with any logical dimensions.

### Post-ops and Attributes

The concat primitive doesn't support any post-ops or attributes.


## Implementation Limitations

1. No primitive specific limitations. Refer to @ref dev_guide_data_types for
   limitations related to data types support.


## Performance Tips

1. Whenever possible, avoid specifying the destination memory format so that the
   primitive is able to choose the most appropriate one.

2. The concat primitive is highly optimized for the cases in which all source
   tensors have same memory format and data type matches the destination tensor
   data type. For other cases, more general but slower code is working.
   Consider reordering sources to the same data format before using the concat
   primitive.
