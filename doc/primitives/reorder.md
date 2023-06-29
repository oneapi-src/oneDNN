Reorder {#dev_guide_reorder}
============================

>
> [API Reference](@ref dnnl_api_reorder)
>

## General

The reorder primitive copies data between different memory formats but doesn't
change the tensor from mathematical perspective (the variable names follow the
standard @ref dev_guide_conventions):

\f[
    \dst(\overline{x}) = \src(\overline{x})
\f]

As described in @ref dev_guide_basic_concepts in order to achieve the best
performance some primitives (such as convolution) require special memory format
which is typically referred to as an *optimized* memory format. The *optimized*
memory format may match or may not match memory format that data is currently
kept in. In this case a user can use reorder primitive to copy (reorder) the
data between the memory formats.

Using the attributes and post-ops users can also use reorder primitive to
quantize the data (and if necessary change the memory format simultaneously).

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index              |
|------------------------|---------------------------------------|
| \src                   | DNNL_ARG_FROM                         |
| \dst                   | DNNL_ARG_TO                           |
| \f$src scale\f$        | DNNL_ARG_ATTR_SCALES \| DNNL_ARG_FROM |
| \f$dst scale\f$        | DNNL_ARG_ATTR_SCALES \| DNNL_ARG_TO   |

## Implementation Details

### General Notes

1. The reorder primitive requires the source and destination tensors to have
   the same shape. Implicit broadcasting is not supported.

2. While in most of the cases the reorder should be able to handle arbitrary
   source and destination memory formats and data types, it might happen than
   some combinations are not implemented. For instance:

   - Reorder implementations between weights in non-plain memory formats might
     be limited (but if encountered in real practice should be treated as a
     bug and reported to oneDNN team);

   - Weights in one Winograd format cannot be reordered to the weights of the
     other Winograd format;

   - Quantized weights for convolution with #dnnl_s8 source data type cannot
     be dequantized back to the #dnnl_f32 data type;

3. To alleviate the problem a user may rely on fact that the reorder from
   original plain memory format and user's data type to the *optimized* format
   with chosen data type should be always implemented.

### Data Types Support

The reorder primitive supports arbitrary data types for the source and
destination according to the @ref dev_guide_data_types page.

When converting the data from one data type to a smaller one
saturation is used. For instance:

~~~
    reorder(src={1024, data_type=f32}, dst={, data_type=s8})
    // dst == {127}

    reorder(src={-124, data_type=f32}, dst={, data_type=u8})
    // dst == {0}
~~~

### Data Representation

The reorder primitive works with arbitrary data tensors. There is no special
meaning associated with any logical dimensions.

### Post-Ops and Attributes

The reorder primitive support the following attributes and post-ops:

| Attributes / Post-ops                                          | Meaning                                                      |
|:---------------------------------------------------------------|:-------------------------------------------------------------|
| [Scales](@ref dnnl::primitive_attr::set_scales_mask)           | Scales the corresponding tensor by the given scale factor(s) |
| [Zero points](@ref dnnl::primitive_attr::set_zero_points_mask) | Sets zero point(s) for the corresponding tensors             |
| [Sum post-op](@ref dnnl::post_ops::append_sum)                 | Instead of copy the data accumulate it to the previous data  |

For instance, the following pseudo-code

~~~
    reorder(
            src = {dims={N, C, H, W}, data_type=dt_src, memory_format=fmt_src},
            dst = {dims={N, C, H, W}, data_type=dt_dst, memory_format=fmt_dst},
            attr ={
                scales={ src={mask=0} },
                zero_points= { src={mask=0}, dst={mask=0} },
                post-ops = { sum={scale=beta} },
            })
~~~

would lead to the following operation:

\f[
    \dst(\overline{x}) =
            scale_{src} \cdot \src(\overline{x} - shift_{src}) +
            \beta  \cdot \dst(\overline{x}) + shift_{dst}
\f]

@note
    * The intermediate operations are being done using single precision
      floating point data type.
    * \f$scale_{src}\f$ and \f$scale_{dst}\f$ must be passed during execution runtime
      as a separate memory argument. Using \f$scale_{src}\f$ argument will lead to
      multiplication of tensor values by a scale value. Using \f$scale_{dst}\f$
      argument will lead to division of tensor values by a scale value.

## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. **CPU**
   - Reorders between bf16, f16 and s32 data types are not supported.

3. **GPU**
   - Only tensors of 6 or fewer dimensions are supported.
   - Runtime dimensions are not supported.

## Performance Tips

N/A

## Example

[Reorder Primitive Example](@ref reorder_example_cpp)

@copydetails reorder_example_cpp_short
