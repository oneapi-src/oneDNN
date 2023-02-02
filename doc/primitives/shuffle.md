Shuffle {#dev_guide_shuffle}
============================

>
> [API Reference](@ref dnnl_api_shuffle)
>

## General

The shuffle primitive shuffles data along the shuffle axis (here designated as
\f$C\f$) with group parameter \f$G\f$. If the shuffle axis is thought of as a
\f$(\frac{C}{G} \times G)\f$ matrix in row-major order, then the shuffle
operation transposes the shuffle axis to a \f$(G \times \frac{C}{G})\f$ matrix
in row-major order.

### Forward

The formal definition is as follows (variable names follow the standard
@ref dev_guide_conventions):

\f[
    \dst(\overline{ou}, c, \overline{in}) =
    \src(\overline{ou}, c', \overline{in})
\f]

where

- \f$c\f$ dimension is called a shuffle axis,
- \f$G\f$ is a `group_size`,
- \f$\overline{ou}\f$ is the outermost indices (to the left from shuffle axis),
- \f$\overline{in}\f$ is the innermost indices (to the right from shuffle axis), and
- \f$c'\f$ and \f$c\f$ relate to each other as define by the system:

\f[
    \begin{cases}
        c  &= u + v\frac{C}{G}, \\
        c' &= uG + v, \\
    \end{cases}
\f]

Here, \f$0 \leq u < \frac{C}{G}\f$ and \f$0 \leq v < G\f$.

#### Difference Between Forward Training and Forward Inference

There is no difference between the #dnnl_forward_training
and #dnnl_forward_inference propagation kinds.

### Backward

The backward propagation computes
\f$\diffsrc(\overline{ou}, c', \overline{in})\f$,
based on
\f$\diffdst(\overline{ou}, c, \overline{in})\f$.

Essentially, backward propagation is the same as forward propagation with
\f$G\f$ replaced by \f$C / G\f$.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index |
|------------------------|--------------------------|
| \src                   | DNNL_ARG_SRC             |
| \dst                   | DNNL_ARG_DST             |
| \diffsrc               | DNNL_ARG_DIFF_SRC        |
| \diffdst               | DNNL_ARG_DIFF_DST        |

## Data Types

The shuffle primitive supports the following combinations of data types:

| Propagation        | Source / Destination |
|:-------------------|:---------------------|
| forward / backward | f32, bf16, f16       |
| forward            | s32, s8, u8          |

@warning
    There might be hardware and/or implementation specific restrictions.
    Check the [Implementation Limitations](@ref dg_shuffle_impl_limits) section
    below.

## Data Layouts

The shuffle primitive works with arbitrary data tensors. There is no special
meaning associated with any logical dimensions. However, the shuffle axis is
typically referred to as channels (hence in formulas we use \f$c\f$).

Shuffle operation typically appear in CNN topologies. Hence, in the library the
shuffle primitive is optimized for the corresponding memory formats:

| Spatial | Logical tensor | Shuffle Axis | Implementations optimized for memory formats                       |
|:--------|:---------------|:-------------|:-------------------------------------------------------------------|
| 2D      | NCHW           | 1 (C)        | #dnnl_nchw (#dnnl_abcd), #dnnl_nhwc (#dnnl_acdb), *optimized^*     |
| 3D      | NCDHW          | 1 (C)        | #dnnl_ncdhw (#dnnl_abcde), #dnnl_ndhwc (#dnnl_acdeb), *optimized^* |

Here *optimized^* means the format that
[comes out](@ref memory_format_propagation_cpp)
of any preceding compute-intensive primitive.

### Post-Ops and Attributes

The shuffle primitive does not support any post-ops or attributes.

@anchor dg_shuffle_impl_limits
## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. **GPU**
   - Only tensors of 6 or fewer dimensions are supported.

## Performance Tips

N/A

## Example

[Shuffle Primitive Example](@ref shuffle_example_cpp)

@copydetails shuffle_example_cpp_short
