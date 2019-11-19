Pooling {#dev_guide_pooling}
============================

>
> [API Reference](@ref dnnl_api_pooling)
>

The pooling primitive performs forward or backward max or average pooling
operation on 2D or 3D spatial data.

The pooling operation is defined by the following formulas.
We show formulas only for 2D spatial data which are straightforward to
generalize to cases of higher and lower dimensions. Variable names follow the
standard @ref dev_guide_conventions.

### Forward

Max pooling:

\f[
    dst(n, c, oh, ow) =
        \max\limits_{kh, kw}
        \left(
            src(n, c, oh \cdot SH + kh - ph_0, ow \cdot SW +kw - pw_0)
        \right)
\f]

Average pooling:

\f[
    dst(n, c, oh, ow) =
        \frac{1}{DENOM}
        \sum\limits_{kh, kw}
            src(n, c, oh \cdot SH + kh - ph_0, ow \cdot SW +kw - pw_0)
\f]

where \f$ph_0, pw_0\f$ are `padding_l[0]` and `padding_l[1]` respectively,
and output spatial dimensions are calculated similarly to
how they are done in convolution.

Average pooling supports two algorithms:
- #dnnl_pooling_avg_include_padding, in which case \f$DENOM = KH \cdot KW\f$,
- #dnnl_pooling_avg_exclude_padding, in which case \f$DENOM\f$ equals to the
  size of overlap between an averaging window and images.

> TODO: a picture would be nice here.

#### Difference Between Forward Training and Forward Inference

- Max pooling requires `workspace` output for the #dnnl_forward_training
  propagation kind, and doesn't require it for #dnnl_forward_inference
  (see details below).

### Backward

The backward propagation computes
\f$diff\_src(n, c, h, w)\f$,
based on
\f$diff\_dst(n, c, h, w)\f$ and (in case of max pooling) `workspace`.

## Implementation Details

### General Notes

1. During training, max pooling requires a workspace on forward
   (#dnnl_forward_training) and backward passes to save indices where a
   maximum was found. The workspace format is opaque, and the indices cannot be
   restored from it. However, one can use backward pooling to perform
   up-sampling (used in some detection topologies).

2. A user can use memory format tag #dnnl_format_tag_any for `dst` memory
   descriptor when creating pooling forward propagation. The library would
   derive the appropriate format from the `src` memory descriptor. However,
   the `src` itself must be defined. Similarly, a user can use memory format tag
   #dnnl_format_tag_any for the`diff_src` memory descriptor when creating
   pooling backward propagation.

### Data Type Support

The pooling primitive supports the following combinations of data types:

| Propagation        | Source / Destination | Accumulation data type (used for average pooling only)
| :--                | :--                  | :--
| forward / backward | f32, bf16            | f32
| forward            | f16                  | f16
| forward            | s8, u8, s32          | s32

@warning
    There might be hardware and/or implementation specific restrictions.
    Check [Implementation Limitations](@ref dg_pool_impl_limits) section below.

### Data Representation

#### Source, Destination, and Their Gradients

Like other CNN primitives, the pooling primitive expects data
to be \f$N \times C \times H \times W\f$ tensor in case 2D spatial data
and \f$N \times C \times D \times H \times W\f$ tensor in case 3D spatial data.

The pooling primitive is optimized for the following memory formats:

| Spatial | Logical tensor | Data type   | Implementations optimized for memory formats                               |
| :--     | :--            | :--         | :--                                                                        |
| 2D      | NCHW           | f32         | #dnnl_nchw (#dnnl_abcd), #dnnl_nhwc (#dnnl_acdb), *optimized^*     |
| 2D      | NCHW           | s32, s8, u8 | #dnnl_nhwc (#dnnl_acdb), *optimized^*                                  |
| 3D      | NCDHW          | f32         | #dnnl_ncdhw (#dnnl_abcde), #dnnl_ndhwc (#dnnl_acdeb), *optimized^* |
| 3D      | NCDHW          | s32, s8, u8 | #dnnl_ndhwc (#dnnl_acdeb), *optimized^*                                |

Here *optimized^* means the format that
[comes out](@ref memory_format_propagation_cpp)
of any preceding compute-intensive primitive.

### Post-ops and Attributes

The pooling primitive doesn't support any post-ops or attributes.


@anchor dg_pool_impl_limits
## Implementation Limitations

1. No primitive specific limitations. Refer to @ref dev_guide_data_types for
   limitations related to data types support.


## Performance Tips

N/A
