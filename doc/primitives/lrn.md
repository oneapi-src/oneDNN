Local Response Normalization (LRN) {#dev_guide_lrn}
====================================================

>
> API reference: [C](@ref c_api_lrn), [C++](@ref cpp_api_lrn)
>

The LRN primitive performs a forward or backward local response normalization
operation on 2D spatial data and is defined by the following formulas:

### Forward

LRN [across channels](#mkldnn_lrn_across_channels):

\f[
    dst(n, c, h, w) =
        \left\{k + \frac{\alpha}{n_{l}}
            \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2-1}
                (src(n, c+i, h, w))^2
        \right\}^{-\beta}
        \cdot
        src(n, c, h, w),
\f]

LRN [within channel](#mkldnn_lrn_within_channel):

\f[
    dst(n, c, h, w) =
        \left\{k + \frac{\alpha}{n_{l}}
            \sum\limits_{i=-(n_{l}-1)/2}^{(n_{l}+1)/2-1}
            \sum\limits_{j=-(n_{l}-1)/2}^{(n_{l}+1)/2-1}
                (src(n, c, h+i, w+j))^2
        \right\}^{-\beta}
        \cdot
        src(n, c, h, w),
\f]

where \f$n_{l}\f$ is the @p local_size.

### Backward

The backward propagation computes
\f$diff\_src(n, c, h, w)\f$,
based on
\f$diff\_dst(n, c, h, w)\f$ and \f$src(n, c, h, w)\f$.

## Implementation Details

### General Notes

1. During training, LRN might or might not require a workspace on forward and
   backward passes. The behavior is implementation specific. Optimized
   implementations typically require a workspace and use it to save some
   intermediate results from the forward pass that accelerate computations on
   the backward pass. To check whether a workspace is required, query the LRN
   primitive descriptor for the workspace. Success indicates that the workspace
   is required and its description will be returned.

2. The memory format and data type for `src` and `dst` are assumed to be the
   same, and in the API are typically referred as `data` (e.g., see `data_desc`
   in mkldnn::lrn_forward::desc::desc()). The same holds for `diff_src` and
   `diff_dst`. The corresponding memory descriptors are referred to as
   `diff_data_desc`.

### Data Type Support

The LRN primitive supports the following combinations of data types:

| Propagation        | Source / Destination |
| :--                | :--                  |
| forward / backward | f32                  |
| forward            | f16                  |

@warning
    There might be hardware and/or implementation specific restrictions.
    Check [Implementation Limitations](@ref dg_lrn_impl_limits) section below.

### Data Representation

#### Source, Destination, and Their Gradients

The LRN primitive supports only 2D spatial data. Like other CNN primitives, the
LRN primitive expects data to be \f$N \times C \times H \times W\f$ tensor.

The LRN primitive is optimized for the following memory formats:

| Spatial | Logical tensor | Implementations optimized for memory formats
| :--     | :--            | :--
| 2D      | NCHW           | #mkldnn_nchw (#mkldnn_abcd), #mkldnn_nhwc (#mkldnn_acdb), *optimized^*

Here *optimized^* means the format that
[comes out](@ref cpu_memory_format_propagation_cpp)
of any preceding compute-intensive primitive.

### Post-ops and Attributes

The LRN primitive doesn't support any post-ops or attributes.


@anchor dg_lrn_impl_limits
## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.


## Performance Tips

1. For backward propagation, use the same memory format for `src`, `diff_dst`,
   and `diff_src` (the format of the `diff_dst` and `diff_src` are always the
   same because of the API). Different formats are functionally supported but
   lead to highly suboptimal performance.
