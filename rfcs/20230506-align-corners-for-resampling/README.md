# RFC: Align corners in resampling primitive

## Introduction

The resampling primitive in oneDNN does not align the corner pixels.
For example let us take an example in one dimension where we upsample a set of
3 points at (0.0, 1.0, 2.0) to 6 points. oneDNN would upsample them to
(0.0, 0.5, 1.0, 1.5, 2.0, 2.5). As seen in this example the left-most pixels in
the source and destination are aligned, but the right-most pixels are not
aligned.

In this RFC, we propose to implement an option in oneDNN to align the
corner pixels. The `align_corners` attribute will upsample the points in the
destination to (0.0, 0.4, 0.8, 1.2, 1.6, 2.0).

In linear resampling, without `align_corners`, we have
```
Fh = OW/IW
dst(n, c, oh, ow) = src(n, c, ih, iw)
ih = (oh + 0.5)/Fh - 0.5
```

With align corners, the formula changes to
```
Fh = (OW - 1)/(IW - 1)
dst(n, c, oh, ow) = src(n, c, ih, iw)
ih = (oh + 0.5 - 0.5)/Fh - 0.5 + 0.5
```

When vector of factors are given, those will be used instead of `Fh`.

This attribute is present in other frameworks like [pytorch](https://github.com/pytorch/pytorch/blob/3b966a6ce3d39122998a362c2b4cb95e34a79d0b/aten/src/ATen/native/UpSample.h#L34).

## Proposal

### Option 1

We propose adding resampling flags in the descriptor constructor
as currently there are no flags for the resampling primitive.

``` cpp
/// Flags for resampling primitive.
typedef enum {
    dnnl_resampling_flags_none = 0x0U,
    dnnl_align_corners = 0x1U,
} dnnl_resampling_flags_t;
```

``` cpp
/// Flags for resampling primitive.
enum class resampling_flags : unsigned {
    none = dnnl_resampling_flags_none,
    align_corners = dnnl_align_corners
};
```

``` cpp
...
primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &dst_desc,
                resampling_flags flags,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
...
primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const std::vector<float> &factors,
                const memory::desc &src_desc,
                resampling_flags flags,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
...
primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const std::vector<float> &factors,
                const memory::desc &src_desc, const memory::desc &dst_desc,
                resampling_flags flags,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
...
```

Pros:
- The possibility of using this information for each algorithm we support.
- Possibility to add additional flags for the resampling primitive in the future.
  For eg: `half_pixel_centers=False` flag.

Cons:
- API change for resampling primitive.
- In case of using C API to avoid ABI break we need to add another version of 
  init function(similar to pooling primitive - version v1 and v2).
- Need to check for combinations that we do not support.
  For eg: `align_corners=True` and nearest neighbor.


### Option 2

In this option we would add a new resampling primitive.

``` cpp
/// Flags for resampling primitive.
typedef enum {
    ...
    /// Nearest Neighbor Resampling Method
    dnnl_resampling_nearest = 0x2fff0,
    /// Linear Resampling Method
    dnnl_resampling_linear = 0x2fff1,
    /// Linear Resampling Method with Align Corners
    dnnl_resampling_linear_align_corners = 0x2fff3,
    ...
} dnnl_alg_kind_t;
```

Pros:
- No API change for resampling primitive

Cons:
- Adding more flags will lead to a combinatorial explosion of algorithms

### Recommendation
I would recommend Option 1 as the flag approach would enable adding more
flags to resampling primitives in the future.
