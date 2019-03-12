# [RFC] Memory descriptor

## 0. About this document

This document describes suggested changes for the memory descriptor structure
that address several significant issues with the current approach.

The changes can be split into two categories:
- changes in the physical description that remove named formats;
- changes in the compensation description used, for example, for
  s8s8-convolutions.


## 1. Introduction

There are two main problems with the current memory description:
- Multi-level blocking cannot be described by the structure itself
  (current workaround is to use named formats).
- Compensation cannot be described by the structure itself
  (same workaround currently -- named formats).

There are a few other problems, mainly caused by named formats:
- Named formats lead to doubling of the number of the formats (and reorders)
  whenever a new format kind is added. For example adding a format for
  deconvolution that requires swapping the `o` and `i` dimensions leads to
  duplicating all double-blocked named formats (regular layouts can be
  replaced with `blocked` with swapped `o` and `i` entries).
- There is ambiguity in physical format description. The same physical format
  can be directly named (e.g. `nChw16c`) or be called `blocked` with identical
  blocking structure.
- There is no clear definition of named formats. For instance, for the format
  `nchw` it is not clear whether the tensor must be dense or may have
  non-trivial strides. If arbitrary strides are allowed, it is not clear what
  is the difference between `nchw` versus `nhwc` (assuming the strides are the
  same). This *duality* is misleading for the users and error-prone for the
  developers.
- No optimized primitives support non-trivial strides because they assume
  that the named format is dense or leads to dense structure, nor do they
  support the `blocked` layout format even if the physical structure describes
  exactly the dense `nChw16`. One consequence is that `view` is not supported
  on the primitive level; that is, there is no sense in creating convolution on
  viewed memory because Intel(R) MKL-DNN  would currently always fall back to a
  reference implementation even though the code itself may easily support
  some of the `view`s.

To resolve these issues, a new memory description is proposed (currently this
is just a concept to be finalized later):

``` c++
#define MKLDNN_MAX_NDIMS 12 // previously 16, 12 should be enough

// previously typeof(dims_t[0]) == int, now ptrdiff_t to handle big 1D tensors
typedef ptrdiff_t dims_t[MKLDNN_MAX_NDIMS];

typedef struct {
    // logical description of the tensor
    int ndims;              // number of logical dimension
    dims_t dims;            // the dimensions themselves
    data_type_t data_type;  // (main) data type

    // information about padded dimensions
    dims_t padded_dims;     // the dimensions of the parent tensor incl. padding
    dims_t padded_offsets;  // the offsets of parent tensor with padded tensor

    // basic offset (useful in the cases when there is no pointer arithmetic)
    ptrdiff_t offset0;      // to access the first element of the padded
                            // tensor, one should dereference ptr + offset0

    // physical description
    format_kind_t format_kind;  // { undef, any, blocking, wino }
    union {
        blocking_desc_t;    // must be able to handle multi-level blocking
        wino_desc_t;
    } format_desc;

    // section with *extra* information
    struct {
        uint64_t flags;     // flags contain arbitrary extra info,
                            // such as compensation
        float scale_adjust; // scale applied to the data (used on SKX)
        char reserved[64];  // for future backwards compatibility
    } extra;
} mkldnn_memory_desc_t;
```

The details are covered in the subsequent sections.


## 2. You can(not) avoid named formats

Blocking descriptor is changed so that it allows describing structure with
multi-level blocking. Named formats (such as `nchw`, `OIhw4i16o4i`) are no
longer used to define formats; common `mkldnn_blocked` should be used instead.

However, getting rid of named formats completely doesn't seem feasible:
- users need to express what format they use;
- the code internally uses named formats extensively (e.g. to set a memory
  descriptor from `any` to `nChw16c`), because this is convenient;
- for testing purposes: for instance, **benchdnn** can test batch normalization
  on a given format. Hence there should be a (convenient) way to set the
  format for testing.

Hence the idea is to separate the formats names and format kinds that specify
which structure is used for physical layout description. Format names (aka tags)
would be of type `mkldnn_format_tag_t` while the kind would be of type
`mkldnn_format_kind_t`. The latter can be only `{undef, any, blocked,
wino}`.

Named formats are used only to fill in the blocking structure, keeping the
`md.format_kind` equal to `mkldnn_blocked`.

Hereafter named formats would be called either *format tags* or simply *tags*
and refer to the `mkldnn_format_tag` type. The memory format kind that is used
in memory descriptor structure would be called *memory format kind*, or
*format kind* for short.

It is also important to remove any semantics that a former named format gave to
a tensor (for example, `nchw` implied that the tensor is data, while `oihw` is
used for weights; though for both named formats the blocking structure is filled
the same). To achieve this effect, we would use dimension-agnostic letters, such
as `a`, `b`, `c`, ... So `nchw` would be the same as `oihw`, and both would be
aliases to an abstract `abcd`. Similar, `nChw16c` and `oIhw16i` would be aliases
to `aBcd16b`.

``` c++
typedef enum {
    mkldnn_a,
    mkldnn_ab,
    mkldnn_ba,
    //...
    mkldnn_abcd,     // corresponds to nchw, oihw
    mkldnn_acdb,     // corresponds to nhwc, ohwi
    // ...
    mkldnn_aBcd16b,  // corresponds to nChw16c, oIhw16c
    // ...
    mkldnn_nchw = mkldnn_abcd,
    mkldnn_nChw16c = mkldnn_aBcd16b,
    // ...
} mkldnn_format_tag_t;
```

Previously, formats were used to allow the user to create Intel MKL-DNN memory.
With the new API, users would have two ways to describe their memory layout: the
generic one that takes the dimensions and strides between them (close to what
Intel MKL had), and another one that would be really close to the original
`mkldnn_memory_desc_init`, which takes dimensions and memory format tag.

``` c++
// inits memory desc for the given dims and strides between them
mkldnn_memory_desc_init_by_strides(mkldnn_memory_desc_t *md,
        int ndims, const dims_t dims, const dims_t strides,
        data_type_t data_type);

// inits memory desc for the given dims and memory format
// for those who is used to the previous versions
mkldnn_memory_desc_init_by_tag(mkldnn_memory_desc_t *md,
        int ndims, const dims_t dims, data_type_t data_type,
        mkldnn_format_tag format);
```

See additional discussion of usage of format tags after the next section, which
introduces the new blocking structure.


## 3. New blocking descriptor (for multi-level blocking)

*So far there is no good known solution...*

But the interim thought is to focus on the description of innermost blocks,
allowing outer blocks to have arbitrary strides. The blocking structure might
then look like:

``` c++
typedef struct {
    dims_t strides;     // the strides between the *major* dimensions, i.e.
                        // between `O`, `I`, `h`, and `w`, in `OIhw4i16o4i`

    // innermost section.
    // ASSUMPTION: the innermost blocks are always dense, unlike the *major* dimensions
    int inner_nblks;    // number of innermost blocks, e.g. 3 in case OIhw_4i16o4i_
    dims_t inner_blks;  // the size of blocks, {4, 16, 4} in case OIhw4i16o4i
    dims_t inner_idxs;  // the logical indices of the innermost blocks, e.g.
                        // {1, 0, 1} in case of 4i16o4i, because `i` is 1st dim
                        // and `o` is 0st dim
} blocking_desc_t;
```

There is a very important assumption that is applied along with the structure:
the innermost blocks are **always** dense. The *major* dimensions are allowed to have
arbitrary strides. That means if a convolution implementation works with
`nChw16c` format and it is really important that only `Chw16c` is dense while
`n` might have arbitrary stride, this implementation should check that:

``` c++
auto &bd = data_md.format_desc.blocking_desc;
book ok = true
    && bd.inner_nblks == 1
    && bd.inner_blks[0] == 16
    && bd.inner_idxs[0] == 1     // c is the dim with index 1
    && bd.strides[3] == 16       // stride for `w`
    && bd.strides[2] == bd.strides[3] * 16
    && bd.strides[1] == bd.strides[2] * md.dims[2];
    // bd.strides[0] might be whatever

    if (!ok) return unimplemented; // unsupported memory format
```

> **NOTE**
>
> There are many open questions regarding the suggested structure. One of them
> is more about details: for example, is it better to have `inner_blks` or to
> have the cumulative sizes (e.g. `{4*16*4, 16*4, 4}` instead of `{4, 16, 4}`)?
> Or is it better to keep the dimension in reverse order (at least for the
> innermost blocks)?

The proposed blocking descriptor restrictions:
- As already mentioned above, the innermost blocks must be dense.
- The blocks for a given dimension lie in order; that is, the elements with
  lower indices always appear before the elements with higher indices.
- One should parse the description of innermost blocks completely to identify
  the *major* dimensions; for example, it is not that easy to understand that
  `I` equals `i / 16` in `OIhw4i16o4i`, because this `16 = 4 * 4` is **not**
  explicitly written anywhere.
- The (minor) number of innermost blocks must be 12 or less.

### New blocking descriptor and format tags

As already mentioned above, a complete removal of named format seems illogical.
Intel MKL-DNN developers and users currently use named format in the following
situations:

- Users and tests use them to describe the format of the data they have.

The replacements for `mkldnn_memory_desc_init()` were already mentioned.
In most of the cases users can use `mkldnn_memory_desc_init_by_strides()` to
create memory descriptors with plain formats. Alternatively, we allow them to
directly use `mkldnn_memory_desc_init_by_tag()` to define the layout by
format tag, including the blocked one (like `nChw16c`). Ideally users should
not need to create memory in blocked format, although that may happen sometimes
(for example, see the Intel MKL-DNN integration in TensorFlow). Sometimes that
is done by Intel MKL-DNN tests as well (for example, benchdnn), so this
functionality seems useful.

- Convolution primitives use them to specify the format they want if the user
  specifies format tag `any`.

Currently Intel MKL-DNN has
``` c++
memory_primitive_desc_t::set_format(memory_format_t fmt)
```
method that specifies the memory descriptor structure based on the input
format. The replacement of that code would be:
``` c++
memory_primitive_desc_t::set_by_tag(memory_format_tag_t tag,
        const dims_t strides = nullptr)
```
which does the same but also supports arbitrary strides for the *major*
dimensions. It is not clear how useful it is to have the `strides` parameter,
but for RNN it may be used to set a good leading dimension.

Note that if `strides` are not specified, the `tag` is used to define the
strides for *major* dimensions (assuming dense data layout). However, if a
developer provides the `strides`, the `tag` is used to fill the innermost
part of the blocking structure only.

- Convolution and other primitives to check whether provided format is supported.

That is the weakest point of the current Intel MKL-DNN, because currently the
format is checked by name.  For example:

``` c++
    bool ok = src_md.format == mkldnn_nChw16c;
```

This code doesn't cover the case when the data layout corresponds to
`nChw16c` but the name of the format is `blocked`. For this legitimate case
the implementation would fail.

Another problem is that if `C` or `h` have non-trivial strides,
implementations do not check that, and most likely primitives would
compute an incorrect result.

The suggested replacement for the code above is:

``` c++
    // returns true if memory desc `md` corresponds to the given named format
    // (optionally) there might be provided strides for *major* dimensions
    // in which case the strides are also checked (value `-1` ignored).
    bool memory_desc_matches_tag(const memory_desc_t *md,
            memory_format_tag_t tag, const dims_t strides = nullptr);


    // example 1: an implementation works strictly with dense format `nChw8c`
    bool ok = memory_desc_matches_tag(&src_md, mkldnn_nChw8c, nullptr);

    // example 2: an implementation works with `nChw8c` where `Chw8c` must be
    //            dense, while `n` might have an arbitrary stride
    const int c = src_md.dims[1], h = src_md.dims[2], w = src_md.dims[3];
    bool ok = memory_desc_matches_tag(&src_md, mkldnn_nChw8c,
            {-1, h * w * 8, w * 8, 8});
```

> Suggestions on better names and internal API are very welcome.

In particular, this approach allows supporting views and even in-place
concat (though that requires some really skillful and inconvenient manipulations
with primitive creation).

- Non-JIT-ed reorder is template-ized by named memory format to implement
  reorders from plain to blocked formats with padded area.

There are multiple ways to port this code. One is to try to keep the current
structure with format tags. Exactly this approach was chosen for the
implementation.


## 4. Compensation

Having compensation as a first-class citizen (along with the blocking
structure) is essential for moving to a mamed-format free paradigm. Currently
we embed the knowledge about the compensation into the format name (for example,
`hwigo_s8s8`).

Since the proposal is to have very few format ids (`undef`, `any`, `blocked`,
and `wino`), we need to put the information about the compensation somewhere
else. That's exactly why we need an `extra` field in the memory descriptor.

``` c++
typedef struct {
    // flags contain arbitrary extra info, such as compensation, e.g.:
    // - MKLDNN_MDEF_NONE (0u)
    // - MKLDNN_MDEF_COMP_CONV_S8S8 (1u) - tensor contains compensation
    //                                     information (along what dim?)
    // - MKLDNN_MDEF_ADJ_SCALE (2u) - tensor is adjusted by the given value
    //                                (used in s8s8-conv and wino2x3)
    // - MKLDNN_MDEF_Q10N_PARAMS (?) - maybe somewhere in the distant future
    //                                 (that would mean scale and shift are
    //                                 kept right after the values)...
    uint64_t flags;
    float scale_adjust;
    char reserved[64];  // for future backwards compatibility
} mkldnn_md_extra_t;

struct mkldnn_memory_desc_t {
    // ...
    mkldnn_md_extra_t extra;
};
```

It seems we don't need to put a lot of information into this `extra` part.
The flags that would identify different kinds of compensations (e.g. int8
convolution, int8 RNN, something else) should be enough.

One more thing that makes sense to put in `extra` is `scale_adjust`, which is
used on SKX for int8 convolution (`s8s8` and `wino_2x3`) and indicates that
the weights are actually scaled (by `0.5`) due to a potential intermediate
overflow when the result is accumulated to `int16_t`. While the Winograd
implementation currently keeps this information in `wino_desc_t`, the
`s8s8`-convolution and corresponding reorder knows it by always checking

``` c++
    scale_adjust = mayiuse(avx512) && !mayiuse(avx512_vnni) ? 1.f / 2.f : 1.f;
```

which seems inelegant. Moreover it probably makes sense to expose this value to
the user.

Finally, the field `reserved` is a precautionary measure for future extensions
without breaking API / ABI.


## 5. Other

### 5.1 Reshape (?)

This proposal should also make it possible to increase/decrease the number of
dimensions in a tensor (instead of reshape?):

``` c++
// new_dims should be the same as dims, except for ones (there might be more
// ones or less). Example: `{1, 3, 16, 1} --> {3, 1, 1, 16, 1, 1}`
mkldnn_status_t mkldnn_memory_desc_change_dims(memory_desc_t *dst_md,
        const memory_desc_t *src_md, int new_ndims, const dims_t new_dims);
```
