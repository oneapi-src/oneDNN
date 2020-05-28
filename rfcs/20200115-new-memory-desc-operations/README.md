Proposal for Adding New Memory Descriptor Operations
====================================================

## 1. Introduction

This RFC touches upon the memory-related operations that do not perform
any mathematical operations on tensors, but merely change their logical
representation.
This includes adding or removing the axes in tensors (of size 1), permuting the
logical axes, and tensor reinterpretation
(aka [NumPy reshape](https://docs.scipy.org/doc/numpy-1.3.x/reference/generated/numpy.reshape.html)).

The RFC starts with the discussion on the difference between the way DNNL and
frameworks represent and understand tensors / memory, and some consequences of
this difference. Next, the new operations are proposed, including the
motivation, API and behavior definitions. Finally, there are some notes on the
operations that are proposed to not be included (at least publicly).

### 1.1. Executive Summary

- Extend currently existing **reshape** function to support splitting single
  dimension into multiple ones or vice versa, collapsing several dimensions
  into a single one (restrictions apply).
  The most important usage scenario of this function is to change the number of
  dimensions in a tensor by adding or removing the dimensions of size `1`.
  The function changes the memory descriptor only, the data is kept untouched.
  - Applications: binary primitive, layer normalization, mimic _flatten_.
- Add **permute_axes** function that permutes the logical axes and applies the
  corresponding changes to the physical memory description (aka blocking
  descriptor). As in the previous case, the data is kept untouched.
  - Applicability: deconvolution, generic framework integration.
- Do **not** add anything else. In particular, **extended reshape** function
  that would allow relaxing some of the restrictions of the **reshape** above
  as it is very error-prone.


## 2. Tensor Representation Now and Then, Here and There

Moved to a [separate file](tensor-representation.md).
The section explains why memory related operations in DNNL should be used by
frameworks with extreme caution.
The takeaways are that the functions suggested here can be used in very narrow
context, mostly for convenience inside non-memory-related operations like
binary and layer normalization. With the integrations we have today, the
functions could not be used to perform framework operations with similar names.
For instance, for framework reshape operation there is zero usefulness in the
suggested functions.
Finally, the section ends with inspirational conclusion about complexity and
robustness of DNNL integration to frameworks.


## 3. New or Extended Memory Descriptor Operations

### 3.1. Reshape

The purpose is to collapse or split axes:
- `({a, b, c * d}, fmt=aBc8b)` --> `({a, b, c, d}, fmt=aBcd8b)`
- `({a, b, c}, fmt=bca)` --> `({a, b * c}, fmt=ba)`

A particular (and maybe the most useful) case is to add or remove axes of size
`1`:
- `{a, b, c}` --> `{a, 1, b, 1, 1, c, 1}`
- `{a, 1, b, c, 1}` --> `{a, b, c}`

The function is already available in the library with limited functionality:
it currently supports appending the axes of size `1` to the right.

#### 3.1.1. Motivation

- Binary supports broadcast semantics, assuming that both inputs have the same
  number of dimensions and the dimensions for broadcast are of size `1`.
  Frameworks have relaxed requirement and support broadcast for buffers with
  less dimensions. For these cases the integration code might have to
  add axes of size `1` explicitly to satisfy DNNL requirements.

- Layer normalization expects mean and variance tensors to have one dimension
  less than the data tensor. However, some frameworks want mean and variance to
  have the same number of dimensions, but with the last one being equal to `1`.
  So **reshape** could help to ping-pong the memory descriptors according to
  their needs.

- Sometimes a framework may want to join multiple axes and consider them as one
  contiguous axis (with DNNL this requires additional caution).

- The functionality can be useful for the internal development to reduce the
  number of dimensions when they don't matter.
  - For instance, no matter if data is 1D, 2D, or 3D, all batch normalization
    primitives treat the spatial dimension as a single entity. However, the
    code has to calculate the offsets according the logical number of
    dimensions. Hence, we often have the constructions like:
    ``` cpp
    dim_t offset = is_1d ? md.blk_off(n, c, w) :
                   is_2d ? md.blk_off(n, c, h, w) :
                   is_3d ? md.blk_off(n, c, d, h, w) : ERROR;
    ```
    The functionality could help easily convert the incoming memory descriptor
    to a `{batch, channels, cumulative-spatial-dim}` case, effectively make the
    implementation to work with 1D case only.

#### 3.1.2. API

``` cpp
dnnl_status_t
dnnl_memory_desc_reshape(
        dnnl_memory_desc_t *out_memory_desc,        // output memory desc, write only. in-place supported
        const dnnl_memory_desc_t *in_memory_desc,   // input memory desc, read only
        int new_ndims, const dnnl_dims_t new_dims   // new dimensions
        );
```

#### 3.1.3. Behavior

- Supported memory formats in `in_memory_desc`: `blocked` and `any`.
  - An attempt to pass any other memory format leads to `unimplemented` status
    except for the memory format `undef`, in which case `invalid_arguments` is
    returned.

- Assuming that only one axis of size `1` at the position `d` is added, the
  stride for this dimension is defined by the following formula:
  ``` cpp
  dim_t stride(in_memory_desc, d) {
      md = in_memory_desc;
      md_blocks = in_memory_desc.format_desc.blocking.inner_blks;

      if (d >= in_memory_desc.ndims) return product(md_blocks);

      cumulative_block_size_of_d = product(md_blocks for which dim_idx == d);
      return md.padded_dim[d] / cumulative_block_size_of_d * md.stride[d];
  }
  ```
  In terms of the physical format this approach *attaches* the newly added axis
  to the axis on the right. Examples:
    1. `({n, c, h, w}, fmt=nChw16c)` --> `({n, c, 1, h, w}, fmt=nCdhw16c)`,
       i.e. `d` axis is physically adjacent to the `h` axis. Note, that in a
       sense this is just a coincidence that the added `1` is treated as `d`
       and the format seems to match perfectly what we wanted.
    2. `({o, i, h, w}, fmt=hwio)` --> `({1, o, i, h, w}, fmt=hwigo)`,
       i.e. `g` axis is physically adjacent to the `o` axis. The same
       *coincidence* happens as above.

  Of course, this is not a reliable and absolute solution, but in most of the
  cases the behavior is desired. In the worst case, the verbose output would be
  slightly misleading, but there should be no logical issues.
  Example:
    1. `({o, i, h, w}, fmt=IOhw8i8o)` --> `({g=1, o, i, h, w}, fmt=IgOhw8i8o)`,
       while one might expect to get `({g=1, o, i, h, w}, fmt=gIOhw8i8o)`.
       - The different format is caused by the stride attached to the `g`
         dimension. According to the function above, the stride for `g` will be
         `O*h*w*8*8`, hence in the format it appears between `I` and `O`.
         However, if stride would be equal to `I*O*h*w*8*8` the format would be
         depicted as the latter one. It is worth mentioning, that no matter
         what the stride will be, the memory descriptors should be treated as
         equal.  The drawback, however, is that equal memory descriptors might
         produce different outputs for primitives like resampling (when the
         dimension of size `1` will be enlarged to `2` or greater, in which
         case the stride will significantly affect the physical data
         representation).
       - IMPORTANT: there is no right or wrong answer here. It is only about
         possible expectations on users' side.

- Remove axes of size `1` only when the corresponding padded dimensions equal
  to `1` as well. An attempt to remove axis which padded dimension is greater
  than `1` will lead to `invalid_arguments` status.

- If multiple dimensions of size `1` happen to be in a row, remove the leftmost
  one.
  - Open question: maybe prioritize the one with the smallest stride? This
    gives no obvious benefit, except for some (potential) convenience at a
    price of more complex code. Again, there is no right or wrong behavior here
    no matter which axis of size `1` will be chosen -- both would conceptually
    return the same logical tensor. The problem arises on the boundaries of
    logical tensors and DNNL way of their physical representation.

- Both `in_memory_desc.dims` and `new_dims` are split in the maximum disjoint
  groups of consequent dimensions, so that the product of dimension in i-th
  nontrivial group of `in_memory_desc.dims` is equal to the product of the
  dimensions in i-th nontrivial group of `new_dims`. The group is called
  trivial if it consists of one dimension equal `1`. For each i-th group pair
  one of the following 2 conditions must hold:
  - Either both consist of one dimension, or
  - For the axes in `in_memory_desc.dims` group all the conditions hold:
    - The padded dimensions equal the logical dimensions,
    - The dimensions are not blocked,
    - The physical order of the dimensions must match the logical order,
    - The physical dimensions must be dense (the stride times the logical
      dimension equals to the stride of the next dimension within the group).

  Some examples:
  ``` cpp
  // Note: initial format abcd
  {
    Reshape(({2, 3, 4, 5}, fmt=abcd), new_dims={6, 2, 2, 5}) --> ({6, 2, 2, 5}, fmt=abcd); // oK

    Reshape(({2, 3, 4, 5}, fmt=abcd), new_dims={6, 2, 10}) --> ({6, 2, 10}, fmt=abc); // oK
  }

  // Note: initial format dabc
  {
    Reshape(({2, 3, 4, 5}, fmt=dabc), new_dims={6, 2, 2, 5}) --> ({6, 2, 2, 5}, fmt=dabc); // oK

    // The following will fail, as in_memory_desc.dims group {4, 5} that corresponds
    // to {2, 10} group of new_dims doesn't have the proper physical order of
    // dimensions (they even not contiguous in memory).
    Reshape(({2, 3, 4, 5}, fmt=dabc), new_dims={6, 2, 10}) --> invalid_arguments
  }

  // Note: initial format abdc
  {
    Reshape(({2, 3, 4, 5}, fmt=abdc), new_dims={6, 4, 5}) --> ({6, 4, 5}, fmt=acb); // oK

    Reshape(({2, 3, 4, 5}, fmt=abdc), new_dims={6, 2, 2, 5}) --> ({6, 2, 2, 5}, fmt=adbc); // oK

    // The following will fail, as in_memory_desc.dims group {4, 5} that corresponds
    // to {20} group of new_dims doesn't have the proper physical order of
    // dimensions (even though they are contiguous in memory).
    Reshape(({2, 3, 4, 5}, fmt=abdc), new_dims={2, 3, 20}) --> invalid_arguments
  }
  ```

#### 3.1.3.1. Restrictions

- Memory format must be `blocked` or `any`.

- The product of dimensions in `in_memory_desc.dims` must be the same as
  the product of `new_dims`.

- Adding one or more axes of size `1` is always possible.

- Removing one or more axes of size `1` is always possible if and only if the
  corresponding padded dimensions equal to 1 as well.

- Joined axes must be physically consecutive in memory. Moreover, their
  physical order must match their logical order. Finally, the padded dimensions
  must be equal to the logical dimensions for joined axes.

- Split axes cannot be physically blocked. The padded dimensions must be equal
  to the logical dimensions for split axes.

#### 3.1.4. Discussion

The complicated restrictions on splitting and joining axes are dictated by the
requirement of the data consistency, as well as a desire to make function
behavior meet users' expectation. This is especially relevant for the joining
axes.

Consider the following inner product example:
``` cpp
// The src and weights tensors are incompatible with each other.
// Simply calling GEMM would not produce the correct result.
src_4d     = ({N, I, H, W}, fmt=nhwc)
weights_4d = ({O, I, H, W}, fmt=oihw)

// Thoughtless collapsing {I, H, W} dimension will hide the incompatibility
// issue:
src_2d     = reshape(src_4d,     {N, I * H * W}) = ({N, I * H * W}, fmt=nc)
weights_2d = reshape(weights_4d, {O, I * H * W}) = ({O, I * H * W}, fmt=oi)

InnerProduct(src_4d, weights_4d) != InnerProduct(src_2d, weights_2d)
```

The drawback is that the function might fail though. The hope is that users
will mostly add or removes axes of size `1` which works in most of the cases
just fine.


### 3.2. Permute / Rename Axes

The purpose is to remap / permute / rename the axes **keeping the data
unchanged**. This requires changing the physical memory description
according to the axes permutation.

Few examples:
- `({a, b, c}, fmt=abc)` with `(a->b, b->a, c->c)` permutation -->
  `({b, a, c}, fmt=bac)`
- `({a, b, c, d}, fmt=aBcd8b)`, with `(a->c, b->a, c->d, d->b)` permutation -->
  `({c, a, d, b}, fmt=cAdb8a)`.
  - This example shows the required permutation on DNNL side if a user will
    pass an activations tensor as a weights tensor in TensorFlow:

    DNNL activations | TF activations | TF weights | DNNL weights | Permutation
    ---------------- | -------------- | ---------- | ------------ | -----------
    a                | N              | H          | c            | a -> c
    c                | H              | W          | d            | c -> d
    d                | W              | I          | b            | d -> b
    b                | C              | O          | a            | b -> a

#### 3.2.1. Motivation

- The axes renaming may be extremely useful to reinterpret the axes because of
  the transition between *framework* world and DNNL one.
  - One of the examples is shown above.
  - Another one: imagine a user creates a framework buffer of a certain shape.
    Until this buffer comes into some operation the integration code cannot
    reliably map the axes to the DNNL ones, because depending on the context
    the dimensions are treated differently (say `(2, 3, 4, 5)` in TF world
    should be `dims=(2, 5, 3, 4)` in DNNL world if it is an activations tensor,
    but `dims=(5, 4, 2, 3)` if it is a weights tensor. So, the best thing the
    framework integration code could do at the point of buffer creation is to
    use the same order of dimensions in DNNL as in framework. Later, perform
    the axes permutation when there would be enough context for that.
- In frameworks the deconvolution typically uses inverse meaning of `O` and `I`
  axes of the weights to make it easier to use backward convolution: `O` stands
  for input channels, and `I` stands for output channels. In DNNL the axes
  follow the common approach: `O` is for output channels, and `I` is for input
  channels. This makes it harder for frameworks to describe the format of the
  weights in their native layout. With *permute* function they could create a
  memory descriptor for weights with `O` and `I` axes following their
  definition, but then permute them to comply with DNNL requirements.
- Internal deconvolution code could use permutation auxiliary function to
  easily adapt its weights to the corresponding convolution one by remapping
  logical `I` axis to `O` axis, and vice versa (conceptually, similar to the
  bullet above).

#### 3.2.2. API

**Warning** The name is subject for a discussion (as usual).

``` cpp
dnnl_status_t
dnnl_memory_desc_permute_axes(
        dnnl_memory_desc_t *out_memory_desc,        // output memory desc, write only. in-place supported
        const dnnl_memory_desc_t *in_memory_desc,   // input memory desc, read only
        const int *perm                             // permutation
        );
```
> **Permute** in the name means we move the axis `i` to the position according
> to the permutation given, i.e. `perm[i]`.
>
> In other words:
> ```
> in dim | out dim
> -------|--------
> i      | perm[i]
> ```

The alternative API:

``` cpp
dnnl_status_t
dnnl_memory_desc_rename_axes(
        dnnl_memory_desc_t *out_memory_desc,        // output memory desc, write only. in-place supported
        const dnnl_memory_desc_t *in_memory_desc,   // input memory desc, read only
        const int *renamed_axes                     // renamed axes in order -- this should be the INVERSE of the perm above!
        );
```

> **Rename** in the name means we rename each axis `i` according to the
> `renamed_axes` array (basically, the `renamed_axes` array gives the new
> order of the axes).
>
> In other words:
> ```
> in dim          | out dim
> ----------------|--------
> renamed_axes[i] | i
> ```

The alternative name for `dnnl_memory_desc_rename_axes` is
`dnnl_memory_desc_reinterpret_axes` with the same semantics.

#### 3.2.3. Behavior

- Supported memory formats in `in_memory_desc`: `blocked` and `any`.
  - An attempt to pass any other memory format leads to `unimplemented` status
    except for the memory format `undef`, in which case `invalid_arguments` is
    returned.

- Assuming the function name is `_permute_axes` the permutation happens in the
  following manner:
  ```
  for i in 0 .. ndims:
      out_memory_desc.dims[perm[i]] = in_memory_desc.dims[i]

  change blocking memory descriptor accordingly
  ```

- If the function name is `_rename_axes` or `_reinterpret_axes` the function
  performs:
  ```
  for i in 0 .. ndims:
      out_memory_desc.dims[i] = in_memory_desc.dims[renamed_axes[i]]

  change blocking memory descriptor accordingly
  ```

#### 3.2.3.1. Restrictions

- No restrictions -- the operation is always possible for memory descriptor
  with memory format `blocked` or `any`.


## 4. Memory Descriptor Operations that Won't Go to the Library

### 4.1. Flatten

Collapse 2 or more dimensions into one with size equal to the product of the
collapsed dimensions:
- `{a, b, c}` --> `{a, b * c}`

#### 4.1.1. Motivation

- Sometimes it might be required to collapse multiple dimensions into a single
  one. For instance, this might happen when CNN topology will migrate from 4D
  tensors (Batch, Spatial, and Channels) to a 2D to emit GEMM as a fully
  connected layer.

#### 4.1.2. Discussion

While the functionality seems to be useful, it has an implicit assumption on
the physical order of the dimensions which makes it quite dangerous. Consider
the following two similar cases:
1. `({a, b, c}, fmt=abc)` --> `({a, b * c}, fmt=ab)`
2. `({a, b, c}, fmt=acb)` --> `({a, b * c}, fmt=ab)`

It is obvious that the resulting memory objects are different, even though
there is no difference in their memory descriptions.

So whenever the flatten operation is safe, it is covered by
[*reshape* function](#31-reshape). For the cases when user might want to
force the dimensions to join, the [*extended reshape*](#43-extended-reshape)
is considered below.


### 4.2. Reorder According to the Axes Permutation with Memory Format Preservation

In contrast to other functions here, this is the only one that changes the
actual data, not only changes the memory descriptor. The idea is to permute the
axes, while preserving the original memory format:
- `({a, b, c}, fmt=aBc8b)` --> `({a, c, b}, fmt=aBc8b)`

#### 4.2.1. Motivation

- Implement similar framework operations, like
  [TF Permute](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Permute),
  when picking a particular DNNL memory format and/or axes order is impossible
  due to lack of information. Potential workflow:
  ``` python
  # TF code

  # Create a simple tensor. Since no extra information available at this moment
  # the DNNL integration code cannot make any assumptions on the _proper_ axes
  # order or the format, so the best possible option is just to keep the simple
  # plain format `abc...`.
  x = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
  # DNNL tensor: ({3, 2}, fmt=ab)

  y = tf.permute((2, 1), x)
  print(y.shape) # (2, 3)

  # There might be several options here:
  # 1. Change the format to `ba`, create a reorder, and recreate a memory with
  #    the pointer from the reordered memory, but with dims={2, 3} and format
  #    `ab`.
  # 2. Same as option 1., but directly use this new operation
  # 3. Create a new DNNL memory with dims={2, 3}, but format `ba`, and copied
  #    data from the original memory. Transposition is avoided, but in general
  #    this doesn't seem reasonable.
  ```

#### 4.2.2. Discussion

In the assumption DNNL will have `dnnl_memory_desc_permute_axes()` the required
functionality can be implemented as follows:

``` cpp
    permute_with_reorder(mem0, perm) --> mem1 {
        dnnl_memory_desc_t temp_mem1_md;
        dnnl_memory_desc_permute_axes(&temp_mem1_md, &mem0.md, perm);

        // Assumption: we know the wanted format tag.
        // Typically: a, ab, abc, abcd, ...
        memory_format_tag_t want_format_tag = ...;

        dnnl_memory_desc_init(&mem1.md, ndims, dims, want_format_tag);

        mem1 = memory(mem1.md, ALLOCATE_PTR);

        temp_mem1 = memory(temp_mem1_md, mem0.ptr);
        reorder(temp_mem1, mem1);

        return mem1;
    }
```

Here we assume a user knows the desired format which is (according to the
definition) must be the same as in `mem0`. For plain formats in framework
integration code this should be the case: the format will always be simple
`abc..`.

More advanced users might want to preserve the blocked format, but this will
require them to know that the subsequent operation supports this blocked
format, which in turn means that these more advanced users probably know the
corresponding format tag as well. If that is true -- they could use the same
implementation above with the proper `want_format_tag`.


### 4.3. Extended Reshape

[3.1. Reshape](#31-reshape) and [4.1. Flatten](#41-flatten) could be
generalized to an *extended reshape* function, where the logical shape can be
changed in any way, as long as the data volume stays the same, and user
understands what they do.
Examples:
- `{a, b, c}` --> `{a, b * c}`
- `{a, b * c}` --> `{a, b, c}`
- `{a, b, c * d}` --> `{a * b, c, d}`
- `{a, b, c}` --> `{a / 2, 2 * b * 2, c / 2}`
- `({a, b, c, d}, fmt=aBcd8b)` --> `({a, b * c * d}, ab)`

#### 4.3.1. Motivation

- The functionality can be useful for the internal development to reduce the
  number of dimensions when they don't matter much.
  - Inner product takes 4D inputs, while the GEMM itself works with 2D tensors
    only. By framework definition, inner product first flattens 2nd to the last
    dimensions and performs GEMM. This could be done in the implementation too:
    ``` cpp
    inner_product(src, weights, dst) {
        // assuming compatibility of src.md and weights.md
        src_2d = memory(
                reshape(src.md,
                    {src.md.dims[0], product(src.md.dims[1..])}),
                src.ptr);
        weights_2d = memory(
                reshape(weights.md,
                    {weights.md.dims[0], product(weights.md.dims[1..])}),
                weights.ptr);
        dst = GEMM(src_2d, weights_2d);
    }
    ```
    Note, that regular *reshape* might fail here for some valid combinations.
    Example:
    ```
    src.md     = ({n, i, h, w}, fmt=aBcd16b)
    weights.md = ({o, i, h, w}, fmt=aBcd16b)
    // Reshape from 3.1. will fail to collapse {i, h, w} as `i`-dim is blocked.
    ```

#### 4.3.2. API

``` cpp
typedef enum {
    extended_reshape_flags_none = 0u,
    extended_reshape_allow_logical_physical_order_mismatch = 1u,
    extended_reshape_allow_dims_padded_dims_mismatch = 2u, // maybe at some point
} dnnl_extended_reshape_flags_t;

dnnl_status_t
dnnl_memory_desc_extended_reshape(
        dnnl_memory_desc_t *out_memory_desc,        // output memory desc, write only. in-place supported
        const dnnl_memory_desc_t *in_memory_desc,   // input memory desc, read only
        int ndims, const dims_t dims,               // new dims
        unsigned flags                              // ORed dnnl_extended_reshape_flags_t
        );
```

#### 4.3.3. Behavior

- Supported memory formats in `in_memory_desc`: `blocked` and `any`.
  - An attempt to pass any other memory format leads to `unimplemented` status
    except for the memory format `undef`, in which case `invalid_arguments` is
    returned.

- If `flags` are `extended_reshape_flags_none` the behavior must match the
  *reshape* function from the [section 3.1](#31-reshape).

- The collapsed axes must always form a contiguous chunk of memory. No matter
  what the flags are the follow reshape is not permitted:
  - `({a, b, c}, fmt=acb)` --> `({a * b, c}, fmt=...)`.

- If `flags` have `extended_reshape_allow_logical_physical_order_mismatch`, do
  allow joining the axes with physical order different from the logical one.
  This might be useful if user controls the correspondence between the tensors
  or generally understands what they are doing.
  - Example with inner product:
    ``` cpp
    // The src and weights tensors are compatible with each other, but the orders
    // of logical axes do not much the physical ones
    src_4d     = ({N, I, H, W}, fmt=nhwc)
    weights_4d = ({O, I, H, W}, fmt=hwio)

    // The `extended_reshape` could be safely applied.
    src_2d     = reshape(src_4d,     {N, I * H * W}) = ({N, I * H * W}, fmt=nc)
    weights_2d = reshape(weights_4d, {O, I * H * W}) = ({O, I * H * W}, fmt=io)
    ```

  - Example with framework integration code:
    ``` cpp
    buf = (dims={N, C, H, W}, fmt=nhwc) // the buffer from TF in format NHWC

    // oK, even though the logical order is C->H->W while the physical one is
    // H->W->C, the following flatten is intentional:
    buf_flatten = (dims={N, C * H * W}, fmt=nc)
    ```

#### 4.3.3.1. Restrictions

- Memory descriptor format must be `blocked` or `any`.

- Unless `extended_reshape_allow_dims_padded_dims_mismatch` flag is set, the
  changed dimension must have equal dimensions and their padded counterparts.

- Unless `extended_reshape_allow_logical_physical_order_mismatch` flag is set,
  the collapsed dimensions must have the same physical (contiguous) order as
  their logical counterparts.
  - In particular, this means that collapsing one or more dimensions that are
    blocked is impossible unless the other collapsed dimensions are of size
    `1`.

#### 4.3.4. Discussion

The extended reshape requires very careful use. To use it in frameworks, one
should always have DNNL-to-framework permutation in mind and confirm that the
memory descriptor after extended reshape has logical meaning. Alternatively, if
it is used for plain format `abcd...` (i.e. where the physical format
corresponds to the logical one) the usefulness of the function is questionable,
as it does the same as just creating a new memory descriptor with the adjusted
dimensions.

To prevent the function from misuse the suggestion is to not implement it at
all. If at some point there would be a use-case for which this function would
be really helpful, we consider adding it (most likely as an internal function
that advanced users may use by including the corresponding header file).


## 5. Other Changes

1. Memory descriptor comparison function should be fixed. Currently it compares
   the dimensions and strides arrays element by element. For dimensions of size
   `1` which padded dimension is also `1` the stride is irrelevant and should
   be ignored.
   Examples:

   | #   | dims (common) | padded dims (common) | md1 strides | md2 strides | `memory_desc_equal()` returns now | Should return |
   | --- | ------------- | -------------------- | ----------- | ----------- | --------------------------------- | ------------- |
   | 1   | (1, 2)        | ( 1, 2)              | (2, 1)      | (1, 1)      | false                             | true          |
   | 2   | (1, 2)        | (16, 2)              | (2, 1)      | (1, 16)     | false                             | false         |


## 6. Afterword

### 6.1. Doomed Memory Bandwidth Bound Primitives (Almost)

DNNL programming model assumes that primitives like pooling and batch
normalization should use whatever memory format comes from the previous
operation. With the `permute_axes()` function the format might be quite
questionable, which will lead to executing slow reference code. So there is a
chance the programming model should be changed to something along the lines:
- Create a primitive desc, query chosen algorithm, and if it contains `ref`
  reorder the input memory descriptors to the plain format.

An example. Assume TensorFlow user wants to execute pooling on weights (even
though the operation will be very weird):
``` python
    buf = ({H, W, I, O})
    buf2 = tf.pool(buf) # the pooling happens along W and I dimensions
```

In this case the integration code will permute the axes using
`dnnl_memory_desc_perute_axes()`, and original format like `OIhw16i16o` may end
up being `IWoh16w16i = CWnh16w16c`.

Fortunately, no one we know of is doing something like that. Yet.


---

EOD
