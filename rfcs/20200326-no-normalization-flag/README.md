# Introducing no normalization flag

## Introduction

In its current implementation, the operation descriptor API for
`batch_normalization` and `layer_normalization` requires the user to provide a
normalization flag, and the library currently supports four options:

- `use_global_stats` to use mean and variance as input,
- `use_scale_shift` to use scale and shift as input,
- `fuse_norm_relu` compute the primitive with fused ReLU, or
- any bitwise OR combination between the above flags.

In terms of primitive functionality, by providing any of the above flags the
user is also required to provide additional input for the primitive execution,
in particular allocated memory regions for, e.g., mean and variance (c.f. [Batch
Normalization](https://intel.github.io/mkl-dnn/dev_guide_batch_normalization.html),
[Layer
Normalization](https://intel.github.io/mkl-dnn/dev_guide_layer_normalization.html)
documentation).

The library does not provide any flag for the situation when the user wishes to
execute the batch/layer normalization primitive without any of the options
listed above. In this case, the user goes through a counter-intuitive process by
constructing the `batch_normalization::desc` or `layer_normalization::desc`
with, e.g., `(normalization_flag)0u`, `(normalization_flag)0x0`, or any other
user's preference. Here, the user also relies on the assumption that the library
expects the zero as a *no normalization flag*. While this assumption holds true
in the current version of the library, this behavior may likely change in the
future, making user's code incompatible with the library.



## Proposal

Below, a proposal to introduce a new flag that denotes no normalization options
is presented. This is expected to make the usability of the API discussed above
friendlier to the user and more future-proof.

### Introduce none flag

A new `dnnl_normalization_flags_none` flag will be introduced, that supports
user's intent not wishing to provide any extra options to the
`batch_normalization` or `layer_normalization` operation descriptor. This flag
will be introduced to C API as

~~~cpp
typedef enum {
    /// Use no normalization flags
    ///
    /// If specified
    ///  - on forward training propagation mean and variance are computed and
    ///    stored as output
    ///  - on backward propagation compute full derivative wrt data
    ///  - on backward propagation prop_kind == #dnnl_backward_data has the same
    ///    behavior as prop_kind == #dnnl_backward
    ///
    dnnl_normalization_flags_none = 0x0U,

    ...
} dnnl_normalization_flags_t;
~~~

as well as C++ API as
~~~cpp
enum class normalization_flags : unsigned {
    /// Use no normalization flags. If specified, the library computes mean and
    /// variance on forward propagation for training and inference, outputs them
    /// on forward propagation for training, and computes the respective
    /// derivatives on backward propagation.
    none = dnnl_normalization_flags_none,

    ...
}
~~~
