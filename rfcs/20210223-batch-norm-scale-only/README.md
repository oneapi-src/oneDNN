# RFC: Support only scale parameter in normalization primitives.

## Motivation

There is a request from OpenVINO team to support scaleshift in a different way
than it is supported now. The main pain point is having just scales (no shifts)
and using just these scales with current API results is inconvenient due to
requirement of doubling channels in a memory descriptor for scaleshift argument,
doubling the memory with a separate memory object, copying the data from
original scales - this all leads to a suboptimal performance and excessive code.

## Proposal

### Option 1 (recommended) - new flags and memory argument constants.

By historical reasons oneDNN API required a user to pass shift and scale as a
single memory object which internally would be split into two. This option
suggests separating entities into two independent memories, thus, making it
easier to pass scales and shifts directly into primitive arguments if they were
used prior a batch normalization call.

~~~c
/* dnnl_types.h */

/// Flags for normalization primitives.
typedef enum {
    ...
    /// Use scale parameter only
    ///
    /// If specified:
    ///  - on forward propagation use scale for the batch normalization results
    ///  - on backward propagation (for prop_kind == #dnnl_backward) compute
    ///    diff wrt scale (hence one extra output used)
    ///
    /// If no specified:
    ///  - on backward propagation prop_kind == #dnnl_backward_data has the
    ///    same behavior as prop_kind == #dnnl_backward
    dnnl_use_scale = 0x8U,

    /// Use shift parameter only
    ///
    /// If specified:
    ///  - on forward propagation use shift (aka bias) for the batch
    ///    normalization results
    ///  - on backward propagation (for prop_kind == #dnnl_backward) compute
    ///    diff wrt shift (hence one extra output used)
    ///
    /// If no specified:
    ///  - on backward propagation prop_kind == #dnnl_backward_data has the
    ///    same behavior as prop_kind == #dnnl_backward
    dnnl_use_shift = 0x16U,
} dnnl_normalization_flags_t;

/// A special mnemonic for scale argument of normalization primitives.
#define DNNL_ARG_SCALE 51 // or alternatively DNNL_ARG_WEIGHTS_1 (34)
/// A special mnemonic for shift argument of normalization primitives.
#define DNNL_ARG_SHIFT 52 // or alternatively DNNL_ARG_WEIGHTS_2 (35)
~~~

For compatibility reasons, oneDNN would still support both versions of flags
but internally would rely only on new ones. It means that using
`dnnl_use_scaleshift` is same as using `dnnl_use_scale | dnnl_use_shift`.

As for memory arguments, pointers will be taken from argument map based on flag
value specified.

### Option 2 - add binary post-op support.

Since scale and shift are linear operations of multiplication and addition,
binary post-op could work as a nice substitution for shiftscale memory. The
implementation is straightforward. The only drawback is that this solution works
for forward propagation and doesn't help for backward propagation.

In some sense, this is a bit different kind of support but due to the same
mathematical semantics works fine.

### Option 3 - implement both options 1 and 2.

This is a combination of both options to implement giving user a choice between
using binary post-op or native API in case of forward.

EOD.
