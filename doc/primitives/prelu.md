PReLU {#dev_guide_prelu}
============================

>
> [API Reference](@ref dnnl_api_prelu)
>

## General

The PReLU primitive (Leaky ReLU with trainable alpha parameter) performs
forward or backward operation on data tensor. Weights (alpha) tensor supports
broadcast-semantics. Broadcast configuration is assumed based on src and
weights dimensions.

Example broadcasts:

| broadcast type | src dimensions       | weights dimensions   |
|----------------|----------------------|----------------------|
| Channel-shared | \f$\{n, c, h ,w\}\f$ | \f$\{1, 1, 1 ,1\}\f$ |
| Channel-wise   | \f$\{n, c, h ,w\}\f$ | \f$\{1, c, 1 ,1\}\f$ |
| Whole-tensor   | \f$\{n, c, h ,w\}\f$ | \f$\{n, c, h ,w\}\f$ |
| Shared-axes    | \f$\{n, c, h ,w\}\f$ | \f$\{n, 1, h ,1\}\f$ |

@note
   Shared-axes indicates broadcast with any combination of shared
dimensions.

### Forward

The PReLU operation is defined by the following formulas.
We show formulas only for 2D spatial data which are straightforward to
generalize to cases of higher and lower dimensions. Variable names follow the
standard @ref dev_guide_conventions.
For no broadcast case, results are calculated using formula:

\f[
    \dst(n, c, h, w) =
        \begin{cases}
        \src(n, c, h, w)  & \mbox{if } \src(n, c, h, w) > 0 \\
        \src(n, c, h, w) \cdot \weights(n, c, h, w) & \mbox{if }
        \src(n, c, h, w) \leq 0
        \end{cases}
\f]

Depending on broadcast configuration, result is calculated taking into account
shared dimensions of weights tensor.

#### Difference Between Forward Training and Forward Inference

There is no difference between the #dnnl_forward_training
and #dnnl_forward_inference propagation kinds.

### Backward

The backward propagation computes \f$\diffsrc\f$ and \f$\diffweights\f$.
For no broadcast case, results are calculated using formula:

\f[
    \begin{align}
    \mbox{diff_src}(n, c, h, w) &=
        \begin{cases}
        \mbox{diff_dst}(n, c, h, w)  & \mbox{if } \src(n, c, h, w) > 0 \\
        \mbox{diff_dst}(n, c, h, w) \cdot \weights(n, c, h, w) &
        \mbox{if } \src(n, c, h, w) \leq 0
        \end{cases}\\\\
    \mbox{diff_weights}(n, c, h, w) &=
        \min(\src(n, c, h, w), 0) \cdot \mbox{diff_dst}(n, c, h, w)
    \end{align}
\f]

Similar to forward propagation, result is calculated taking into
account shared dimensions of weights tensor.
\f$\diffweights\f$ results are accumulated according to weights tensor shared
dimensions, since \f$\diffweights\f$ tensor must match \f$\weights\f$ tensor.


## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output | Execution argument index |
|------------------------|--------------------------|
| \f$\src\f$             | DNNL_ARG_SRC             |
| \f$\dst\f$             | DNNL_ARG_DST             |
| \f$\weights\f$         | DNNL_ARG_WEIGHTS         |
| \f$\diffsrc\f$         | DNNL_ARG_DIFF_SRC        |
| \f$\diffdst\f$         | DNNL_ARG_DIFF_DST        |
| \f$\diffweights\f$     | DNNL_ARG_DIFF_WEIGHTS    |


## Implementation Details

### General Notes

 * Prelu primitive requires all input/output tensors to have the
   same number of dimensions. Dimension sizes can differ however.

 * \weights tensor dimensions sizes must follow broadcast semantics.
   Each dimension can either equal corresponding data dimension or
   equal 1 - to indicate that dimension is shared.

 * Prelu primitive requires that \diffweights tensor has exact same dimensions
   sizes as \weights tensor, \diffsrc as src and \diffdst as dst.

 * \weights tensor can be initialized with format_tag::any
   primitive will match it to data tensor format.

### Data Type Support

The PReLU primitive supports the following combinations of data types:

| Propagation        | Source / Destination        |
|:-------------------|:----------------------------|
| forward / backward | f32, s32, bf16, f16, s8, u8 |

### Data Representation

The PReLU primitive works with arbitrary data tensors. There is no special
meaning associated with any logical dimensions.

## Implementation Limitations

Current implementation supports all tensors up to 3D spatial (n, c, d, h, w).

## Performance Tips

Its recommended to allow PReLU primitive to choose the appropriate weights
memory format by passing weights_md with format_tag::any.
For best performance, the weights memory format should match
data memory format.

## Example

[PReLU Primitive Example](@ref prelu_example_cpp)

@copydetails prelu_example_cpp_short
