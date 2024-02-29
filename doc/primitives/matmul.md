Matrix Multiplication {#dev_guide_matmul}
=========================================

>
> [API Reference](@ref dnnl_api_matmul)
>

## General

The matrix multiplication (MatMul) primitive computes the product of two 2D
tensors with optional bias addition (the variable names follow the standard @ref
dev_guide_conventions):

\f[
    \dst(m, n) =
        \sum_{k=0}^{K - 1} \left(
            \src(m, k) \cdot \weights(k, n)
        \right) +
        \bias(m, n)
\f]

The MatMul primitive also supports batching multiple independent matrix
multiplication operations, in which case the tensors can be up to 12D:

\f[
    \dst(bs_0, bs_1, \ldots, m, n) =
        \sum_{k=0}^{K - 1} \left(
            \src(bs_0, bs_1, \ldots, m, k) \cdot
            \weights(bs_0, bs_1, \ldots, k, n) \right) + \bias(bs_0, bs_1, \ldots, m, n)
\f]

MatMul also supports implicit broadcast semantics i.e., \src can be broadcasted
into \weights if the corresponding dimension in \src is 1 (and vice versa).
However, all tensors (including \bias, if it exists) must have the same number
of dimensions.

The shape of \dst only depends on \src and \weights tensors. The \bias cannot
change the dimensions of \dst by broadcasting. In other words, for every
dimension, the following constraint must hold true:
`dimension(bias) == dimension(dst) || dimension(bias) == 1`.

## Execution Arguments

When executed, the inputs and outputs should be mapped to an execution
argument index as specified by the following table.

| Primitive input/output      | Execution argument index                                                   |
|-----------------------------|----------------------------------------------------------------------------|
| \src                        | DNNL_ARG_SRC                                                               |
| \weights                    | DNNL_ARG_WEIGHTS                                                           |
| \bias                       | DNNL_ARG_BIAS                                                              |
| \dst                        | DNNL_ARG_DST                                                               |
| \f$\text{binary post-op}\f$ | DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) \| DNNL_ARG_SRC_1  |
| \f$\text{prelu post-op}\f$  | DNNL_ARG_ATTR_MULTIPLE_POST_OP(prelu_post_op_position) \| DNNL_ARG_WEIGHTS |

## Implementation Details

### General Notes

1. The MatMul primitive supports input and output tensors with run-time
   specified shapes and memory formats. The run-time specified dimensions or
   strides are specified using the #DNNL_RUNTIME_DIM_VAL wildcard value during
   the primitive initialization and creation stage. At the execution stage, the
   user must pass fully specified memory objects so that the primitive is able
   to perform the computations. Note that the less information about shapes
   or format is available at the creation stage, the less performant execution
   will be.  In particular, if the shape is not known at creation stage, one
   cannot use the special format tag #dnnl::memory::format_tag::any to enable an
   implementation to choose the most appropriate memory format for the
   corresponding input or output shapes. On the other hand, run-time specified
   shapes enable users to create a primitive once and use it in different
   situations.

2. Inconsistency with dimensions being "primitive-creation-time-defined" vs
   "runtime-defined" is invalid. For example, \src and \weights with dimensions
   set to `{3, 4, 4}` and `{DNNL_RUNTIME_DIM_VAL, 4, 4}` respectively is
   invalid.

3. The broadcasting shape consistency check is not done for the dimensions with
   #DNNL_RUNTIME_DIM_VAL. It is user responsibility to make sure the dimensions
   for the tensors are valid.

4. Multiple batch dimensions and broadcasting of batch dimensions of `src` and
   `weights` are supported for both CPU and GPU engines.

   Please check tutorials below to see #DNNL_RUNTIME_DIM_VAL support in use.

### Data Types

The MatMul primitive supports the following combinations of data
types for source, destination, weights, and bias tensors:


| Source         | Weights   | Destination                 | Bias                        |
|:---------------|:----------|:----------------------------|:----------------------------|
| f32            | f32       | f32                         | f32                         |
| f16            | f16       | f16, u8, s8                 | f16, f32                    |
| bf16           | bf16      | f32, bf16                   | bf16, f32                   |
| f32, bf16, f16 | u8, s8    | f32, bf16, f16              | f32, bf16, f16              |
| u8, s8         | s8        | u8, s8, s32, f32, f16, bf16 | u8, s8, s32, f32, f16, bf16 |
| f8_e5m2        | f8_e5m2   | f32, f16, bf16, f8_e5m2     | f32, bf16, f16              |


### Data Representation

The MatMul primitive expects the following tensors:

| Dims | Source                          | Weights                         | Destination                     | Bias                                                      |
|:-----|:--------------------------------|:--------------------------------|:--------------------------------|:----------------------------------------------------------|
| 2D   | M \f$\times\f$ K                | K \f$\times\f$ N                | M \f$\times\f$ N                | None or \f$(M \text{ or } 1) \times (N  \text{ or } 1)\f$ |
| ND   | S \f$\times\f$ M \f$\times\f$ K | W \f$\times\f$ K \f$\times\f$ N | D \f$\times\f$ M \f$\times\f$ N | None or B                                                 |

where for the sake of notational convenience, we have

\f[
S = \prod_{i = 0}^{ND - 3} \mathrm{src\_dims}[i], \; W = \prod_{i = 0}^{ND - 3} \mathrm{weights\_dims}[i] \\
D = \prod_{i = 0}^{ND - 3} \mathrm{\dst\_dims}[i], \; B = \prod_{i = 0}^{ND - 1} \left( \mathrm{\dst\_dims}[i] \mbox{ or } 1 \right)
\f]

The MatMul primitive is generally optimized for the case in which memory objects
use plain memory formats. Additionally, the \src and \weights must have at least
one of the axes `m` or `k` and `n` or `k` contiguous (i.e., stride=1)
respectively. However, it is recommended to use the placeholder memory format
#dnnl::memory::format_tag::any if an input tensor is reused across multiple
executions. In this case, the primitive will set the most appropriate memory
format for the corresponding input tensor.

The memory format of the destination tensor should always be plain with `n` axis
contiguous. For example, #dnnl::memory::format_tag::ab for the 2D case and
#dnnl::memory::format_tag::abc or #dnnl::memory::format_tag::bac for the 3D one.

### Attributes and Post-ops

Attributes and post-ops enable modifying the behavior of the MatMul primitive.
The following attributes and post-ops are supported:

| Type      | Operation                                                      | Description                                                                   | Restrictions                        |
|:----------|:---------------------------------------------------------------|:------------------------------------------------------------------------------|:------------------------------------|
| Attribute | [Scales](@ref dnnl::primitive_attr::set_scales_mask)           | Scales the result by given scale factor(s)                                    |                                     |
| Attribute | [Zero-points](@ref dnnl::primitive_attr::set_zero_points_mask) | Sets zero point(s) for the corresponding tensors                              | Int8 computations only              |
| Post-op   | [Eltwise](@ref dnnl::post_ops::append_eltwise)                 | Applies an @ref dnnl_api_eltwise operation to the result                      |                                     |
| Post-op   | [Sum](@ref dnnl::post_ops::append_sum)                         | Adds the operation result to the destination tensor instead of overwriting it |                                     |
| Post-op   | [Binary](@ref dnnl::post_ops::append_binary)                   | Applies a @ref dnnl_api_binary operation to the result                        | General binary post-op restrictions |
| Post-op   | [Prelu](@ref dnnl::post_ops::append_prelu)                     | Applies an @ref dnnl_api_prelu operation to the result                        |                                     |

The following masks are supported by the primitive:
- 0, which applies one scale / zero point value to an entire tensor, and
- 2, which applies a scale value per column along the
  `n`dimension for `DNNL_ARG_WEIGHTS`.

When scales and/or zero-points masks are specified, the user must
provide the corresponding scales and/or zero-points as additional
input memory objects with argument `DNNL_ARG_ATTR_SCALES |
DNNL_ARG_${MEMORY_INDEX}` or `DNNL_ARG_ATTR_ZERO_POINTS |
DNNL_ARG_${MEMORY_INDEX}` during the execution stage. For instance, a
source tensor zero points memory argument would be passed with index
(`DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC`).

@note Please check tutorials below to see run-time attributes in use.

## Implementation Limitations

1. Check @ref dev_guide_data_types.

2. **GPU**
   - Supports up to 6 dimensions.
   - Source zero point mask of `0` is only supported.
   - Sum post-op doesn't support data type other than destination data type.
   - Bias of bf16 data type is supported for configuration with bf16 source data
     type and weights bf16 data type, and up to three dimensional matrices.
   - Only reference support is available for f8_e4m3. Optimized implementation
     for f8_e5m2 is available only on Intel(R) Data Center GPU Max Series.
   - Configuration with int8 source data type, s8 weight data type and bf16
     destination data type don't support:
     * Destination zero point.
     * Runtime dimensions.
     * Three and higher dimensional matrices.

3. **CPU**
   - Configuration with int8 source data type, s8 weight data type and f16
     destination data type isn't supported.
   - Configuration with floating point source data type, integer weights data
     type and floating point destination data type is not optimized.
   - Only reference support for fp8 data types (f8_e5m2, f8_e4m3) is
     is available on CPU.
 
## Performance Tips

- Use #dnnl::memory::format_tag::any for either of the input tensors if and
  only if the shape of the corresponding tensor is fully known at creation
  time and it is possible to cache reordered tensors across multiple primitive
  executions. For instance, a good candidate for reuse are the weights tensors
  during inference: their shapes and data types are known in advance; thus
  they can be reordered during the first inference pass and can be reused
  during the subsequent passes. However, if any of the input tensors cannot be
  reused, it is best to force the primitive to use the same format as that used
  by the tensors.

## Examples

The following examples are available:

### Matrix Multiplication Primitive Examples

[MatMul Primitive Example](@ref matmul_example_cpp)

@copydetails matmul_example_cpp_short


[MatMul Tutorial: Comparison with SGEMM](@ref cpu_sgemm_and_matmul_cpp) (CPU only)

@copydetails cpu_sgemm_and_matmul_cpp_short


[MatMul Tutorial: INT8 Inference](@ref inference_int8_matmul_cpp)

@copydetails inference_int8_matmul_cpp_short


[MatMul Tutorial: Quantization](@ref cpu_matmul_quantization_cpp) (CPU only)

@copydetails cpu_matmul_quantization_cpp_short

[MatMul Tutorial: Weights decompression](@ref weights_decompression_matmul_cpp) (CPU only)

@copydetails weights_decompression_matmul_cpp_short
