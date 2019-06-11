Convolution {#dev_guide_convolution}
=====================================

>
> API reference: [C](@ref c_api_convolution), [C++](@ref cpp_api_convolution)
>

The convolution primitive computes forward, backward, or weight update for a
batched convolution operation on 1D, 2D, or 3D spatial data with bias.

The convolution operation is defined by the following formulas. We show formulas
only for 2D spatial data which are straightforward to generalize to cases of
higher and lower dimensions. Variable names follow the standard
@ref dev_guide_conventions.

Let \f$src\f$, \f$weights\f$ and \f$dst\f$ be \f$N \times IC \times IH \times
IW\f$, \f$OC \times IC \times KH \times KW\f$, and \f$N \times OC \times OH
\times OW\f$ tensors respectively. Let \f$bias\f$ be a 1D tensor with \f$OC\f$
elements.

The following formulas show how Intel MKL-DNN computes convolutions. They are
broken down into several types to simplify the exposition, but in reality the
convolution types can be combined.

To further simplify the formulas, we assume that \f$src(n, ic, ih, iw) = 0\f$
if \f$ih < 0\f$, or \f$ih \geq IH\f$, or \f$iw < 0\f$, or \f$iw \geq IW\f$.

### Forward

#### Regular Convolution

\f[dst(n, oc, oh, ow) =  bias(oc) + \\
    + \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
        src(n, ic, oh \cdot SH + kh - ph_0, ow \cdot SW + kw - pw_0)
        \cdot
        weights(oc, ic, kh, kw).\f]

Here:

- \f$OH = \left\lfloor{\frac{IH \cdot SH - KH + ph_0 + ph_1}{sh}}
        \right\rfloor + 1,\f$

- \f$OW = \left\lfloor{\frac{IW \cdot SW - KW + pw_0 + pw_1}{sw}}
        \right\rfloor + 1.\f$

#### Convolution with Groups

In the API, Intel MKL-DNN adds a separate groups dimension to memory objects
representing weights tensors and represents weights as \f$G \times OC_G \times
IC_G \times KH \times KW \f$ 5D tensors for 2D convolutions with groups.

\f[
    dst(n, g \cdot OC_G + oc_g, oh, ow) =
        bias(g \cdot OC_G + oc_g) + \\
        +
        \sum_{ic_g=0}^{IC_G-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
            src(n, g \cdot IC_G + ic_g, oh + kh - ph_0, ow + kw - pw_0)
        \cdot
        weights(g, oc_g, ic_g, kh, kw),
\f]

where
- \f$IC_G = \frac{IC}{G}\f$,
- \f$OC_G = \frac{OC}{G}\f$,
- \f$ic_g \in [0, IC_G)\f$ and
- \f$oc_g \in [0, OC_G).\f$

The case when \f$OC_G = IC_G = 1\f$ is also known as *a depthwise convolution*.

#### Convolution with Dilation

\f[
    dst(n, oc, oh, ow) =
        bias(oc) + \\
        +
        \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1}
            src(n, ic, oh + kh \cdot dh - ph_0, ow + kw \cdot dw - pw_0)
            \cdot
            weights(oc, ic, kh, kw).
\f]

Here:

- \f$OH = \left\lfloor{\frac{IH - DKH + ph_0 + ph_1}{sh}}
        \right\rfloor + 1,\f$ where \f$DKH = 1 + (KH - 1) \cdot (DH + 1)\f$, and

- \f$OW = \left\lfloor{\frac{IW - DKW + pw_0 + pw_1}{sw}}
        \right\rfloor + 1,\f$ where \f$DKW = 1 + (KW - 1) \cdot (DW + 1)\f$.

#### Deconvolution (Transposed Convolution)

Deconvolutions (also called fractionally strided convolutions or transposed
convolutions) work by swapping the forward and backward passes of a
convolution. One way to put it is to note that the weights define a
convolution, but whether it is a direct convolution or a transposed
convolution is determined by how the forward and backward passes are computed.

#### Difference Between [Forward Training](#mkldnn_forward_training) and [Forward Inference](#mkldnn_forward_inference)

There is no difference between the #mkldnn_forward_training
and #mkldnn_forward_inference propagation kinds.

### Backward

The backward propagation computes \f$diff\_src\f$
based on \f$diff\_dst\f$ and \f$weights\f$.

The weights update computes \f$diff\_weights\f$ and \f$diff\_bias\f$
based on \f$diff\_dst\f$ and \f$src\f$.

@note The *optimized* memory formats \f$src\f$ and \f$weights\f$ might be
different on forward propagation, backward propagation, and weights update.

## Implementation Details

### General Notes

N/A.

### Data Types

Convolution primitive supports the following combination of data types for
source, destination, and weights memory objects:

| Propagation        | Source   | Weights   | Destination       | Bias
| :--                | :--      | :--       | :--               | :--
| forward / backward | f32      | f32       | f32               | f32
| forward            | f16      | f16       | f16               | f16
| forward            | u8, s8   | s8        | u8, s8, s32, f32  | u8, s8, s32, f32
| forward            | bf16     | bf16      | f32, bf16         | f32, bf16
| backward           | f32, bf16| f32, bf16 | bf16              | f32, bf16

@warning
    There might be hardware and/or implementation specific restrictions.
    Check [Implementation Limitations](@ref dg_conv_impl_limits) section below.

### Data Representation

Like other CNN primitives, the convolution primitive expects the following
tensors:

| Spatial | Source / Destination                        | Weights
| :--     | :--                                         | :--
| 1D      | \f$N \times C \times W\f$                   | \f$[G \times ] OC \times IC \times KW\f$
| 2D      | \f$N \times C \times H \times W\f$          | \f$[G \times ] OC \times IC \times KH \times KW\f$
| 3D      | \f$N \times C \times D \times H \times W\f$ | \f$[G \times ] OC \times IC \times KD \times KH \times KW\f$

Physical format of data and weights memory objects is critical for convolution
primitive performance. In the Intel MKL-DNN programming model, convolution is
one of the few primitives that support the placeholder memory format tag
 #mkldnn::memory::format_tag::any (shortened to `any` from now on) and can
define data and weight memory objects format based on the primitive parameters.
When using `any` it is necessary to first create a convolution primitive
descriptor and then query it for the actual data and weight memory objects
formats.

While convolution primitives can be created with memory formats specified
explicitly, the performance is likely to be suboptimal.

The table below shows the combinations for which **plain** memory formats
the convolution primitive is optimized for.

| Spatial    | Convolution Type | Data / Weights logical tensor | Implementation optimized for memory formats
| :--        | :--              | :--                           | :--
| 1D, 2D, 3D |                  | `any`                         | *optimized*
| 1D         | f32, bf16        | NCW / OIW, GOIW               | #mkldnn_ncw (#mkldnn_abc) / #mkldnn_oiw (#mkldnn_abc), #mkldnn_goiw (#mkldnn_abcd)
| 1D         | int8             | NCW / OIW                     | #mkldnn_nwc (#mkldnn_acb) / #mkldnn_wio (#mkldnn_cba)
| 2D         | f32, bf16        | NCHW / OIHW, GOIHW            | #mkldnn_nchw (#mkldnn_abcd) / #mkldnn_oihw (#mkldnn_abcd), #mkldnn_goihw (#mkldnn_abcde)
| 2D         | int8             | NCHW / OIHW, GOIHW            | #mkldnn_nhwc (#mkldnn_acdb) / #mkldnn_hwio (#mkldnn_cdba), #mkldnn_hwigo (#mkldnn_decab)
| 3D         | f32, bf16        | NCDHW / OIDHW, GOIDHW         | #mkldnn_ncdhw (#mkldnn_abcde) / #mkldnn_oidhw (#mkldnn_abcde), #mkldnn_goidhw (#mkldnn_abcdef)
| 3D         | int8             | NCDHW / OIDHW                 | #mkldnn_ndhwc (#mkldnn_acdeb) / #mkldnn_dhwio (#mkldnn_cdeba)

### Post-ops and Attributes

Post-ops and attributes enable you to modify the behavior of the convolution
primitive by applying the output scale to the result of the primitive and by
chaining certain operations after the primitive. The following attributes and
post-ops are supported:

| Propagation | Type      | Operation                                                      | Restrictions           | Description
| :--         | :--       | :--                                                            | :--                    | :--
| forward     | attribute | [Output scale](@ref mkldnn::primitive_attr::set_output_scales) | int8 convolutions only | Scales the result of convolution by given scale factor(s)
| forward     | post-op   | [eltwise](@ref mkldnn::post_ops::append_eltwise)               |                        | Applies an @ref c_api_eltwise operation to the result (currently only #mkldnn_eltwise_relu algorithm is supported)
| forward     | post-op   | [sum](@ref mkldnn::post_ops::append_sum)                       |                        | Adds the operation result to the destination tensor instead of overwriting it

@note The library doesn't prevent using post-ops in training, but note that
not all post-ops are feasible for training usage. For instance, using ReLU
with non-zero negative slope parameter as a post-op would not produce an
additional output `workspace` that is required to compute backward propagation
correctly. Hence, in this particular case one should use separate convolution
and eltwise primitives for training.

The following post-ops chaining is supported by the library:

| Type of convolutions      | Post-ops sequence supported
| :--                       | :--
| f32 and bf16 convolution  | eltwise, sum, sum -> eltwise
| int8 convolution          | eltwise, sum, sum -> eltwise, eltwise -> sum

The attributes and post-ops take effect in the following sequence:
- Output scale attribute,
- Post-ops, in order they were attached.

The operations during attributes and post-ops applying are done in single
precision floating point data type. The conversion to the actual destination
data type happens just before the actual storing.

#### Example 1

Consider the following pseudo code:

~~~
    attribute attr;
    attr.set_output_scale(alpha);
    attr.set_post_ops({
            { sum={scale=beta} },
            { eltwise={scale=gamma, type=tanh, alpha=ignore, beta=ignored }
        });

    convolution_forward(src, weights, dst, attr)
~~~

The would lead to the following:

\f[
    dst(\overline{x}) =
        \gamma \cdot \tanh \left(
            \alpha \cdot conv(src, weights) +
            \beta  \cdot dst(\overline{x})
        \right)
\f]

#### Example 2

The following pseudo code:

~~~
    attribute attr;
    attr.set_output_scale(alpha);
    attr.set_post_ops({
            { eltwise={scale=gamma, type=relu, alpha=eta, beta=ignored }
            { sum={scale=beta} },
        });

    convolution_forward(src, weights, dst, attr)
~~~

That would lead to the following:

\f[
    dst(\overline{x}) =
        \beta \cdot dst(\overline{x}) +
        \gamma \cdot ReLU \left(
            \alpha \cdot conv(src, weights),
            \eta
        \right)
\f]

## Algorithms

Intel MKL-DNN implements convolution primitives using several different
algorithms:

- _Direct_. The convolution operation is computed directly using SIMD
  instructions. This is the algorithm used for the most shapes and supports
  int8, f32 and bf16 data types.

- _Winograd_. This algorithm reduces computational complexity of convolution
  at the expense of accuracy loss and additional memory operations. The
  implementation is based on the [**Fast Algorithms for Convolutional Neural
  Networks by A. Lavin and S. Gray**](https://arxiv.org/abs/1509.09308). The
  Winograd algorithm often results in the best performance, but it is
  applicable only to particular shapes. Moreover, Winograd only supports
  int8 and f32 data types.

- _Implicit GEMM_. The convolution operation is reinterpreted in terms of
  matrix-matrix multiplication by rearranging the source data into a
  [scratchpad memory](@ref dev_guide_attributes_scratchpad). This is a fallback
  algorithm that is dispatched automatically when other implementations are
  not available. GEMM convolution supports the int8, f32, and bf16 data types.

#### Direct Algorithm

Intel MKL-DNN supports the direct convolution algorithm on all supported
platforms for the following conditions:

- Data and weights memory formats are defined by the convolution primitive
  (user passes `any`).

- The number of channels per group is a multiple of SIMD width for grouped
  convolutions.

- For each spatial direction padding does not exceed one half of the
  corresponding dimension of the weights tensor.

- Weights tensor width does not exceed 14

In case any of these constraints are not met, the implementation will silently
fall back to an explicit GEMM algorithm.

#### Winograd Convolution

Intel MKL-DNN supports the Winograd convolution algorithm on systems with
Intel AVX-512 support and above under the following conditions:

- Data and weights memory formats are defined by the convolution primitive
  (user passes `any` as the data format).

- The spatial domain is two-dimensional.

- The weights shape is 3x3, there are no groups, dilation or strides (\f$kh =
  kw = 3\f$, and \f$sw = sh = 1\f$, \f$dw = dh = 0\f$).

- The data type is either int8 or f32.

In case any of these constraints is not met, the implementation will silently
fall back to the direct algorithm.

The Winograd convolution algorithm implementation additionally chooses tile
size based on the problem shape and
[propagation kind](@ref mkldnn_prop_kind_t):

- For `forward_inference` Intel MKL-DNN supports
  \f$F(2 \times 2, 3 \times 3)\f$ or
  \f$F(4 \times 4, 3 \times 3)\f$

- Intel MKL-DNN supports only \f$F(4 \times 4, 3 \times 3)\f$ Winograd for all
  the training propagation kinds.

The following side effects should be weighed against the (potential)
performance boost achieved from using the Winograd algorithm:

- _Memory consumption_. Winograd implementation in MKL-DNN requires additional
  scratchpad memory to store intermediate results. As more convolutions using
  Winograd are added to the topology, the amount of memory required can grow
  significantly. This growth can be controlled if the scratchpad memory can be
  reused across multiple primitives. See @ref dev_guide_attributes_scratchpad
  for more details.

- _Accuracy_. In some cases Winograd convolution produce results that are
  significantly less accurate than results from the direct convolution.

Create a Winograd convolution by simply creating a convolution descriptor
(step 6 in [simple network example](@ref cpu_cnn_inference_f32_cpp) specifying
the Winograd algorithm. The rest of the steps are exactly the same.

~~~cpp
auto conv1_desc = convolution_forward::desc(
    prop_kind::forward_inference, algorithm::convolution_winograd,
    conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
    conv1_strides, conv1_padding, padding_kind::zero);
~~~

#### Automatic Algorithm Selection

Intel MKL-DNN supports `mkldnn::algorithm::convolution_auto` algorithm that
instructs the library to automatically select the *best* algorithm based on
the heuristics that take into account tensor shapes and the number of logical
processors available.  (For automatic selection to work as intended, use the
same thread affinity settings when creating the convolution as when executing
the convolution.)


@anchor dg_conv_impl_limits
## Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. **CPU**
   - Winograd are implemented only for Intel(R) AVX-512 or
     Intel(R) AVX512-DL Boost instruction sets

3. **GPU**
    - No support for Winograd algorithm


## Performance Tips

- Use #mkldnn::memory::format_tag::any for source, weights, and destinations
  memory format tags when create a convolution primitive to allow the library
  to choose the most appropriate memory format.
