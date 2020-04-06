Primitive Attributes: Post-ops {#dev_guide_attributes_post_ops}
===============================================================

oneDNN implements some basic capabilities of operation fusion using the
**post-ops attributes** API. The operation fusion typically reduces the memory
bandwidth pressure hence leading to higher performance.

The post-ops change the default behavior of a primitive and hence are
implemented through the @ref dev_guide_attributes mechanism.

Currently the following post-ops are supported by the library:

| Post-ops \ Primitive                                              | @ref dev_guide_convolution | @ref dev_guide_inner_product | @ref dev_guide_batch_normalization
| :--                                                               | :--                        | :--                          | :--
| [Eltwise](@ref dev_guide_attributes_post_ops_eltwise)             | Partial                    | Partial                      | Partial
| [Sum](@ref dev_guide_attributes_post_ops_sum)                     | Partial                    | N/A                          | N/A
| [Depthwise](@ref dev_guide_attributes_post_ops_depthwise)         | Partial                    | N/A                          | N/A

Just like @ref dev_guide_attributes, the post-ops are represented by
an opaque structure (@ref dnnl_post_ops_t in C API and @ref dnnl::post_ops
in C++ API) which is copied once it is attached to the attributes using C++
@ref dnnl::primitive_attr::set_post_ops or C
@ref dnnl_primitive_attr_set_post_ops functions.  These attributes then are
passed to a primitive descriptor creation function to take effect. Below is a
simple skeleton for C++ API:

~~~cpp
dnnl::post_ops po; // default empty post-ops
assert(po.len() == 0); // no post-ops attached

po.append_SOMETHING(params); // append some particular post-op
po.append_SOMETHING_ELSE(other_params); // append one more post-op

// (!) Note that the order of appending matters!
assert(po.len() == 2);

dnnl::primitive_attr attr; // default attributes
attr.set_post_ops(po); // attach the post-ops to the attr

// further po changes would not affect attr

primitive::primitive_desc op_pd(params, attr); // create a pd with the attr
~~~

@note
    Different post-ops can be chained together by appending one
    after another. Note that the appending order matters: the sequence of
    the post-ops is executed in the order of appearance.

@warning
    Different primitives have different capabilities on supporting post-ops.
    Moreover, the support might also depend on the actual implementation of a
    primitive. For instance, the library generally doesn't support post-ops
    for reference primitives (which are typically very slow, so there is no
    point in doing the actual fusion). So the robust integration should handle
    errors accordingly. See the 
    [section on attributes error handling](@ref dev_guide_attributes_error_handling).

The post-op object can be inspected by @ref dnnl::post_ops::kind()
function that takes an index of the post-op (that must be less than the
value returned by @ref dnnl::post_ops::len()) and returns it's kind.

## Supported Post-ops

@anchor dev_guide_attributes_post_ops_eltwise
### Eltwise Post-op

The eltwise post-op enables fusing a primitive with a @ref dev_guide_eltwise
primitive. This is probably one of the most popular kinds of fusion:
an eltwise (typically an activation function) with preceding convolution
or inner product.

The @ref dnnl::primitive::kind of this post-op
is #dnnl::primitive::kind::eltwise.

API:
- C: @ref dnnl_post_ops_append_eltwise
- C++: @ref dnnl::post_ops::append_eltwise

The parameters (C++ API for simplicity):
~~~cpp
void dnnl::post_ops::append_eltwise(
        float scale, // scaling factor (described below)
        algorithm alg, float alpha, float beta // same as in Eltwise primitive
        );
~~~

The `alg`, `alpha`, and `beta` parameters are the same as in @ref dev_guide_eltwise.

The Eltwise post-op replaces:
\f[
    \dst(:) = \operatorname{Op}(...)
\f]

with

\f[
    \dst(:) = scale \cdot \operatorname{Eltwise}( \operatorname{Op}(...) )
\f]

The intermediate result of \f$\operatorname{Op}(...)\f$ is not stored. Hence in most of the
case this kind of fusion cannot be used with the training.

The \f$scale\f$ factor is supported in
[INT8](@ref dev_guide_attributes_quantization) inference only. For other
cases the scale must be equal to `1.0`.

@anchor dev_guide_attributes_post_ops_sum
### Sum Post-op

Appends an accumulation (sum) post-op. Prior to accumulating the result, the
previous value would be multiplied by scale.

The kind of this post-op is #dnnl::primitive::kind::sum.

This feature might improve performance for cases like residual learning
blocks, where the result of a convolution is accumulated to the previously
computed activations. The scale parameter can be used in
[INT8](@ref dev_guide_attributes_quantization) inference only when the result
and previous activations have different logical scaling factors.

The sum post-op replaces
\f[
    \dst(:) = \operatorname{Op}(...)
\f]

with

\f[
    \dst(:) = scale \cdot \dst(:) + \operatorname{Op}(...)
\f]

@warning
    This post-op (as well as all the others) disregards the original layout of
    the destination; that is, the layout of the original destination is
    expected to be the same as the layout of the output destination.

@anchor dev_guide_attributes_post_ops_depthwise
### Depthwise Post-op

Appends a Depthwise convolution as a post-op. This post-op can only be fused 
with 1x1 convolution as generally seen in models (like MobileNet_v1) that use a
stack of Separable convolutions: Depthwise convolution followed by 1x1
convolution. The stack of these Separable convolutions (like in MobileNet_v1)
provide an opportunity to fuse 1x1-Convolution with bandwidth-limited Depthwise
convolution.

The @ref dnnl::primitive::kind of this post-op
is #dnnl::primitive::kind::convolution.

There are two variants of this post-op: dw_k3s1p1 and dw_k3_s2p1 for stride-1
and and stride-2 respectively.

API:
- C: @ref dnnl_post_ops_append_dw_k3s1p1 , @ref dnnl_post_ops_append_dw_k3s2p1 
- C++: @ref dnnl::post_ops::append_dw_k3s1p1 , @ref dnnl::post_ops::append_dw_k3s2p1

For better readability, below we assume a 2D convolution and use the following
notations:

  `conv_1x1` Convolution with weights spatial=1 i.e., `kh` = `kw` = 1.

  `conv_dw` Depthwise convolution with weights spatial=3 i.e., `kh` = `kw` = 3, 
  `g` = `oc` = `ic` and `pad_l` = `pad_r` = {1, 1}.

The Depthwise post-op replaces

\f[
    dst(:) = Conv_{1x1}(...)
\f]

with

\f[
    dst(:) = Conv_{dw}(Conv_{1x1}(...))
\f]


The final output dimensions of the after post-op is defined as

\f[
    dst_{conv_dw} = \{ n, oc_{1x1}, \operatorname{ceil}(oh_{conv_{1x1}}/stride),
     \operatorname{ceil}(ow_{conv_{1x1}}/stride) \}
\f]

where `oh_conv_1x1`, `ow_conv_1x1` are height and width of conv_1x1 destination.

![Fusion](images/img_depthwise_fusion.jpg)

Supported data types

| conv 1x1 output data type        | depthwise post-op output data type | depthwise post-op weights data type | depthwise post-op bias data type
| :--                              | :--                                | :--                                 | :--
| u8, s8                           | u8, s8, s32, f32                   | s8                                  | f32, s32
| f32                              | f32                                | f32                                 | f32
| bf16                             | bf16, f32                          | bf16                                | f32, bf16

@note
  * Currently only supported for 2D 1x1 convolution.

  * Only eltwise post-op can be part of post-op chain (i.e., sum post-op is 
    not supported)

  * The `dst_1x1`, `wei_dw` and `dst_dw` are assumed to be #dnnl_format_tag_any.


## Examples of Chained Post-ops

Different post-ops can be chained together by appending one after another.
Note that the order matters: the post-ops are executed in the order they have
been appended.

Let's consider some examples.

### Sum -> ReLU

This pattern is pretty common for the CNN topologies from the ResNet family.

~~~cpp
dnnl::post_ops po;
po.append_sum(
        /* scale = */ 1.f);
po.append_eltwise(
        /* scale     = */ 1.f,
        /* alg kind  = */ dnnl::algorithm::eltwise_relu,
        /* neg slope = */ 0.f,
        /* unused for relu */ 0.f);

dnnl::primitive_attr attr;
attr.set_post_ops(po);

convolution_forward::primitive_desc(conv_d, attr, engine);
~~~

This will lead to the following primitive behavior:

\f[
    \dst(:) = \operatorname{ReLU}(\dst(:) + \operatorname{conv}(\src(:), \weights(:))
\f]


@anchor dev_guide_attributes_post_ops_with_scales
### Tanh -> Sum -> ScaleShift

The hypothetical example to illustrate the sequence of operations applied.
We also set all the scales to non-one to as well as use
@ref dnnl::primitive_attr::set_output_scales which will be covered
in @ref dev_guide_attributes_quantization.
Unfortunately (or fortunately) the sequence is not supported by the library
and is merely used to illustrate the semantics of post-ops.

~~~cpp
dnnl::post_ops po;
po.append_eltwise(
        /* scale     = */ s_tanh,
        /* alg kind  = */ dnnl::algorithm::eltwise_tanh,
        /* unused for tanh */ 0.f,
        /* unused for tanh */ 0.f);
po.append_sum(
        /* scale     = */ s_sum);
po.append_eltwise(
        /* scale     = */ s_linear,
        /* alg kind  = */ dnnl::algorithm::eltwise_linear,
        /* scale     = */ alpha,
        /* shift     = */ beta);

dnnl::primitive_attr attr;
attr.set_output_scales(0, {s_conv});
attr.set_post_ops(po);

convolution_forward::primitive_desc(conv_d, attr, engine);
~~~

This will lead to the following primitive behavior (for better readability
the tensors are designated by their names only; i.e., `(:)` is omitted):

\f[
    \dst
        =
        s_{linear} \cdot
        (
            \alpha \cdot
            (
                s_{sum} \cdot \dst
                +
                s_{tanh} \cdot \tanh
                (
                    s_{conv} \cdot \operatorname{conv}(\src, \weights)
                )
            )
            + \beta
        )
\f]


@anchor dev_guide_attributes_post_ops_depthwise_fusion
### Relu -> Depthwise -> Relu

An example of fusing depthwise convolution with 1x1 convolution in MobileNet.

~~~cpp
dnnl::post_ops po;

po.append_eltwise(
        /* scale     = */ 1.f,
        /* alg kind  = */ dnnl::algorithm::eltwise_relu,
        /* neg slope = */ 0.f,
        /* unused for relu */ 0.f);

po.append_dw_k3s1p1( /* or po.append_dw_k3s2p1 for depthwise with stride=2*/
        /* depthwise weights data type = */ dnnl::memory::data_type::s8,
        /* depthwise bias data type (undef implies no bias) = */ dnnl::memory::data_type::undef,
        /* depthwise destination data type = */ dnnl::memory::data_type::u8,
        /* mask for output scales of depthwise output = */ mask,
        /* output scales for depthwise output = */ scales_depthwise)

po.append_eltwise(
        /* scale     = */ 1.f,
        /* alg kind  = */ dnnl::algorithm::eltwise_relu,
        /* neg slope = */ 0.f,
        /* unused for relu */ 0.f);

dnnl::primitive_attr attr;
attr.set_output_scales(0, {output_scales_1x1_conv});
attr.set_post_ops(po);

auto cpd = convolution_forward::primitive_desc(conv_1x1, attr, engine);
auto dw_weight_md = cpd.query(query::exec_arg_md,
                DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
auto dw_bias_md = cpd.query(query::exec_arg_md,
                DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS);

~~~

This will lead to the following primitive behaviour:

\f[
    dst 
        = 
        ReLU_{depthwise}
        (
            scales_{depthwise} \cdot
            (
                conv_{depthwise}
                (
                    ReLU_{1x1}
                    (
                        scales_{conv_{1x1}} \cdot
                        (
                            conv_{1x1}()
                        )
                    )
                )
            )
        )
\f]
