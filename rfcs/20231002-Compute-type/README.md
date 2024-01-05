Proposal for weight decompression

# 1. Problem statement

There is increasing demand to support weight decompression as it is
proven to provide enough accuracy while reducing memory capacity and
bandwidth requirement for some workloads.  In particular, int8 storage
and bf16/f16 compute seems to provide a good balance between size
reduction, performance and accuracy.

Such up-conversions require to consume quantization parameters for
these inputs. This RFC discusses the API as well as the numerics of
adding support for weights decompression.

A note on potential extension: openVINO expressed interest for future
extension to allow f32->int8 down conversion (e.g. fusing first
quantization reorder to first layer). So we can keep that in mind when
picking an option for weights decompression support.

# 2. Numerics considerations

oneDNN API already has support to specify quantization parameters for
each input/output tensor separately.  When the storage and compute
type for a tensor are integral, the current semantic is to use integer
arithmetic whenever possible (the op is typically computed exactly in
integer), and compensation is added afterwards to account for
quantization.

When tensor storage type is integral but compute type is
floating-point, it is expected that quantization parameters will be
applied before the primitive main operation:
- up-conversion will happen anyway before the main op, we might as
  well apply the quantization parameters
- this has the advantage of predictability from user perspective and
  ability to avoid overflows for example.

# 3. API extensions

# 3.a Implicitly apply up-conversion for int8 weights and floating-point activations.

The current library behavior is that the compute type is inferred by
the weights type. In other words, int8 weights imply using integer
arithmetic for primitive computation. We hence cannot change that
behavior to enable weights up-conversion, since it breaks the current
semantic of the library.

# 3.b. `force_fpmath` knob

This option just adds a boolean `force_fpmath` knob to the
primitive descriptor constructor. When set to false (default), the
behavior is the same as today (weights type and `fpmath_mode` dictate
compute type).  When set to `true`, int8 inputs are up-converted based
on `fpmath_mode` attribute (so `f32` by default). This option has the
benefit to be very simple and apply only to primitives that will
support up-conversion.

The downside of this approach though is that it cannot be extended to
support down-conversion to int8 data-type.

# 3.c Extend `fpmath_mode` with a more general `math_mode` (recommended)
Currently, `fpmath_mode` attribute allows the user to specify
down-conversions of the inputs from f32 to specific datatype.  Here we
would extend `fpmath_mode`:
- to apply to non floating point data-types as input, in particular u8/s8.
- to allow up-conversions from u8/s8 to a particular floating-point
  type. For example, passing u8 weights, bf16 activations and
  `math_mode=bf16` to a matmul primitive should convert its weights to
  bf16 and then compute.

Currently, `fpmath_mode` is ignored for primitives with integral input
tensors. Adding integral type support hence can be considered either
as an extension or as a breaking change.  If we consider it breaking,
this option would require to deprecate `fpmath_mode` in favor of the
new `math_mode`.

Now to avoid weird conversions and numerical behavior, `math_mode`
would behave as follow:
- up-conversion is possible only from u8/s8 to a floating-point
  datatype (typically f16/bf16)
- down-conversion is possible only to a "compatible" floating-point
  datatype (f32->f16, f32->tf32, f32->bf16). To allow for future
  extension (e.g. down-conversion to integral type), we would have to
  properly document that the attribute is not ignored in that
  case, but simply return unimplemented.
- other configurations should not be ignore like they currently are,
  but should return unimplemented (e.g. conversions between two types
  that are the same size like f16<->bf16, or down conversions to
  int8).

Pros: 
- very little changes to API.
- flexible enough to support practical use cases
- compatible with current implicit down-conversion semantics (the
existing default `fpmath_mode` would affect only primitive with
floating-point `math_mode=f32`).
- can be extended to support integral down-conversion
(f32/f16->int8). This would require to specify multiple data-types in
`math_mode` (e.g. `math_mode=u8s8`, `math_mode_s8u8`, `math_mode_s8`,
...) which would require properly documenting which input is mapped to
which type for each primitive supporting it.

Cons:
- need to deprecate `fpmath_mode` to supersede it with `math_mode`.


Example for weights decompression.
```c++
memory::desc src_md({M, K}, memory::data_type::f32, {K, 1});
memory::desc wei_md({K, N}, memory::data_type::s8, memory::format_tag::any);
memory::desc dst_md({M, N}, memory::data_type::f32, {N, 1});

primitive_attr attr;
attr.set_scales_mask(DNNL_ARGS_WEIGHTS, weights_scales_mask);
attr.set_zero_points_mask(DNNL_ARGS_WEIGHTS, weights_zero_points_mask);

// default fpmath_mode should fail to create the primitive
// as it is ambuiguous if weights need upconversion or src need downconversion
auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);

// this will upconvert/downconvert weights/src to bf16 and compute in bf16
attr.set_fpmath_mode(fpmath_mode::bf16);
auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
```

# 3.d Introduce compute type to primitive_desc constructor

This option relies on passing the compute type to the primitive
descriptor constructor directly instead of an attribute.

The main benefits of this approach are:
- it is more explicit from API which primitive kind supports weights
  decompression (e.g. matmul and convolution)
- we can specifically spell out the compute type for each input tensor
  (e.g. `s8s8s32`).
- it can be extended to support down-conversion to integral datatype
  (e.g. f32->int8).

The main downside here is how it interacts with `fpmath_mode`, for
example, if a user passes `compute_type=f32` and `fpmath_mode=bf16`.
There are a couple of options here:
- The `compute_type` parameter takes precedence over
  `fpmath_mode`. This works well when user wants to do explicit
  up/down conversions. The risk though is if user always specify
  compute type (e.g. setting `compute_type=f32` for all f32
  primitives), in which case the library implicit down-convert feature
  will be disabled application wide. To avoid that, we can allow users
  to specify non-default `compute_type` only when integral inputs are
  specified.
- The `fpmath_mode` takes precedence when `compute_type` is a
  floating-point type. This can cause unexpected behavior when user is
  doing explicit weights decompression (e.g. weights are s8,
  `compute_type=bf16` and `fpmath_mode=strict|f32`).

If this option gets traction, I would recommend to make `compute_type`
take precedence, and allow it to be non-default only when one of the
inputs is integral.

# 3.e Introduce `intmath_mode` attribute

Comparing to the 3.c this option separates math_mode into 2 independent
compute modes for integer and floating primitives. The reason is that
while `fpmath_mode` is a hint to speed-up the computations,
`intmath_mode` is a requirement to compute in higher precision to
preserve accuracy.

The main benefit of this approach is that we split these 2 different
features into 2 independent APIs to make them simpler for both
implementations and users.

The downside is it is not always clear if a primitive has integral or
floating-point compute data type, especially if it does not have weights
(for example, normalization) or has multiple weights with different
data types (for example, RNN). We will have to document which primitives
rely on math mode and which ignore it.

Example for weights decompression.
```c++
memory::desc src_md({M, K}, memory::data_type::f32, {K, 1});
memory::desc wei_md({K, N}, memory::data_type::s8, memory::format_tag::any);
memory::desc dst_md({M, N}, memory::data_type::f32, {N, 1});

primitive_attr attr;
attr.set_scales_mask(DNNL_ARGS_WEIGHTS, weights_scales_mask);
attr.set_zero_points_mask(DNNL_ARGS_WEIGHTS, weights_zero_points_mask);

// default fpmath_mode should fail to create the primitive
// as it is ambuiguous if weights need upconversion or src need downconversion
auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);

// this will upconvert/downconvert weights/src to bf16 and compute in bf16
attr.set_intmath_mode(fpmath_mode::bf16);
auto pd = matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
```

# 4. Summary
We would recommend to go with option 3.c, as it allows to preserve
current implicit down-conversion in a natural way, and can be extended
to support down-conversion to integral types.  If the belief is that
this extension will not be needed, we can go with option 3.b, as it is
very simple to implement and use from user perspective (and does not
necessitate deprecating APIs).
