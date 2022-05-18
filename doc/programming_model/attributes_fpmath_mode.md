Primitive Attributes: floating-point math mode {#dev_guide_attributes_fpmath_mode}
===================================================================

For some applications, it can be beneficial to allow down-conversions to
speedup computations without noticeable impact on accuracy.

This section describes how the default numerical behavior of
oneDNN (described in @ref dev_guide_data_types) can be altered to
allow implicit down-conversions of floating-point types.

## The floating-point math mode attribute.

When passed to a primitive creation, the @ref dnnl::fpmath_mode
primitive attribute specifies which implicit down-conversions are
allowed for that given primitive. Only down-conversions from f32 to
narrower data-types (f16, bf16, or tf32) are currently allowed. Furthermore
these down-conversions are allowed only during computation, and do not
affect the storage datatype (which must remain f32).

The @ref dnnl::fpmath_mode primitive attribute can take 3 types of values:
- the `strict` mode disables any down-conversion.
- the `any` mode allows all conversions from f32 to a smaller
  floating-point datatype (f16, bf16, or tf32).
- a specific datatype (f16, bf16, or tf32) which specifically allows
  down-conversion only from f32 to a datatype at least as accurate as
  the specified data-type (at least same number of exponent and
  mantissa bits).

This attribute is ignored if a primitive computation data-type is
integral.

## A note on default floating-point math mode

The default floating-point mode is `strict`, which means no implicit
down-conversion is allowed.  However, this default behavior can be
changed with the `ONEDNN_DEFAULT_FPMATH_MODE` environment variable, the
@ref dnnl_set_default_fpmath_mode (C API) or the @ref
dnnl::set_default_fpmath_mode (C++ API) functions.

@note
For builds where Arm Compute Library is enabled, setting
`ONEDNN_DEFAULT_FPMATH_MODE` to `BF16` or `ANY` will instruct Compute Library to
dispatch bfloat16 kernels where available, provided the hardware supports
bfloat16 instructions. _Note: this may introduce a drop in accuracy._
