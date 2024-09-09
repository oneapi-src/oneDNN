Primitive Attributes: rounding mode {#dev_guide_attributes_rounding_mode}
===================================================================

The default numerical behavior of oneDNN with respect to rounding
(described in @ref dev_guide_data_types) can be altered to allow the
use of non-standard rounding mode upon down-conversion.

When passed to a primitive descriptor creation, the @ref
dnnl::rounding_mode primitive attribute specifies which rounding mode
to use when a given argument is down-converted.

The @ref dnnl::rounding_mode primitive attribute accepts:
- `environment` (default): For floating-point primitives (as defined
  in @ref dev_guide_data_types), the rounding behavior is controlled
  by the C/C++ floating-point environment non-SYCL/non-OpenCL devices,
  or by SYCL/OpenCL language defaults otherwise.
- `stochastic`: In this mode, a random bias is generated in a
  deterministic way, and gets added to the mantissa bits of the
  destination before truncation to destination datatype precision.
  The seed for the random number generator has to be passed as an
  execution argument with `DNNL_ARG_ATTR_ROUNDING_SEED` argument
  kind. This is a single-value `s32` input buffer.

The stochastic rounding mode is of particular interest to avoid
vanishing gradients during low-precision training.  As such, it is
only supported for \diffsrc and \diffweights arguments for
primitive supporting propagation kind, and for \dst for Matmul
primitive.

