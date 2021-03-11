Proposal for implicit reduced precision arithmetic

# 1. Introduction

Many customers are showing interested in trying to run their topology
with reduced precision, without modifying their model implementation.
This can provide speedups, without compromising accuracy in some
cases, and can also be useful to quickly assess if a model can be run
with lower precision without losing too much accuracy.

One extra flexibility would be to offer scoped reduced precision, so
user would be able to enable lower precision only for sub-graphs in
the whole topology in order to control accuracy loss.

It has to be noted that the implicit down-convert feature is planned
to be exposed to end-user (FWK or toolkit users) as both a global
setting and/or scoped to a sub-graph.

# 2. Main considerations for the feature behavior

Before making a proposal of API changes, let's first consider a few
traits that will define the behavior of that feature.

## Scoped vs global setting
A global setting would be a library state, possibly controlled with an
environment variable and/or a function call. The benefit is that it is
easy to use and would require very little change to user
code. However, a scoped setting is necessary if some users want to
allow lower precision for selected portions of the graph.

Proposal: use scoped setting, with a global (`thread_local`)
default. This will allow the scoped use case the framework would
expose, and also allow users to test implicit down-conversion without
any framework change (with environment variable).

## Allow only "compatible" down-conversion vs any data-type.

Here compatible data-types are data-types that are typically used in
the same way, and that are at least as precise. This excludes implicit
conversion to/from any integer data-type, since integer data-types
typically require quantization parameters that depend on the
computation data-type. For floating point data-types
- `bf16` and `f32` are compatible for both training and inference,
- and `f16` and `f32` only for inference since `f32` and `f16` are not
  typically interchangeable for training.

Proposal: support only down-conversion to "compatible"
data-types. This will allow to restrict the feature to only commonly
used combinations and would make internal logic and validation
simpler. Here we would return `invalid_argument` if users pass
incompatible data-type for down-conversion hint/requirement, to help
them debug incorrect logic.

## Down-conversion policy: requirement vs hint.
Here the question is if the user specifies a target type to
down-convert to, should we treat it as a requirement (we can
down-convert only to this type) or should it be a hint (we can
down-convert to a type other than the one specified as long as it is
"compatible").

Proposal: support only hints, to allow library for optimal dispatching
without user needing to tweak down-conversion policy for better
performance. Also, this is more scalable as we introduce new
data-types (e.g. if we were to introduce `tf32`).

## Per tensor vs per primitive setting
This would be the granularity at which the user would allow conversion
to happen. Per tensor seems to have the most flexibility, however it
is unlikely that users would need this level of flexibility. For the
per-primitive setting, the implicit conversion will be assumed to
happen only for input tensors, and not output.

Proposal: support only per primitive setting. The main points against
per tensor are: a) it would require more user changes to enable this
feature and b) the implicit down-convert feature is planned to be
exposed to end-user (FWK or toolkit user), and there is no plan to
expose any fine grain control there (it will be global or scoped to a
sub-graph).

## Allow down-conversion through reorder vs hide it in execution

This is basically deciding what data-type should be returned when a
primitive is queried for one of its memory descriptor when implicit
down-conversion happens. One option is to preserve the data-type
passed by user so that down-conversion is fully hidden and
implicit. The other option is to return a memory descriptor with the
data-type that will be used during computation. The first option has
the benefit to completely hide the down-conversion and does not
require the user to add logic to handle potential down-conversion
through reorder if creation data-type is different than queried
data-type. The second option has the benefit of allowing to
down-convert ahead of time and reduce the execution time (e.g. weights
down-conversion can be computed ahead of time during inference).

Proposal: When queried for a `memory_desc`, a primitive is allowed to
return a `memory_desc` with the data-type it down-converts to for
computation only when the corresponding user provided `memory_desc`
has data layout `any`. The rationale is that when the user does not
pass `any`, we cannot assume that they properly handle a potential
reorder for that given tensor. However, when they pass `any`, we can
assume that a memory_descriptor comparison happens (using
`dnnl_memory_desc_equal` or `memory::desc::operator==`) and proper
reorder logic is implemented.

Note: This is not mandated for a primitive implementation to expose
the data-type they down-convert to when queried. There are some
scenario where this is beneficial to not expose it actually. For
example, we might want to return plain layouts for activation when
user passes `any` in order to reduce potential reorder overhead. In
those same cases, we would likely preserve the user datatype in the
queried `memory_desc` and handle down-conversion internally (not
through reorder).

Example: here is an example of what would be the behavior for
convolution:
```c++
auto conv_desc = convolution_forward::desc(prop_kind::forward,
    algorithm::convolution_direct, src_md, weights_md, dst_md,
    strides, padding_l, padding_r);
auto conv_pd
    = convolution_forward::primitive_desc(conv_desc, engine);
auto impl_src_md = conv_pd.src_desc();
auto impl_weights_md = conv_pd.weights_desc();
auto impl_dst_md = conv_pd.dst_desc();
```

In that snippet, `src_md`, `weights_md` and `dst_md` are the memory
descriptors passed by the user to create the convolution primitive.
And `impl_src_md`, `impl_weights_md` and `impl_dst_md` are the
corresponding memory descriptors returned by the primitive
implementation when queried.

The table below summarizes what the primitive is allowed to return
when queried. To simplify, we use `id` for `implementation defined`.
| `<user_md>` | `<user_md>` tag | `<user_md>.data_type()` | `<impl_md>` tag | `<impl_md>.data_type()` ( downconvert disabled) | `<impl_md>.data_type()` ( downconvert enabled) |
|-------------|-----------------|-------------------------|-----------------|-------------------------------------------------|------------------------------------------------|
| src_md      | any             | f32                     | id              | f32                                             | id                                             |
|             | nhwc            | f32                     | nhwc            | f32                                             | f32                                            |
| weights     | any             | f32                     | id              | f32                                             | id                                             |
|             | oihw            | f32                     | oihw            | f32                                             | f32                                            |
| dst         | any             | f32                     | id              | f32                                             | id                                             |
|             | nhwc            | f32                     | nhwc            | f32                                             | f32                                            |

# 3. API changes proposal

The following API proposals assume the above recommendation for the
feature behavior. However, they could be adapted accordingly if a
different behavior is agreed upon.

## 3.1 New primitive attribute.

As usual, we would introduce the setter and getter for the FP math mode attribute.

```c++
dnnl_status_t DNNL_API dnnl_primitive_attr_set_fp_math_mode(
        dnnl_primitive_attr_t attr, dnnl_fp_math_mode_t mode);

dnnl_status_t DNNL_API dnnl_primitive_attr_get_fp_math_mode(
        dnnl_primitive_attr_t attr, dnnl_fp_math_mode_t *mode);
```

The math mode attribute will have the following values:
```c++
typedef enum {
    dnnl_fp_math_mode_strict, // default behavior, described in doc
    dnnl_fp_math_mode_bf16,   // implicit f32->bf16 conversion allowed
    dnnl_fp_math_mode_f16,    // implicit f32->f16 conversion allowed
    dnnl_fp_math_mode_any,    // implicit f32->f16 or f32->bf16 conversion allowed
} dnnl_fp_math_mode_t;
```

As mentioned in the previous section, these will apply only to
primitives created with floating-point computation. Using any value
other value than `dnnl_fp_math_mode_strict` for primitives with
integer computation data-type should return `invalid`. As a first
step, we would allow implicit down-conversion only to formats that are
at least as accurate as the format specified in the math mode (so same
number or greater number of mantissa bits, and same or greater number
of exponent bits). In particular, we would have this relationships:
- `f32 > tf32 > f16`
- and `f32 > tf32 > bf16`.

Note that `bf16` and `f16` are not comparable since `bf16` has a wider
range (more exponent bits) but less accuracy (fewer mantissa
bits). Because `f32` is the only FP type that is comparable to all
other FP types, we suggest to restrict the implicit down-convert
feature only when the original format is `f32` (this should be the
most common use-case anyway).  We could later extend it to also
support implicit down-conversion from any FP data-type (hence why the
enum and attribute name mentions `fp` and not `f32`).

We could reuse the `dnnl_data_type_t` enum type instead of introducing
a `dnnl_fp_math_mode_t` enum type, but I believe it is better to keep
them separate as
1. it does not make sense to have `dnnl_datatype_any` for data_type
2. and we don't want to support integer types.

## 3.2 New environment variable and associated function.

To allow end-users to experiment with down-conversion without depending
on FWK/toolkit knobs, we would introduce a `DNNL_DEFAULT_FP_MATH_MODE`
environment variable. This would change the `default` value for the
math mode attribute. The accepted values would be:
- `STRICT` to disable implicit down-conversion. This would be the
  default for v2.x to preserve current behavior. The default could be
  changed in future major versions.
- `BF16`, `F16` to allow implicit down-conversions from `f32` to
  `bf16`/`f16` or compatible FP type.
- `ANY` where we allow down-conversion from `f32` to any compatible
  data-type with lower accuracy.

As usual, we would introduce the corresponding function APIs, which
supersedes the environment variable.

```c++
dnnl_status_t DNNL_API dnnl_set_default_fp_math_mode(dnnl_fp_math_mode_t mode);
```

Because some users create primitives in parallel, this setting should
be thread local.

## 3.3 Verbose support

There are two kind of information a user might need/want:
1. The value of the attribute passed when not default. This describes
  exactly what was passed to the API and aligns with the other
  attributes.
2. The computation data-type of a primitive when down-conversion
  happens. Since the `fp_math_mode` is a hint, users might want to
  know about when down-conversion effectively happens, vs when it
  could happen.

Supporting both of these seems to be orthogonal usages though: option
1 is useful for creating reproducers, option 2 might be useful for
performance profiling.

Here the proposal is to stick with option 1 through verbose for
now. If a need arise for option 2, we can still extend the verbose
with the primitive computation type.  Regarding the string we would
print, as usual, we have to pick a shortname (e.g. `fpm:strict` for
`dnnl_fp_math_mode_strict`, ...).

