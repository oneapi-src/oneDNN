Proposal for setting accumulation datatype.

# 1. Introduction

In order to maintain good accuracy in DNN workloads, it is important
to use wide-enough accumulators during computation.  The default
accumulation datatype in oneDNN is f32, for floating point
computations, and s32 for integer computation.

Some platforms allow additional speedups when using smaller
accumulation datatype.  This is in particular the case for f16
primitives, where up to 2x speedups can be obtained by using f16
accumulation datatype on some platforms.

# 2. Problem statement

As of now, oneDNN officially supports only inference for f16 models,
and many data show that good levels of accuracy can be maintained
while using f16 accumulation datatype for most f16 models. We hence
have been dispatching implementations with f16 accumulation on some
platforms.

Even though dispatching implementations with f16 accumulation is
acceptable for inference, it is not for training, which requires the
full f32 precision accumulators. Hence, as oneDNN is planning
support for f16 training, it is important that we dispatch only
implementation with f32 accumulation for training workloads.

Note that the main user for oneDNN f16 implementations is openVINO,
which is the requester for the implementations with f16 accumulation
in the first place.


# 3. Options

## Using the propagation kind

One natural option would be to rely on propagation kind.  If the
propagation kind, is `forward_inference`, we allow dispatching of
implementations with f16 accumulation, otherwise we don't.  This works
for most primitives except for Matmul, which does not have a
propagation kind. Hence this option does not cover the whole library.

## Using the `ONEDNN_ENABLE_WORKLOAD` build option

oneDNN currently has a `ONEDNN_ENABLE_WORKLOAD` build option that can
be set to `INFERENCE` or `TRAINING` (default). This option is
currently used to disable dispatching of some primitive
implementations based on propagation_kind. We could rely on this
mecanism to guard dispatching of implementations using f16
accumulation with the two following caveats:
- this would increase validation costs as we would need to validate both builds
- it adds development burden. Currently we don't know any information
  on the accumulation datatype for each implementations. Properly
  guarding implementation dispatch would either need to be scattered
  in implementation using f16 accumulation (which is brittle), or
  would require to add internal APIs to query implementations for their
  accumulation datatype.

## A new `accumulation_datatype` attribute (recommended)

The last option is to rely on a new primitive attribute to get a hint
on minimal accuracy required in accumulators. There are multiple
options for what values it should accept.

For all those options, note that implementations would still be able
to use a different datatype internally if the result is unchanged and
it is faster (e.g. using `f32` intermediate accumulators when
accumulation datatype is `f16`, or in some `s8/u8` computations when
it yields the same result).

### Use a datatype (recommended).

In those cases, the user will have to explicitely pass `f32`, `f16`,
`s32`, ...  When the user does not set the attribute, we will use a
default value of `undef`, which would allow all implementation to be
dispatched, including those with `f16` accumulation.

Pros:
- no new datatype introduced
- the user explicitly passes the datatype

Cons:
- not fully clear from API that it is a hint

### Use a new accumulation mode

Here, we would define a new datatype

```c++
enum class accumulation_mode {
    /// Default behavior, uses accumulators with 32-bit precision 
    accurate = dnnl_accumulation_mode_accurate,
    /// Might use lower accuracy accumulators that were tested to work for most models
    fast = dnnl_accumulation_mode_fast,
};
```

The pros and cons are the exact opposite of using datatypes.

Pros:
- Clear from API that it is a hint
Cons:
- the user does not know exactly which datatype is used for accumulation
- it introduces another datatype.

### A note on the default behavior.

Note that for both options, we can use the `ONEDNN_ENABLE_WORKLOAD` to
set change the default behavior.  For the datatype flavor, `undef`
would enable `f16` accumulation dispatch for
`ONEDNN_ENABLE_WORKLOAD=INFERENCE` but not for
`ONEDNN_ENABLE_WORKLOAD=TRAINING`.  For the mode flavor, default would
be `fast` for `ONEDNN_ENABLE_WORKLOAD=INFERENCE`, and accurate for
``ONEDNN_ENABLE_WORKLOAD=TRAINING`.

This would allow to run full validation on a single build, and let the
current openVINO/ONNXruntime integrations unchanged, as they use
`ONEDNN_ENABLE_WORKLOAD=INFERENCE`.  Though it will require extra
validation still (build + interface test for default value).

If we elect to go with a fixed default behavior, openVINO and
ONNXruntime integrations will need to explicitely pass the new
attribute to not get performance regressions on some platforms.

