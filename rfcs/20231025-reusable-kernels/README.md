# Proposal to add a reusable kernel knob

## Introduction

Several models in use by frameworks have dynamic shapes, meaning that there are
many unique buffer shapes in use in different primitives throughout the model.
In these cases, primitive creation time can be significantly longer than
execution time. To address primitive creation time, oneDNN uses a primitive
cache, a persistent cache, and a kernel cache. However, most existing
implementations compile the shape of the buffers into the kernel, which
prohibits them from being reused via the kernel cache. Although the goal is to
convert all implementations to use the kernel cache, we need a near-term
solution to prioritize the use of kernel cache-friendly implementations.

Although reusable kernels can help mitigate primitive creation time by
amortizing it across several primitive executions, in general it's difficult
to develop reusable kernels that achieve execution time parity with existing
kernels. Passing values at runtime gives the compiler less information with
which to optimize, so designing reusable kernels in such a way that this does
not affect performance can be difficult.

As such, the process to achieving full reusability of all kernels with the same
performance as existing kernels is likely to take some time to realize. In the
meantime, frameworks have requested an option to prioritize dispatching to
reusable kernels over the typical performance-optimized kernels. This proposal
is to add a knob the user can control to specify whether they would like
reusable kernels to be dispatched preferentially over the standard
performance-optimized kernel ordering.

> **Note:** Reusable kernels are defined as kernels which have internally
opted-into the kernel cache API, and support a notable subset of primitive
configurations with a single compilation. Reusable kernels may depend on
problem shape, but only in a limited way. For example, a reusable kernel may
impose a divisibility requirement (e.g. input channels must be a multiple of
16), but the specific shape cannot be compiled directly into the kernel.

## Proposal

The proposal has four potential options: using DNNL_RUNTIME_DIM_VAL, a build
flag, an environment variable, or a primitive attribute. In any of the options,
the default behaviour will be the current one: kernels are ordered according to
internal heuristics, roughly in order of decreasing likelihood for optimal
performance for the given problem.

If the reusable kernel option is selected, kernels designed to be reusable will
be selected first. In the case that no reusable kernels are compatible with the
problem, kernel selection will continue with non-reusable kernels.

Option 4 (primitive attribute) is the preferred option by frameworks, as they
don't know which dimensions should be considered dynamic at runtime. It
provides a balance between user control, flexibility, and ease of integration.
Compared to the build flag it's more flexible, and compared to the environment
variable it's less opaque.

### Option 1: Using `DNNL_RUNTIME_DIM_VAL`

With this option, the user will select one or more dimensions to have the
`DNNL_RUNTIME_DIM_VAL` special value during primitive creation. oneDNN will
then dispatch to the first implementation that supports the given set of
runtime dimensions. One downside to this approach is that the implementation
must either compile a kernel that's general enough to support any possible
value for the runtime dimension (which limits optimization potential) or it
must identify and compile all specialized kernels which support the given set
of runtime dimensions (which would increase primitive creation time).

In addition to existing limitations on primitives that support runtime
dimensions, feedback from frameworks has been that which dimensions should be
specified as `DNNL_RUNTIME_DIM_VAL` is unknown at primitive creation time.
Between these two factors, this option is unlikely to lead to a satisfactory
solution, at least in the short term.

<u>Pros:</u>
- The most fine-grained approach
- Ties into existing mechanisms
- Can be toggled without a rebuild

<u>Cons:</u>
- Several primitives have no implementations that support runtime dims -
cannot be adopted until such implementations have been developed
- Support for all possible runtime dim configurations is difficult
- Requires users to know which dimensions will have dynamic shapes ahead of
time

### Option 2: Build flag

With this option, the build flag `ONEDNN_PREFER_REUSABLE_KERNELS` would be
added to select the preferred functionality. Default is `off`, and reusable
kernels will be chosen preferentially when set to `on`.
<u>Pros:</u>
- Easy integration

<u>Cons:</u>
- Requires a rebuild to change the option
- The setting is not apparent in the code, leads to opaque behaviour
- Requires internal changes to detect implementation reusability

### Option 3: Environment variable

With this option, the user can set `ONEDNN_PREFER_REUSABLE_KERNELS=1` as an
environment variable, which will cause the library to select reusable kernels
preferentially. This option makes for easy integration, and provides a way to
expand upon the functionality in the future (by setting a different value), for
example if different levels of "reusability" are defined.
<u>Pros:</u>
- Easy integration
- Allows for future expansion
- Can be toggled without a rebuild

<u>Cons:</u>
- The setting is not apparent in the code, leads to opaque behaviour
- Requires internal changes to detect implementation reusability

### Option 4: Primitive attribute (preferred)

With this option, the following API functions would be added:

```C++
status_t dnnl_primitive_attr_get_hint_reusable_kernels(
        const primitive_attr_t *attr, bool *option);
status_t dnnl_primitive_attr_set_hint_reusable_kernels(
        const primitive_attr_t *attr, bool option);
```
This value defaults to `false` - only when the user sets this to `true` will
reusable kernels be selected preferentially. 
<u>Pros:</u>
- Middle-ground between fine-grained control and user knowledge of the library
internals.
- Can be toggled without a rebuild

<u>Cons:</u>
- Requires internal changes to detect implementation reusability
