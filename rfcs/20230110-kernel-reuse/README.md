# Proposal for Primitive Compilation in oneDNN

## 1. Introduction
Slow primitive compilation consistently causes customer issues in oneDNN. To
handle this, we introduced a [global primitive
cache](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20200207-global-primitive-cache)
and a [persistent cache
API](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20210624-persistent-cache).
These features significantly increase primitive creation performance when oneDNN
users create the same primitive many times. This is the case for many workloads
and PyTorch, TensorFlow, and OpenVino have seen significant benefit. The
assumption primitives will be reused enough to justify their creation is proving
insufficient, especially for the GPU software stack. In particular, customers
have provided workloads which dynamically change based on previous computations.

In addition, even the resources in OpenCL C first compilation alone are an
issue. As an example, GPU OpenCL C kernels take around 500ms to compile. Some
fast running primitives, such as some eltwise primitives, can take on the order
of 0.04ms to execute. Because of this, primitives need reused at least `12,500`
times for compute time to match the time generating the primitive. Recently, we
encountered a first compilation issue brought to us by OpenVino in relation to
compiling an OpenCL binary to query some of GPU device capabilities. While this
particular scenario was resolved modifying our persistent cache API, similar
issues could occur when the persistent cache is infeasible. In addition, we are
encountering the first compilation issue in the oneDNN testing process. It is
time consuming to test thousands of primitives when 500ms is necessary to
construct the primitive. While we have worked around OpenCL C compilation using
[cl_cache](https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#feature-cl_cache),
there is limited benefit during development as the source code is often
modified. This is enough of an issue that there are plans to modify benchdnn to
improve testing throughput.

The main focus of the RFC is to resolve issues with kernel reuse for dynamically
changing workloads. The effect on first compilations will be considered though,
as solutions for dynamically changing workload can reduce the number of first
compilations required.

## 2. Methods to Resolve this issue

At its core, there are two possible methods:
1. Improve compilation time
2. Reduce the number of kernels compiled.

In regards to improving compilation time, this is infeasible for our OpenCL C
kernels. As such, one solution is to use a different method of kernel
generation. We have done this for convolution, matrix multiplication and reorder
primitives as the OpenCL C language was hindering optimizations. This resulted
in implementing a custom IR and optimization passes to meet our needs for these
problems. There have been discussions about move element-wise and binary
primitives into this infrastructure as most of the required functionality exists
already due to post-op support.

The process of implementing and maintaining kernels in custom generators is more
expensive than using industry standard languages like OpenCL C. Due to limited
expected performance benefit, we are unlikely to ever port all primitives to
this option. In addition, while we have improved kernel compilation time by an
order of magnitude when compared to OpenCL C, it is unclear if a reasonable
investment in optimization will provide sufficient performance to rely on
code generators alone.

Because of this, the calls to the OpenCL C compiler need to be reduced. To
enable this, we need to create reusable OpenCL kernels. In addition, we need to
enable users to reuse these kernels. The rest of this document will focus on how
this reuse can be enabled. The previous discussion suggests a few desired
requirements on the solution.

1. This is a longterm process as it requires modifying many implementations and,
   as such, needs to be exposed on an as implemented basis.
2. The general methodology applies to all kernel generators.
3. Performance improves for benchdnn testing.

## 3. Method of Kernel Distribution
To begin with, we need to discuss how to distribute these kernels, as it places
significant restrictions on implementations. At its core, there are two options:

### Distribution Option 1: Compile Kernels at Build Time
#### Pros:
Users never need to compile a GPU kernel. Solves first compilation issues.

#### Neutral:
OpenCL C Compiler version is fixed at build time. As such users cannot benefit
or see regressions from installing a more recent compiler.

#### Cons:
Increases library size

Increases compilation time

Unclear how many primitives for which this is feasible.

More challenging to implement GPU kernels. This requires all kernels have a
finite number of configurations with reasonable binary size. This is technically
impossible for most primitives due to our API (in particular due to post-ops and
data layouts), but we can implement this feature for a subset of kernels in
common use. Because of this, we still need to maintain runtime compilation.

#### Implementation:
Because this method increases library size, compilation time, and still requires
runtime compilation, this feature should be enabled via a compilation switch
`ONEDNN_COMPILE_KERNELS_FOR_ARCH=<gpu_arch list>`. The default value should be
`NONE` as most users will need at most 2 GPU architectures compiled, an
integrated GPU and a discrete GPU. As such, for most users a significant
fraction of the increased library size is not helpful.

### Distribution Option 2: Compile Kernels Just in Time
#### Pros:
Smaller Library size

Enables more opportunity for JIT compiling.

#### Cons:
Customer has to compile the OpenCL C kernel at least once

#### Implementation:
This is the current behavior, so no changes are required.

### Recommendation:
The recommendation is option 2, keep the current distribution behavior. As
discussed in distribution option 1, runtime compilation will still be necessary.
As such, we need improvements to the runtime behavior in addition to
implementing option 1. Creating reusable kernels is required for improving
runtime behavior or implementing build time compiled kernels. As such we
recommend first improving the runtime behavior, with the addendum that
implementations are modified to support distribution under option 1 when
possible.

## 4. Proposed Kernel Reuse Methodology
The current primitive cache gives no mechanism to reuse kernels. Because of
this, minor changes in workload requires regenerating kernels, and as such
without any changes to the primitive generation infrastructure, there is no
benefit in building reusable kernels.

### Reuse Option 1: Users Rely on cl_cache for OpenCL C Kernels
Allow users to control reusing OpenCL C kernels via the existing cl_cache. This
recommendation has already been made to many customers.

#### Pros:
Allows user control.

Only requires modifying OpenCL C kernels.

#### Cons:
Poor default behavior. The use of cl_cache requires intervention from our users
to setup and manage the generated cache.

No support for non-OpenCL C code generation.

#### Implementation:
None

### Reuse Option 2: Implement Runtime Parameters for primitives
Create primitives where inputs are specified as runtime parameters. Such
functionality already exists for GPU GEMM.

#### Pros:
Allows user control.

Method already exists within oneDNN so there is already precedent.

Feature has been requested by frameworks

Implementable for all kernel generation methods

#### Cons:
Makes API more complicated

Requires user source code changes to benefit

Increases testing load as we need to test combinations of runtime parameters.

Does not improve benchdnn testing performance

Unless care is taken, this will lead to more implementations, increasing
implementation and maintenance burden

Opaque user experience. The user needs to choose between using runtime
parameters to reuse a kernel but with the trade off that they may get worse
performance. Since performance is implementation dependent, users are in a bad
position to when they can make this choice with limited performance degradation.

More work as API support should be implemented for both CPU and GPU.

#### Implementation:
The implementation will be highly dependent on primitive implementations. There
are two main scenarios here. First scenario, we modify our implementations to
generate a finite number of kernels. We then use runtime parameters to
dynamically switch top the best implementation. In this case, since we are
reusing the same finite number of kernels, maintainability is not impacted. The
other scenario requires a separate implementation which increases testing
requirements and maintenance burden.

### Reuse Option 3: Fixed Size Runtime Function Generator Registries
This option creates a fixed size registry for storing reusable kernel
generators. These generators are expected to be performant. For OpenCL C
kernels, these are expected to just be program binaries.

#### Pros:
Requires no user intervention.

No API changes.

Relatively simple implementation

Requirements on Function Generators is similar to Distribution Option 1.

#### Cons:
Requires a finite number of generators, no guarantee generators align with
customer workloads.

Users do not control the registry size

Requires the generator to be GPU agnostic (i.e. should work for 2 or more GPUs
in a system if they have the same architecture) to limit the registry size.

Implementation needs to limit maximum memory usage to a reasonable level.

#### Implementation
This feature will be completely managed by oneDNN developers. As such, there are
no API changes. The only guarantee is that the registry is generated at runtime
and will not use much memory until users creates primitives. It is expected the
implementation will be a hashmap between a kernel configuration and a kernel
generator. Registries will have a compile time defined maximum size to avoid
excessive memory usage. As a registries size is dependent on implementation,
separate registries will exist per implementation. As the requirements on this
implementations are similar to Distribution Option 1,this can be used to enable
build time compilation.

### Reuse Option 4: Global Kernel Generator Cache
This options is a version of the global primitive cache, but for gpu kernel
generators. Kernel generators are expected to be performant. For OpenCL C
kernels, the generator is expected to be a program binary.

#### Pros:
Requires no user intervention.

Implementation for global primitive cache is similar and can be reused.

Users can control memory usage

Generators can be specialized for each GPU in a system, potentially enabling
faster performance.

#### Cons:
Most complicated implementation (except for Option 2) among all the
options.

#### Implementation
Internally, the implementation is expected to reuse the current primitive_cache
implementation. Because of this, much of the cache related complexity is already
implemented. To enable user control of cache memory usage, the kernel cache will
need integration with out external API. There are a few options here:

##### Option A: Reuse Primitive Cache Capacity for Kernel Cache
This options makes no effective changes to the external API. To some extent, the
primitive cache API is for enabling users to make a memory/compute tradeoff in
primitive creation. The only real expectation is that an increase in capacity
results in a roughly proportional increase in memory usage. A kernel cache
targets the same problem and the capacity/memory proportionality is maintained
(or possibly exceeded as a kernel cache can be used to deduplicate primitive
kernels). As such, adding a new API introduces complexity with minimal benefit.

##### Option B: Duplicate Primitive Cache API for Kernel Cache
The API change will be the same as in the [global primitive
cache](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20200207-global-primitive-cache),
except the "primitive" name will be replaced with "kernel". Kernels which use
this cache will need to opt in to its uses. Kernels need to be designed for
reuse for the cache to be effective.

### Recommendation
Short answer: Implement reuse option 4a.

Reuse option 1, rely on cl cache, is not recommended as it has significant
implementation restrictions. In particular this option is only effective for
OpenCL C kernels and it requires active intervention by users, creating a poor
default experience.

Reuse option 2, implement runtime parameters is not recommended due to
implementation complexity and users experience. In particular, this design
allows for (and to some extent requires) multiple implementations, a jit
compiled version and runtime dimension versions. The tradeoff between the jit
version and the runtime versions is expected to be complex and implementation
specific. As such, this option requires knowledge of the oneDNN internals to
use effectively.

Reuse option 3, create a fixed size function registry, is not recommended due to
limitations with post-ops usage and since it is infeasible to generate few
enough kernels covering all uses. In addition, reuse Option 4 provides largely
the same benefit so long as the kernel cache capacity is not exceeded.

Reuse Option 4a is recommended. There are no major limitations enforced by this
design. In addition, the lack of API change makes this easy to use.

Reuse Option 4b is not recommended. While there are major limitations to this
method, the modification to the API increases user complexity with little
expected benefit.

As a data point, the kernel cache from Option 4 similar to how the GEMM
implementation works in oneMKL today. There are a finite number of curated GEMM
implementations which support runtime parameters, but too many for a function
registry. These kernels are stored in a kernel cache to avoid recompilation.
