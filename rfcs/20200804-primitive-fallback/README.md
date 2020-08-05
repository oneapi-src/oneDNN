# Review of primitive fallback options


NOTE: The scope of this RFC is limited to the **CPU primitives** only, and the
GPU primitives will not be impacted by changes brought by this RFC. For now,
there is no error handling mechanism for JIT kernels on GPU, and we may revisit
a similar issue for GPU primitives in the future.

## Background

Primitives in oneDNN internally have multiple implementations that (i) are
optimized for a particular Instruction Set Architecture (ISA), and (ii) can
handle a particular set of user parameters (e.g. attributes, tensor shapes,
memory layout etc). All (or most) primitives contain implementations that use
Just-In-Time (JIT) code generation in order to further improve performance.
Currently, generation of JIT code (or kernel) is a two-step process:
1. An implementation, that satisfies the above criteria, is being looked for
   during primitive descriptor construction (e.g.
   *convolution_forward::primitive_desc()*);
2. If the implementation above is based on a JIT kernel, the corresponding code
   is being generated during construction of the primitive (e.g.
   *convolution_forward()*)

Currently, if an implementation has been successfully found, it is expected that
it will also successfully generate the kernel and execute the generated code.
However, occasionally, a JIT implementation may throw an exception during
generation [1], which may be a simple logical error or something much less
trivial to fix.

## Proposal

This document presents some options to deal with inability of a primitive to be
constructed having created a valid primitive descriptor implementation.

The options below are listed in the order of increased amount of anticipated
changes to support primitive fallback.

Additional options are welcome to be added to the list.

### Option 1: leave as is

For this option, the user will take advantage of the existing
`primitive_desc::next_impl()` API to be able to deploy a different
implementation if the first one fails. In this case, the user is enforced to
introduce some control flow, something like
~~~cpp

auto pd = convolution_forward::primitive_desc(opd, attr, eng);
convolution_forward prim;

do {
    try {
        prim = convolution_forward(pd);
    }
    catch (dnnl::error &e) {...}
} while (prim == convolution_forward() && pd.next_impl());
~~~

In order to ensure that the issue described above will be taken care of by the
user, the documentation will be updated accordingly.

Pros: 
1. Minimal changes required in documentation;

Cons:
1. The user is responsible for ensuring that a valid primitive implementation
   exists.

### Option 2: allow construction of an empty primitive

For this option, we attempt to simplify the control flow in [Option
1](#option-1-leave-as-is) by allowing construction of an empty primitive without
throwing an exception using the `allow_empty` flag:

~~~cpp
primitive::primitive(const_dnnl_primitive_desc_t c_pd, bool allow_empty = false)
~~~

In this case, the control flow can be simplified to:

~~~cpp

auto pd = convolution_forward::primitive_desc(opd, attr, eng);
convolution_forward prim;
do {
    prim = convolution_forward(pd, true);
} while (prim == convolution_forward() && pd.next_impl());

~~~

Pros:
1. Minimal changes required in documentation;
2. A slightly simpler control flow compared to [Option
   1](#option-1-leave-as-is);

Cons:
1. The user is responsible for ensuring that a valid primitive implementation exists.

### Option 3: fallback to the reference implementation (personal preference)

For this option we introduce `std::unique_ptr<primitive_desc_t> pd_ref_`,  which
stores the reference implementation for a particular primitive. in
`dnnl_primitive_desc_iterator` and in `dnnl_primitive_desc`. In case a JIT
implementation fails during its generation for any reason, the reference
implementation will be expected to handle any (or most) use cases supported by
the library, such as attributes, memory layouts, and data types.

In the context of primitive cache, in order to avoid repeatedly failing to
generate a JIT primitive implementation, we can:
1. Set `bool primitive_t::is_initialized_` (which is inherited from `struct
   c_compatible`) to `false` during construction, and change to `true` if
   `primitive_t::init(engine_t *engine, bool use_global_scratchpad)` returns
   `dnnl_success`;
2. Include `bool primitive_t::is_initialized_` into `primitive_hashing::key_t`
   to compute the primitive hash;
2. If `is_initialized_ == true`, a valid JIT implementation is found in cache
   and it will be deployed for computation;
3. If `is_initialized_ == false`, an invalid JIT implementation is found in
   cache and the reference implementation will be deployed for computation.

Pros:
1. No changes are required from the user.
2. Primitive cache does not need to store the reference implementation since it
   is lightweight;
3. Almost unlikely for the application to crash (requires validation);

Cons:
1. The limitations of the fallback reference implementation have to be
   documented, such as unsupported options (e.g. lack of reference winograd
   convolution);
2. Very poor performance;
3. Primitive descriptor queries have to take into account the fallback implementation;
4. Changes required for scratchpad memory tracking to take into account an extra
   implementation.
5. Primitive cache stores an (invalid) JIT implementation 

### Option 4: fallback to the next successful implementation

For this option, we introduce `std::unique_ptr<dnnl::impl::primitive_desc_t>
pd_fb_`, which stores an additional primitive implementation, in
`dnnl_primitive_desc_iterator` and in `dnnl_primitive_desc`. Once the first
valid primitive implementation is assigned to
`dnnl_primitive_desc_iterator::pd_` when iterating over the implementation list,
the iterating process continues until `pd_fb_` will also be assigned some
implementation.

In the context of the primitive cache, in order to avoid repeatedly failing to
generate a JIT primitive implementation, we include the changes discussed in
[Option
3](#option-3-fallback-to-the-reference-implementation-personal-preference).
However, in this case, primitive cache will store two implementations: one valid
(with `primitive_t::is_initialized_ == true`) and one invalid (with
`primitive_t::is_initialized_ == false`). For the computation, the valid
implementation will be deployed.

Pros: 
1. No changes are required from the user.
2. `pd_fb_` most likely will also be an optimized implementation, therefore we
   may not lose too much in terms of performance;

Cons: 
1. The issues described in the introduction can also occur to `pd_fb_` if it is
   a JIT implementation;
2. Primitive cache will have to store two implementations for a single
   primitive.
3. Primitive descriptor queries have to take into account the fallback implementation;
4. Primitive constructor may take twice as much time to be constructed;
5. Changes required for scratchpad memory tracking to take into account an extra
   implementation.

[1] A status code will be returned by Xbyak instead of an exception in a future
version of the library.

## Update

Given that there was no feedback from Framework enabling teams, the decision was
to pursue Option 1. This decision will be revised once there will be an explicit
request to support fallback on library side or in some other form.