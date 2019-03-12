# [RFC] Intel(R) MKL-DNN API 1.0


## CHANGES

- 2018-12-19 Initial version
- 2018-03-08 [v1.0 Preview Candidate](https://github.com/intel/mkl-dnn/releases/tag/v1.0-pc) is published
- 2019-03-12 Remove s16 data type, rounding mode;
             Implicit and explicit scratchpad modes;
             Dropping memory primitive descriptor;
             Minor clean ups;


## Introduction

The Intel(R) MKL-DNN team is planning to release **v1.0** of the library in
mid-2019. With this major version change, we would like to clean up the API to
make Intel MKL-DNN easier to use. For the future releases of v1.x series, we
will preserve API and ABI backward compatibility according to
[Semantic Versioning 2.0.0](https://semver.org/).

This document describes user-visible and some important internal changes to
help developers understand what to expect in the future. Appendix A shows
the new C API types and functions (the C++ API is derived from the C API and
hence omitted for brevity).

The Intel MKL-DNN team would appreciate your feedback and suggestions on the
given proposal! Feel free to post them in the PR created for this purpose.

## Summary of changes

We tried to keep changes minimal to make migration as simple as possible. In
particular, the Intel MKL-DNN programming model would stay the same.
Nevertheless, the new version would bring a lot of incompatible changes
requiring developers to revisit significant portions of the integrated code.

All changes can be split into the following groups:
1. Removing already deprecated functionality
2. Improving the library robustness
3. Simplified execution model
4. Changes in the view primitive
5. Changes in memory description

The first four groups of changes are discussed in detail in this document. The
changes in the memory descriptor deserve a separate topic and are covered here
only from a high-level perspective with detailed explanation available in the
[RFC for memory descriptor](rfc_memory_desc.md).


## 1. Remove deprecated functionality

All the functionality that was marked as deprecated would be removed in v1.0.
That actually already started happening. For instance, see
[this](https://github.com/intel/mkl-dnn/commit/cb04a097c1e692f954702d02211defe986a3b65a)
and
[this](https://github.com/intel/mkl-dnn/commit/b7ab4a7b5985f98eb593cde9bdb7cee78f86bfa2)
commits.

| Deprecated functionality               | Replacement
|:---                                    |:---
| ReLU primitive                         | Eltwise with algorithm kind ReLU
| ConvolutionReLU (single primitive)     | Convolution with ReLU as a post operation
| Double precision scales                | Single precision scales
| RNN backward pd w/o forward pd hint    | RNN backward pd w/ forward pd hint
| `mkldnn_omit_stats` batch norm. flag   | `mkldnn_use_global_stats`
| `mkldnn_eltwise_desc_t.negative_slope` | `mkldnn_eltwise_desc_t.alpha`

The complete list of the deprecated C functions:
``` c++
    mkldnn_relu_forward_desc_init(...);
    mkldnn_relu_backward_desc_init(...);
    mkldnn_convolution_relu_desc_init(...);
```

The complete list of the deprecated C++ classes and functions:
``` c++
    struct mkldnn::convolution_relu_forward {}
    struct mkldnn::relu_forward {}
    struct mkldnn::relu_backward {}

    mkldnn::sum::primitive_desc(const memory::desc &output, std::vector<double> scale, std::vector<memory::primitive_desc> inputs);
    mkldnn::sum::primitive_desc(std::vector<double> scale, std::vector<memory::primitive_desc> inputs);
    mkldnn::eltwise_forward::desc(prop_kind aprop_kind, const memory::desc &src_desc, T negative_slope);
    mkldnn::eltwise_backward::desc(const memory::desc &diff_data_desc, const memory::desc &data_desc, T negative_slope);
    rnn_backward::primitive_desc(const desc &desc, const engine &e);
```

### 1.1. Rename `foo_v2()` to `foo()` and remove old `foo()`

The functions like:
``` c++
mkldnn_primitive_desc_create_v2(...);
```
would be renamed to:
``` c++
mkldnn_primitive_desc_create(...);
```

The old functions (`mkldnn_primitive_desc_create` in this example) would be
removed.

The current list of functions that have `_v2` suffix:

``` c++
mkldnn_primitive_desc_iterator_create_v2(...);
mkldnn_primitive_desc_create_v2(...);
mkldnn_reorder_primitive_desc_create_v2(...);
```

### 1.2. Remove int16 (s16) support

The experimental `s16` data type is not supported any more.

### 1.3. Disallow setting the rounding mode

Rounding mode that was a part of attributes is dropped. All the computations
would respect MXCSR register when performing rounding. Unless set explicitly,
the rounding mode is Round Nearest Even.

## 2. Improve the robustness

### 2.1. The C++ API should take objects by reference whenever possible

Currently there are several functions in the C++ API that take objects by value,
which results in unnecessary copies. So for example `std::vector inputs` would
be replaced with `std::vector &inputs`.

### 2.2. Memory allocation in the C API

In Intel MKL-DNN **v1.0**, constructing a memory object using the C API with no
user-provided memory buffer would result in the buffer being allocated by the
library. This would make the behavior of the C API memory constructor consistent
with its C++ API `mkldnn::memory` counterpart, which allocates memory if called
without a pointer to an existing memory buffer.

### 2.3. Towards stateless primitives: explicit scratchpad management

Currently Intel MKL-DNN primitives may allocate temporary **scratchpad**
memory for storing intermediate computational results. For instance,
convolution backward by weights typically requires extra space to perform a
reduction of the `diff_weights` computed by different threads (the work is
divided across images). Starting with **v1.0**, the library supports two modes:
1. Implicit scratchpad, managed by the library (default);
2. Explicit scratchpad, provided by a user.

The former mode matches the behavior of Intel MKL-DNN v0.x. It is kept for
user convenience and cases where memory is not a concern.

In the explicit scratchpad mode, a new `mkldnn_query_scratchpad_md` query will
return the amount of scratchpad memory needed for a primitive, and the user
will be responsible for allocating and providing the scratchpad memory to a
primitive at a runtime. The explicit scratchpad mode should *explicitly* be
enabled by passing an attribute with `mkldnn_scratchpad_mode_user` to
primitives.

> **WARNING** Scratchpad memory is not the same as workspace.
>
> Workspace is a memory that is passed from a forward pass to the
> corresponding backward pass, e.g. from max pooling forward to max pooling
> backward. Workspace keeps information required to compute results of the
> backward pass. For max pooling, the workspace contains the indices where the
> maximum was found. Workspace must be preserved between the forward and
> backward call.
>
> In contrast, scratchpad memory is temporary: it is used only during a
> primitive's computations (e.g. to perform a reduction across multiple
> threads) and does not hold any useful data after the computations are done.
> The only requirement for the scratchpad memory is that it should be
> sufficiently large to store required data. The same scratchpad memory can be
> reused in the next primitive provided the size requirements are met. That is
> exactly what happens in most of the frameworks that manage all the memory on
> their own.

Explicit scratchpad should make it possible to make Intel MKL-DNN primitives
stateless and hence thread safe: the same primitive can be executed in multiple
independent threads as long as different threads use different scratchpads.

However, if a user chooses implicit scratchpad mode, there is no thread-safety
guarantees.


## 3. Simplified execution model

This is the most notable change in the library. The main idea is to change the
execution API so that memory arguments are specified at a primitive
execution time and not at the primitive creation time. This leads to the
following changes.


### 3.1. Memory is not a primitive anymore

In the current API, memory has a type of primitive. With the new API, a memory
would become a distinct data type. Moreover, memory primitive descriptor would
become redundant and is dropped. The functions that use memory primitive
descriptors now take memory descriptor and (optionally) engine, if the latter
cannot be inferred.

These changes would bring new data types and functions, such as:

``` c++
#define MKLDNN_NATIVE_HANDLE_ALLOCATE  ((void *)-1)
#define MKLDNN_NATIVE_HANDLE_NONE      ((void *)0)

struct mkldnn_memory_t; // memory type, no more equal to mkldnn_primitive_t

// create a memory
// native_handle can:
//  - point to the user allocated memory, i.e. valid handle. In this case the
//    library doesn't own allocated memory.
//  - be MKLDNN_NATIVE_HANDLE_ALLOCATE to ask the library to allocate and
//    attach memory. In this case the library owns allocated memory.
//  - be MKLDNN_NATIVE_HANDLE_NONE to create mkldnn_memory w/o attached memory.
mkldnn_status_t mkldnn_memory_create(mkldnn_memory_t *mem,
    const mkldnn_memory_desc_t *md, mkldnn_engine_t engine,
    void *native_handle);
```


### 3.2. Operation primitives cannot be used as inputs (use memory instead)

The current API allows passing an operation primitive as an input to another
primitive. For instance, a convolution primitive can be passed as an input to
a consequent ReLU. During the execution the ReLU primitive queries the
convolution for its output memory and uses it as an input.

With the new API, users will be allowed to pass only memory type as inputs and
outputs for primitives.


### 3.3. Remove the `mkldnn_primitive_at_t` type

Another consequence is that `mkldnn_primitive_at_t`, which is logically
equivalent to `{primitive, output_index}`, becomes redundant. Previously the
type was used to specify the exact memory to use (if a primitive has several
outputs).


### 3.4. Passing stream and input/output memories at primitive execution

Finally, users would be able to directly run primitives by calling an `execute`
function instead of putting primitives into a stream and running the latter.
This change affects how primitives interact with streams and input/output
memories: with the new API they become arguments to be passed to the primitive
execution function.

The change would significantly simplify primitive creation, which would now
require a primitive descriptor only:

```c++
mkldnn_status_t mkldnn_primitive_create(mkldnn_primitive_t *primitive,
    const_mkldnn_primitive_desc_t *pd);
```

To remove the ambiguity in which order input and output memories need to be
passed, we introduce a map-like argument where each memory argument is paired
with a tag indicating what kind of argument it is: destination, source,
weights, and so on.

``` c++
// types
#define MKLDNN_ARG_SRC_0 1
#define MKLDNN_ARG_SRC   MKLDNN_ARG_SRC_0
#define MKLDNN_ARG_FROM  MKLDNN_ARG_SRC_0
// ...

// C API
typedef struct {
    int arg; // MKLDNN_ARG_SRC, ...
    mkldnn_memory_t memory;
} mkldnn_exec_arg_t;

mkldnn_status_t mkldnn_primitive_execute(mkldnn_primitive_t prim,
    mkldnn_stream_t stream, int nargs, const mkldnn_exec_arg_t *args);

// C++ API
convolution_forward::execute(mkldnn::stream &stream,
    const std::map<int, mkldnn::memory> &exec_args);
// ... other primitives ...


// example C, convolution forward w/ bias
mkldnn_exec_arg_t conv_exec_args = {
    {MKLDNN_ARG_SRC, src_mem},
    {MKLDNN_ARG_WEIGHTS, weights_mem},
    {MKLDNN_ARG_BIAS, bias_mem},
    {MKLDNN_ARG_DST, dst_mem},
};
mkldnn_primitive_execute(conv_fwd, stream, 4, conv_exec_args);


// example C++, in-place eltwise
eltwise.execute(stream, {{MKLDNN_ARG_SRC, mem}, {MKLDNN_ARG_DST, mem}});
```


### 3.5 Short summary

The example below shows conceptual code transformations between versions. The
C++ API is used for brevity.

Version 0.x:
``` c++
// create a convolution, specify all inputs and outputs
auto conv = convolution(conv_pd,
            {src_mem, 0}, {wei_mem, 0}, dst_conv_mem);

// create a relu (note that one of inputs is the convolution)
auto relu = relu(relu_pd,
            {conv, 0}, dst_relu_mem);

// create a stream, submit convolution and relu, and wait for the result
stream().submit({conv, relu}).wait();
```

Version 1.x:
``` c++
// create convolution and relu. no inputs/outputs
auto conv = convolution(conv_pd);
auto relu = relu(relu_pd);

// create stream (based on engine)
stream s(engine, 0);

// execute the convolution with given inputs, outputs
conv.execute(s, {
        {MKLDNN_ARG_SRC, src_mem},
        {MKLDNN_ARG_WEIGHTS, wei_mem},
        {MKLDNN_ARG_DST, dst_conv_mem}});

// execute the relu. cannot pass convolution as an input, only memory is allowed
relu.execute(s, {
        {{MKLDNN_ARG_SRC, dst_conv_mem},
        {MKLDNN_ARG_DST, dst_relu_mem}});

s.wait(); // wait for async streams
```


## 4. View rework

The current library API has a notion of view that, implementation-wise, is
simply an alias for memory. This approach has a significant drawback: it is
not possible to create a view for some combinations of memory formats, window
sizes, and offsets. There is no way around this limitation other than
reordering the original memory into a plain format (like `nchw`) and creating
a view for the resulting memory. Obviously, this requires more memory and
leads to suboptimal performance. Moreover, semantically it is not obvious that
view might fail on creation even if all input parameters seem valid.

Intel MKL-DNN **v1.0** will remove view and replace it with a set of
auxiliary functions. The first one, given a memory descriptor, offsets, and
window sizes, will create a memory descriptor that describes a part of the
original memory. This function will not be guaranteed to always succeed.
Semantics of this function will be very similar to the semantics of the
currently available view primitive, but there will be no extra entities because
it will operate on memory descriptors.

In order to support cases when this function fails, two additional primitives
will be provided to `extract` and `insert` data from/to a part of a memory.
(Alternative names for `insert` and `extract` are `copyin` and `copyout`,
respectively.) Semantically they are similar to reorders operating on parts
of memory objects.

For performance purposes, the preference should be given to creating a
sub-memory.

Several usage scenarios are shown below.

###### Good scenario

*For an experienced user who wants to do an in-place operation on a part of a
memory:*

```
    a. User attempts to create a memory descriptor `md1`, based on the
       original memory descriptor `md0`, offset, and size. This succeeds.
    b. User passes `md1` to a primitive constructor. This guaranteed to
       succeed, but may still fall back to a reference implementation (but
       less frequently than in **Option 2**). User has to check...
```

###### Bad scenario

*For an experienced user who wants to do a in-place operation on a part of a
memory:*

```
    a. User attempts to create a memory descriptor `md1`, based on the
       original memory descriptor `md0`, offset, and size. This fails.
    b. User has to call an `extract` with the same `md0`, offset, and size to
       a separate memory with memory descriptor `md1`.
    c. User creates a primitive based on `md1`.
```

###### General scenario

*Simple approach that always works:*

```
    a. User creates `extract` primitive with the input memory descriptor
    `md0`, offset, and size, and output memory descriptor `md1`.
    b. User creates a primitive using `md1`.
```


### 4.1 Sub-memory related API

Both `extract` and `insert` are suggested to be regular primitives:

``` c++
// Inits memory descriptor for a submemory for given parent memory descriptor,
// size, and offset.
// This function may fail if submemory cannot be represented. In this case
// use `extract` or `insert` primitives to operate with submemory.
mkldnn_status_t mkldnn_memory_desc_init_submemory(
        mkldnn_memory_desc_t *md, const mkldnn_memory_desc_t *parent_md,
        const mkldnn_dims_t dims, const mkldnn_dims_t offset);

typedef struct {
    mkldnn_primitive_kind_t primitive_kind; // must be mkldnn_extract
    mkldnn_prop_kind_t prop_kind;           // must be forward
    mkldnn_memory_desc_t src_data_desc;     // parent descriptor
    mkldnn_memory_desc_t dst_data_desc;     // dst_data_desc.fmt might be `any`
    mkldnn_dims_t offset;
} mkldnn_extract_desc_t;

// Inits extract descriptor.
// Memory format of `dst_d` might be `any`.
mkldnn_status_t mkldnn_extract_desc_init(mkldnn_extract_desc_t *ed,
        const mkldnn_memory_desc_t *src_d, const mkldnn_memory_desc_t *dst_d,
        const mkldnn_dims_t offset);
```

Implementation for `insert` and / or `extract` might be postponed until a
later release.


## 5. Memory descriptor rework

The current way of describing memory format has multiple issues. From the
users' perspective, the main issues are:
- Some memory formats are missing. For example, the `iohw` format is not
  available.
- There are multiple ambiguous ways to describe memory. For example, `oihw`
  describes memory in the same way as `nchw`, but these formats are different
  (see gh#153).
- Support for custom formats is limited.
- Support for memory views is limited.

There are more substantial issues from the library development perspective:
code bloat to support special cases, etc.

We address the issues above by reworking memory descriptors. From the user
perspective, the main changes are:
1. Memory descriptors will support arbitrary strides for plain layouts. For
   example, initializing a memory descriptor with `strides={h*w, o*h*w, w, 1}`
   should be a valid way to define `iohw` format even if Intel MKL-DNN does
   not support it explicitly.
2. Dimensions will be of type `int64_t` instead of int and the maximum number
   of tensor dimensions will be decreased from 16 to 12.
3. The `memory_desc_t.format` field will be replaced with
   `memory_desc_t.format_kind`, which will also have different semantics.

While the first two items are self-explanatory, the last one requires some
additional elaboration.

Currently, most memory formats can be described directly by using appropriate
format names (for example, `nchw`) that fully describe how data is laid out in
memory. However, Intel MKL-DNN also has the `blocked` memory format and the
corresponding `memory_desc_t.layout_desc.blocking_desc` structure, which can
describe a memory format in a unified fashion by specifying block sizes and
strides. The original idea was to used format tags like `nchw` during memory
descriptor initialization only, and always use the `blocked` format internally.
Unfortunately, that was never implemented.

With the new design, Intel MKL-DNN will start distinguishing between the actual
memory format and convenience memory format tags that can be used to describe
memory format concisely.

Users will still be able to initialize memory descriptors with format tags
like `nchw`, but the `memory_desc_t.format_kind` will be set to a canonicalized
kind like `blocked`, and the format name will not be recorded in the memory
descriptor structure. Initialization with strides will always result in
`blocked` format. The API will also use different types for memory format
tags and kinds to aid correctness.

This change affects all existing code. For more details, refer to the
[RFC for memory descriptor](rfc_memory_desc.md) document.


## Appendix A. Changes in C API

See [c_api.h](c_api.h).


###### End of document
