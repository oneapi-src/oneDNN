# Proposal for introducing concurrent and global scratchpad mode

## Background

Currently, some primitive implementations in oneDNN require a temporary storage
for computations, a.k.a. scratchpad, and the library supports two scratchpad
policies: global and concurrent.

In most cases, the scratchpad has a concurrent mode, whereby each thread
allocates a new scratchpad storage during the instantiation of a primitive that
has been defined without `USE_GLOBAL_SCRATCHPAD` flag in the library. The
following is an example of an implementation definition with concurrent
scratchpad:

~~~cpp

DECLARE_COMMON_PD_T("ncsp_bnorm:any", ncsp_batch_normalization_fwd_t);

~~~

In case a primitive implementation does contain the `USE_GLOBAL_SCRATCHPAD`
flag, it can create (and expand, if necessary) a global scratchpad that is
shared among all primitive implementations defined with this flag. Such
scratchpad is defined as `static`, and it is destroyed once the last primitive
in user's application is also destroyed. Since the global scratchpad is defined
with a `thread_local` qualifier, the user has to ensure that all primitives that
use such storage are created **and** executed on the same thread. The default
library build configuration enables global scratchpad support, and it can be
disabled using the `DNNL_ENABLE_CONCURRENT_EXEC` macro. The following is an
example of an implementation definition with global scratchpad support:

~~~cpp
DECLARE_COMMON_PD_T(
                GEMM_IMPL_STR, gemm_convolution_fwd_t, USE_GLOBAL_SCRATCHPAD);
~~~

Both types of scratchpad are internal to the library, and currently there are
two ways a user can have a control over the scratchpad:

1. By providing primitive attributes with a `scratchpad_mode::user` flag.

2. By querying the scratchpad size the primitive implementation requires using
   `query::memory_consumption_s64`.

In case of 1, the library does not allocate an internal storage and relies on
the user to provide one during execution using `DNNL_ARG_SCRATCHPAD`. Although
this ensures thread safety by providing independent (i.e. concurrent) storage
for every primitive, this can dramatically increase memory consumption by all
scratchpad storages allocated by the user.

In case of 2, the library will either allocate an internal buffer of such size,
or expects the user to provide one during execution. In case the library
allocates an internal scratchpad, the user has no knowledge whether it is shared
among all or some primitives or not. Therefore, the query cannot guarantee the
total memory consumption if executing more than one primitive.


## Objective

The aim of this proposal is to more provide more explicit controls to the user
over primitive scratchpad. The main goal is to reduce the total memory usage due
to scratchpad and/or guarrantee thread safety by using (i) new scratchpad modes
and (ii) new compilation flag.

The functionality described below will be supported by CPU only. Currently there
is no support for global scratchpad on the GPU, and this proposal does aim to
not change this behavior.

## Naming adjustments

Given that the scope of this work consists in enabling the user to manage the
scratchpad allocation inside the library, it is proposed to adjust the adjust
the existing types of scratchpad storage inside the library as following:
1. `concurrent_scratchpad_t` --> `primitive_private_scratchpad_t`, thereby
   emphasizing that such scratchpad storage is tied to a particular primitive
   and enables thread safety.
2. `global_scratchpad_t` --> `thread_private_scratchpad_t`, thereby emphasizing
   that such scratchpad storage is tied to a particular thread of execution, and
   can be shared by any primitives executed by this thread, which enables
   reduced memory footprint.


These types, as well as the subsequently proposed additions to the library, are
expected to be more self explanatory for the user.

## API

The scope of this proposal includes introducing two new `scratchpad_mode`
primitive attributes:

1. `primitive_private`. In this case, the user enforces all primitives to use an
   independent scratchpad. The user may use this flag for thread safety, and it
   enables the user to estimate the total scratchpad memory consumption,
   although can increase memory consumption dramatically;

2. `thread_private`. In this case, the user enforces all primitives to use a
   shared global scratchpad. Thread safety is not guaranteed here, however the
   user may work around this if necessary. This flag can significantly reduce
   the memory consumption due to scratchpad, and the user can query the total
   memory usage due to scratchpad from the last primitive.

These flags will work on a primitive by primitive basis, meaning that the
primitives will use different storage if different flags were used for their
instantiation. This enables a flexible decision making process for the user with
respect to prioritizing thread safety and total memory allocation, instead of
being enforced to the current scratchpad implementation in oneDNN.

It is important to note that, by using the above flags, the user enforces the
corresponding behavior discussed above, rather than serving as a hint.

The following values for `dnnl_scratchpad_mode_t` will be introduced:

```cpp
/// The user enforces the primitive-private scratchpad mode for the primitive.
dnnl_scratchpad_primitive_private,
/// The user enforces the thread-private scratchpad mode for the primitive.
dnnl_scratchpad_thread_private,
```

Similarly, the following `scratchpad_mode` enum values will be introduced:

```cpp
/// Scratchpad mode
///
primitive_private = dnnl_scratchpad_primitive_private,
///
thread_private = dnnl_scratchpad_thread_private,
```

### Usage models

Since the new scratchpad modes are intended to complement existing one instead
of replacing them, new usage models are assumed to be favorable for the user in
a particular instance. These usage models are briefly presented in the table
below:

| Scratchpad mode       | Use case | Advantages |
|-----------------------|----------|------------|
| **library**           | The library allocates scratchpad storage.<br>The mode is chosen by primitive implementation.                    | No API usage<br>     |
| **primitive_private** | The library allocates scratchpad storage.<br>New scratchpad storage is created for each primitive.              | Thread-safety        |
| **thread_private**    | The library allocates scratchpad storage.<br>The storage is reused by primitives.                               | Reduced memory usage |
| **user**              | The user will allocate scratchpad at primitive execution.<br>The scratchpad can be reused at user's discretion. | Full user control    |


## Compilation flag

This proposal also aims to introduce a compilation flag
`DNNL_LIBRARY_SCRATCHPAD_MODE`, which enables tweaking the scratchpad mode at
compile-time for all primitives. It is proposed that this compilation flag will
have the following two options:

1. `DNNL_LIBRARY_SCRATCHPAD_MODE = PRIMITIVE_PRIVATE`. If specified, the default
   scratchpad mode, `scratchpad_mode::library`, is equal to
   `scratchpad_mode::primitive_private`, thereby enabling a concurrent
   scratchpad for all primitives.
2. `DNNL_LIBRARY_SCRATCHPAD_MODE = THREAD_PRIVATE`. If specified, the default
   scratchpad mode, `scratchpad_mode::library`, is equal to
   `scratchpad_mode::thread_private`, thereby enabling a global scratchpad for
   all primitives.

If `DNNL_LIBRARY_SCRATCHPAD_MODE` is not specified, the scratchpad mode for a
particular primitive will be determined by its implementation, as showed by the
examples above.

The above flags are expected to be beneficial in case a user does not require a
per-primitive attribute and can rely on a global setting for scratchpad.
Therefore, the users who do not use primitive attributes will benefit the most
from the above compilation flags. The `scratchpad_mode` API will supersede the
above compilation flags, and the functionality of `scratchpad_mode::library`
will be determined by the compilation flag.

Given the above compilation flags, the proposal aims to replace
`DNNL_ENABLE_CONCURRENT_EXEC` flag with `DNNL_LIBRARY_SCRATCHPAD_MODE =
PRIMITIVE_PRIVATE` since they are equivalent.

## Tutorial

Following the discussion above, it is expected that the users will strongly
benefit from this functionality, and the users will be encouraged to use it in
their code if possible. Therefore the scope of this proposal includes
introducing a tutorial show demonstrates the benefits of using a global
scratchpad in a topology, for instance AlexNet[1] or ResNet50.

[1] Alternatively, the `cnn_training_f32.cpp` example can be adjusted to
demonstrate this functionality.