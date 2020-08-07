# Changing `stream::flags::default_order` behavior

> This is a shameless copy of RFC made by Eugene (@echeresh) to keep all oneDNN
> v2 related suggestions in one place.

## Background

Currently stream flags are defined as a bitmask:

```c++
enum class flags : unsigned {
    /// Default order execution. Either in-order or out-of-order depending
    /// on the engine runtime.
    default_order = dnnl_stream_default_order,
    /// In-order execution.
    in_order = dnnl_stream_default_order,
    /// Out-of-order execution.
    out_of_order = dnnl_stream_out_of_order,
    /// Default stream configuration.
    default_flags = dnnl_stream_default_flags,
};
```

The default order was introduced to support DPC++/SYCL. SYCL queues always
behave as out-of-order (SYCL 1.2.1 standard), and the SYCL runtime handles
dependencies automatically via accessors. `stream::flags::default_order` was
introduced specifically to match this behavior. `default_order` is in-order for
OpenCL, and out-of-order for DPC++/SYCL.

This helped to have API consistency and (kind of) consistency in behavior
across different runtimes. The consistency in behavior is that the
default-constructed stream does not require dependency handling on the user
side:

```c++
stream s(eng, stream::flags::default_order);
```

## Problem

The default order definition is vague, and its behavior is, in fact,
inconsistent across runtimes.

## Proposal

This RFC suggests the following changes for 2.0 version:

1. Make `flags::default_flags` equal to `flags::in_order` (in-order semantics by default)
2. Remove `flags::default_order`

This would make the behavior fully consistent across runtimes for the stream
constructed by default.

### Native CPU and OpenCL

Native CPU streams (OpenMP, TBB, sequential, threadpool) are always in-order.
OpenCL GPU streams are in-order by default (aligned with the current oneDNN behavior).

### DPC++/SYCL

oneDNN provides support for oneAPI DPC++ compiler (official, referred here as
"DPC++") and for ComputeCpp (unofficial, being used by frameworks while
they have not moved to DPC++ yet, referred here as "SYCL"). oneAPI
DPC++ compiler relies on DPC++ with its extensions, ComputeCpp relies
on SYCL 1.2.1 standard.

Recently DPC++ introduced an extension for in-order behavior for queues:
[OrderedQueue](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/OrderedQueue/OrderedQueue_v2.adoc).
This extension (queue property) can be used with `flags::in_order`.

SYCL 1.2.1 does not support in-order behavior but it can be emulated using
explicit `queue::wait()`. This is the current behavior of oneDNN streams with
`flags::in_order` for DPC++/SYCL runtimes - in-order behavior is emulated via
explicit synchronization.

## Next steps

Apply the following changes in `dev-v2` branch:

- Update API:
    - Remove `flags::default_order`
    - Make `flags::default_flags` equal to `flags::in_order`
    - Apply similar changes for C and C++ API
- Update the implementation:
    - Use the ordered queue for DPC++ for `flags::in_order`
    - Emulate in-order behavior for ComputeCpp for `flags::in_order`

## Further discussion and comments

The above proposal breaks backward compatibility in oneDNN 2.0. Though there is
another option which does not break backward compatibility:

- Keep `flags::default_order` and `flags::default_flags` but make it clear in
  the documentation that the default order is the same as `flags::in_order`
- Update implementation to treat `flags::default_order` and `flags::in_order` the same way

This keeps binary/API compatibility but slightly changes the default behavior.

One more thing to mention is that C++ API have a bug in the definition of
`flags::in_order`:

```c++
enum class flags : unsigned {
    ...
    /// In-order execution.
    in_order = dnnl_stream_default_order,
    ...
};
```

`flags::in_order` is defined as `dnnl_stream_default_order`. This means that
now there is no way to create a truly in-order stream with C++ API (e.g. this
might be useful with DPC++ USM). However this can be easily fixed as it
requires changes in `dnnl.hpp` header only.
