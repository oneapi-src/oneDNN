# Proposal on caching GPU kernels instead of their binary representations

Motivation

The primitive cache is designed to hold the whole primitive which can contain
CPU or GPU kernel. The GPU kernel can have two states:
1. Compiled kernel
2. Binary representation of the kernel

Currently, GPU primitive that is stored in the primitive cache contains GPU kernel
which has the 2nd state.

There are two cons of such approach:
1. Binary representation is basically an array of bytes hence no 3rd party resources
are stored in the primitive cache
2. Binary representation is associated with a particular GPU device but is context
independent therefore it can be reused for different GPU contexts

Current approach allows to avoid storing OpenCL and SYCL resources globally to
avoid a potential issue which may result in a crash due to the order of unloading
oneDNN and the libraries whose resource oneDNN holds globally.

The main requirement of the approach is that only context independent resources
can be stored in the primitive. To create and carry context dependent resources
oneDNN has a resource abstraction which is created at each primitive creation stage
regardless of whether it's a cache hit or cache miss.

This approach has a number of drawbacks which become critical:
1. There is overhead on creating a kernel out of a binary per each primitive creation
    * That time can vary significantly depending on architecture. For example creating
    kernels from binary for batch normalization on Gen9 takes ~200us but on a new
    architecture it may take ~1.5ms. The reason is likely not optimized software
    stack for that architecture but it means that such an issue can occur for each
    new architecture
2. Unfortunately memory is context dependent and hence cannot be stored in the
primitive. The scales must be copied to the memory and the memory then
is stored in the oneDNN resource. The copy also involves map/unmap operations
3. SYCL is trying to abstract away backend specific behavior and if it cannot do that
it gets rid of some backend specific API. For example not all SYCL abstractions allow
to get native backend handle which can be used to work with backend API directly.
In general, this problem is solvable. There is an option to request Intel extensions
which will provide required API

For the models that have small execution time, for example DLRM or BERT the
overhead becomes critical. All optimizations that can be done at oneDNN side is
completed. They allowed to get rid of overhead for models like ResNet-50 (MB 1024),
but they didn't solve the problem.

The only viable option is to get rid of the overhead completely. The proposal is
to switch to caching compiled kernel rather than their binary representations.

## How to deal with the unloading libraries issue?

In general, it's seem unlikely to run into the issue because:
1. According to the SYCL integration design, frameworks store GPU device and context
abstractions globally. If that was an issue they wouldn't have done that
2. Benchdnn and Gtests have global engine which contains GPU device and context
abstractions and no issues so far

If it becomes an issue, the workaround will be clearing the primitive cache by
calling  `dnnl::set_primitive_cache_capacity(0);` before exit on the frameworks' side.

Also, SYCL is trying to handle such situations properly. For example on Linux,
they use `__attribute__((destructor))` with low priority to make sure that the libraries
that do not use the constructor with lower priority will be unloaded before the SYCL
Runtime. [Here](https://github.com/intel/llvm/blob/sycl/sycl/doc/GlobalObjectsInRuntime.md)
is the document that describes what they do to prevent such issues.

## Benefits from caching compiled kernels

There is a number of benefits:
1. Eliminate overhead on creating kernels from binary and copying scales
2. Significantly reduce code complexity
    * No need in having resource for GPU
    * No need in having complex abstractions that can contain either binary or kernel
3. Everything can be stored in the primitive

## DPC++ cache

Not directly related to the proposal but DPC++ is going to implement so-called
["persistent cache"](https://github.com/intel/llvm/blob/sycl/sycl/doc/KernelProgramCache.md)
which is storing some internal resources in a data storage. It means that time
on first primitive creation for DPC++ kernels can be reduced significantly if
there is a corresponding resource in the cache in the data storage.
If oneDNN caches compiled kernels and the first iterations benefits from the
"persistent cache" then time on primitive creation will be reduced as much as it's
technically possible.


## Primitive cache key

The key should be extended to take into account GPU context. Currently, there is
`device_id_t` which is basically an alias for `std::tuple`. The key doesn't store
GPU devices themselves but a few numbers that can be used to distinguish one
GPU device from another.

In order to switch to caching compiled kernels there will be introduced `engine_id_t`
which will be responsible for holding common information e.g. engine kind,
runtime kind and index. And CPU or GPU specific information too, e.g. GPU device
and GPU context for GPU primitives.

