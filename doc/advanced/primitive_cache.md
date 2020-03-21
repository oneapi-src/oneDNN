Primitive Cache {#dev_guide_primitive_cache}
===========================================================

Primitive creation time largely depends on the underlying implementation,
for instance, oneDNN uses just-in-time compilation (JIT) to generate optimal
code for some CPU and GPU implementations, which introduces overhead.

To mitigate primitive creation overhead, oneDNN provides the primitive cache
which automatically caches created primitives to avoid repeating JIT compilation
for the primitives with identical operation descriptors, attributes, underlying
primitive implementations, etc. It can significantly reduce primitive creation
overhead, especially when an application or a framework creates primitives
for every instance of inference or iteration of training process.

The primitive cache is global hence a user doesn't have to maintain any
persistent oneDNN resources to benefit from the primitive cache.

## Memory consumption
Since the primitive cache has limited capacity, it uses
a replacement policy to evict excess primitives. The capacity indicates
the maximum number of primitives it can hold at a time and it can be adjusted
with an API or an environment variable `DNNL_PRIMITIVE_CACHE_CAPACITY`.
If the capacity is set to 0 then the primitve cache is disabled.
The API takes precedence over the environment variable.

## Primitive cache profiling
Information about primitive cache hits and misses can be used for debug
purposes. That information is part of the verbose output for verbose
level 2 (@ref dev_guide_verbose).
