# Reducing Primitive Creation Overhead for Dynamic Shapes

## Motivation

oneDNN API requires users to specify tensors dimensions to create a primitive,
which makes the primitive dependent on the tensors's shapes. Primitive creation is
an expensive operation, this is why oneDNN has a primitive cache to reduce the
overhead. However, the primitive cache can only help if the tensors shapes do not
change between workload iterations otherwise the primitives have to be created for
the new shapes which incurs additional overhead. The goal of this RFC is to propose
a solution to reduce that overhead.

## Background

There is a reason why the oneDNN API requires users to specify tensors dimensions
to create a primitive. The library may use the information to generate a kernel
specifically for the given shapes to get the best performance.

However, the information may not always be used by the library meaning that the
generated kernels can be used for different shapes, which makes requirement to
always specify tensors dimensions is overly strict.

There was an attempt in the past to relax the requirement by introducing so-called
runtime dimensions support. The idea was to allow users specify a placeholder
(`DNNL_RUNTIME_DIM_VAL`) instead of the actual dimensions. The general rule for
the feature was that the more actual dimensions users provide the better
performance they can get.

It's obvious that in order to support runtime dimensions the kernels must not
depend on the shapes. Or there should be different kernels that are specific to
the amount of information provided. And the kernels should also demonstrate
reasonably good performance otherwise, the creation overhead will be replaced by
a suboptimal performance overhead, which will defeat the purpose of having the
runtime dimensions feature.

The runtime dimensions feature is not a silver bullet, this is why the RFC will
consider another option to reduce the primitive creation overhead.

## Proposal

There are two options to reduce primitive creation overhead:
1. Extend the runtime dimensions feature for the relevant primitives
2. Introduce a cache for kernels

### Option 1: Cache for Kernels (recommended)

The idea is to introduce a cache for kernels (hereinafter *cache*). The cache will
be global similar to the primitive cache.

Pros:
* Doesn't require users to modify their code to make the feature work
* Eliminates the kernel creation overhead completely when it's possible
* Doesn't require to expose kernel implementation details. For example, for
the runtime dimensions feature we will have to document what dimensions can
be runtime without introducing performance regressions

Cons:
* Primitive and primitive descriptor creation overhead will stay

#### Key and Value

The cache implementation will reside in the `common` part of the library. In order
to abstract away `key` and `value` entities we will need to introduce the
following abstractions:

```cpp
namespace kernel_cache {
// Key
struct key_t {
    key_t(key_impl_t *impl);
    virtual ~key_t() = default;

    bool operator==(const key_t &other) const;
    size_t hash() const;
private:
    std::shared_ptr<key_impl_t> impl_;
};

// The `key_impl_t` class will be the base class for all `key` entities.
// Each engine kind and vendor will be able to define it the way the need.
struct key_impl_t {
    key_impl_t() = default;
    virtual ~key_impl_t() = default;

    key_impl_t(const key_impl_t &) = delete;
    key_impl_t &operator=(const key_impl_t &) = delete;

    virtual bool compare(const key_impl_t *key_impl) const = 0;
    virtual size_t hash() const = 0;
};

// The `value_t` class will be the base class for all `value` entities.
// Each engine kind and vendor will be able to define it the way the need.
struct value_t {
    value_t(value_impl_t *impl);
    virtual ~value_t() = default;
    std::shared_ptr<value_impl_t> &impl() const;
private:
    std::shared_ptr<value_impl_t> impl_;
};

struct value_impl_t {
    value_impl_t() = default;
    virtual ~value_impl_t() = default;

    value_impl_t(const value_impl_t &) = delete;
    value_impl_t &operator=(const value_impl_t &) = delete;
};
} // kernel_cache
```

#### Example

Below is a pseudo-code that demonstrates the idea of using the aforementioned
abstractions for GPU engine kind.
```cpp
// GPU specific `key` and `value` implementations.
struct gpu_key_t : public key_impl_t {
    // This could be a serialized configuration structure or anything else.
    std::vector <uint8_t> data;

    compare(const key_impl_t *key_impl) const override {
        const auto *typed_key = static_cast<const gpu_key_impl_t *>(key_impl);
        return data == typed_key->data;
    }

    size_t hash() const override {
        // This is just for demonstration purposes. In the implementation
        // we should use `hash_combine` function.
        size_t seed = 9;
        for (uint8_t val : data)
            seed ^= (val << 1);
        return seed; ^
    }
};

struct gpu_value_t : public value_impl_t {
    compute::kernel_t kernel;
};

`struct kernel_cache_t {
    /* Internal interfaces */
    /* ... */
    lru_cache_t<key_t, value_t, value_t> cache;
};

// Returns a reference to a kernel cache initialized with the
// primitive cache capacity.
kernel_cache_t &global_kernel_cache();

// Inside sycl_gpu_engine_t::create_kernel(...)
auto &kcache = kernel_cache::global_kernel_cache();
// Create a key.
kernel_cache::key_t key = ...;
kernel_cache::value_t value = kcache.get(key);

// Assume bool(value) == true.
gpu_value_t *gpu_value = static_cast<gpu_value_t *>(value.impl().get());
compute::kernel_t k = gpu_value->kernel;
// Now we can use the kernel `k`.
```

#### Memory Consumption Management

oneDNN users usually want to have control over amount of memory that the library
consumes if that amount is significant. Introducing the cache will come with
additional memory consumption though if it is reasonably small then users may not
want to deal with it. Given that, the are two approaches to the cache memory
consumption.

##### Option 1: Keep the Cache Hidden (recommended)

This option suggests making the cache a hidden feature without providing any
control over its capacity.

There are two approaches to defining the cache capacity:
1. Define a reasonable constant
2. Tie the cache capacity to the primitive cache capacity

The first approach is overly rigid and it is hard to define a reasonable
capacity that would work for the majority of workloads therefore we will consider
the second one.

Pros:
* Provides sufficient control to benefit from the cache and to maintain memory
consumption
* Doesn't introduce yet another *cache capacity* definition that will simplify using the cache

Cons:
* There is no guarantee that all kernels in the kernel cache are used by all the
primitives in the primitive cache. It might happen those primitives in the
primitive cache has unique kernels and the kernel cache may have unique kernels
too. This means that memory consumption can be higher that it is now

##### Option 2: Provide API to Control Kernel Cache Capacity

This option suggests adding an API to control kernel cache capacity similar to
the one for the primitive cache.

Pros:
* Provides more control over the kernel cache and therefore more flexibility to
the users. Those users who heavily rely on dynamic shapes can balance between
primitive cache capacity and the kernel cache capacity
* User can fully control memory consumption

Cons:
* Introducing yet another *cache capacity* definition that should be specified.
Also, we will have to explain to the users the difference between kernel cache and primitive cache capacities as well as how they interoperate with each other

###### C API

```c
dnnl_status_t dnnl_get_kernel_cache_capacity(int *capacity);
dnnl_status_t dnnl_set_kernel_cache_capacity(int capacity);
```

###### C++ API

```cpp
int get_primitive_cache_capacity();
void set_primitive_cache_capacity(int capacity);
```

###### Environment Variable

`ONEDNN_PRIMITIVE_CACHE_CAPACITY=<number>`, where `number` is the kernel cache
capacity. The default value is 1024. When `number` is 0 the kernel cache is
disabled.

###### Build Time Control

The kernel cache can be enabled or disabled at the build time using a CMake option
`ONEDNN_ENABLE_KERNEL_CACHE`.

##### Options Comparison

Both options assume that there will be a reasonable default capacity for the
kernel cache. Also, both options allow users to control the caches capacity
therefore, the only difference between the options is in the flexibility.
The second option is in fact the first one with advanced controlling
capabilities.

Given that the first option will address the problem with dynamic shapes
related creation overhead the proposal is to start with the it and add the
API from the option two if we see any need in it.

### Option 2: Extending Runtime Dimensions Support

This option suggests extending the existing runtime dimensions support for
relevant primitives.

Pros:
* Using the existing mechanism, the primitive cache will automatically work
for primitives created with runtime dimensions
* Unlike the kernel cache, with this approach the primitive and primitive descriptor
is not re-created therefore there is no related overhead

Cons:
* The users have to modify their code to enable runtime dimensions
* Some primitives will require to generate multiple kernels so that the
primitives can be used with runtime dimensions. It may not be always feasible to
generate a reasonable number of kernels
* For some kernels it may not be possible to stay performant and support runtime
dimensions for all dimensions. This means that if user uses incorrect combination
of runtime dimensions they may get suboptimal performance
    * There is an option to document all combinations that will not introduce
    performance regressions however this would be essentially exposing
    implementation details to the users. This will require us to make sure that
    the documented combinations are valid for all hardware and implementations
    which will have maintenance cost


### Conclusion

The runtime dimensions feature eliminates most creation overhead (primitive descriptor,
primitive, kernel). However, the primitives may have to generate multiple kernels to
support the feature. And if there are too many (or even indefinite amount of) kernels
to generate then the runtime dimensions cannot be supported or can be supported partially
which would mean the supported combinations will have to be documented (i.e., exposing
implementation details).

The kernel cache feature eliminates only kernel creation overhead that is usually the
most significant one. However, it will work for all primitives whenever the kernels can be
re-used. With this feature there is no need to generate all possible kernels, only those
that are used in the workload will be generated and re-used.

As we can see there is not an ideal solution to the original problem. The conclusion is
that the runtime dimensions feature work better for optimizations very specific cases
but provide does the best job at eliminating any creation overhead. While the kernel
cache feature covers as many cases as possible providing the best result overall.

The proposal is to proceed with the kernel cache feature.