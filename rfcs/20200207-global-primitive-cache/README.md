# Global Primitive Cache in oneDNN

## Introduction
Primitive creation time largely depends on the underlying implementation, for
instance, oneDNN uses just-in-time compilation (JIT) to generate optimal code
for some CPU and GPU implementations, which introduces overhead.
To mitigate primitive creation overhead users have to implement their own
primitive cache (hereinafter cache) which can be suboptimal (e.g. primitives
contain scratchpad memory) and require to maintain persistent oneDNN resources
(e.g. engine).

## Proposal
The proposal is to introduce support for a global cache in the library which
meets the following requirements:
1. Primitive creation overhead is significantly reduced
2. The library provides an API to control the cache
3. There is no a requirement to maintain any persistent oneDNN resources
4. Multithreading support
5. The scratchpad is not stored in the cache
6. The cache should be common for CPU and GPU

## API

The API for the global cache should provide an ability to control cache capacity
as well as cache size.
The capacity is the maximum amount of entries that the cache can hold at the
time, and the size is the number of entries in the cache.

The API is defined as follows:
```cpp
// C
dnnl_status_t dnnl_set_primitive_cache_capacity(int capacity);
dnnl_status_t dnnl_get_primitive_cache_capacity(int *capacity)
// C++
void set_primitive_cache_capacity(int capacity)
int get_primitive_cache_capacity();
```
Also, the library should provide a way to set the cache capacity with an environment
variable. The environment variable is defined as follows:

```sh
set DNNL_PRIMITIVE_CACHE_CAPACITY=<capacity>
```
The API takes precedence over the environment variable.

When a user sets a new capacity and the new capacity is less than the old one
then excess entries will be evicted from the cache following the LRU (Least
Recently Used) eviction policy hence the cache can be cleared as follows:
```c++
set_primitive_cache_capacity(0)
```

## Primitive Cache

The idea is to store the entire `dnnl::impl::primitive_t` (primitive implementation)
in the cache rather than just a CPU or GPU kernel because this approach has the
following advantages:
1. Primitive descriptor is not cloned in case of cache hit which is important
because it takes quite a bit time to copy attributes, information about booked
scratchpad memory, scratchpad memory descriptor, etc., which reduces efficiency
of the cache.
2. Since `dnnl::impl::primitive_t` doesn't know about the implementation it
encapsulates the cache can be implemented at the runtime agnostic level. Therefore
the cache will be common for CPU, GPU and any other runtimes.

### Cache Key
For each `dnnl::impl::primitive_t` there should be associated a unique key that will
help to distinguish one primitive from another. The key class looks as follows:

```cpp
namespace primitive_hashing {
    struct key_t {
        key_t(const primitive_desc_t *pd, const engine_t *engine, int impl_nthr);
        bool operator==(const key_t &rhs) const;
        // To distinguish primitives created for different shapes, attributes and
        // primitive kinds.
        dnnl_primitive_kind_t primitive_kind_;
        // cached_op_desc_t is an implementation of pattern variant to store
        // any operation descriptor.
        const cached_op_desc_t op_desc_;
        const primitive_attr_t attr_;
        // To distinguish primitives created for different implementations
        // (to support primitive descriptor iterator) and number of threads.
        std::type_index impl_id_;
        int impl_nthr_;
        // To distinguish implementations with different diff_(src|dst)_md.
        std::vector<memory_desc_t> mds;
        // To distinguish implementations created for different engines.
        engine_kind_t kind_;
        runtime_kind_t runtime_kind_;
        // In case of OpenCL `device_id_` can be defined as follows:
        // `device_id_ = reinterpret_cast<intptr_t>(device())`
        // device() returns cl_device_id which is an alias for _cl_device_id *
        // For CPU device_id_ is always 0
        intptr_t device_id_;
    };
}
```
The OpenCL specification says that `clGetDeviceIDs` returns a pointer to a list
of unique devices (where device is a pointer), but it doesn't say that the list
of pointers is the same for different `clGetDeviceIDs` calls therefore it's an implementation-dependent part but the current implementation has a global list
of devices hence the list of devices is always the same for different
`clGetDeviceIDs` calls as well as pointers to the devices.

### Hashing and Comparison
One of the main requirements to the `key_t` is that it should be possible to get
its hash value. The idea is to introduce functions that compute hash for all
operation descriptors, attributes and memory descriptors relying on the main hash
function. Other fields can be computed directly with the main hash function which
is defined as follows:

```cpp
template <typename T>
static size_t hash_combine(size_t seed, const T &v) {
    return seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
```
Below is an example of the function that computes hash for convolution operation
descriptor:
```cpp
// (De-)Convolution
size_t get_desc_hash(const convolution_desc_t &desc) {
    size_t seed = 0;
    // Kinds
    seed = hash_combine(seed, static_cast<size_t>(desc.primitive_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.prop_kind));
    seed = hash_combine(seed, static_cast<size_t>(desc.alg_kind));
    // Memory descriptors
    seed = hash_combine(seed, get_md_hash(desc.src_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_src_desc));
    seed = hash_combine(seed, get_md_hash(desc.weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_weights_desc));
    seed = hash_combine(seed, get_md_hash(desc.bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_bias_desc));
    seed = hash_combine(seed, get_md_hash(desc.dst_desc));
    seed = hash_combine(seed, get_md_hash(desc.diff_dst_desc));
    // Strides, dilates, padding
    seed = get_array_hash(seed, desc.strides, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.dilates, DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[0], DNNL_MAX_NDIMS);
    seed = get_array_hash(seed, desc.padding[1], DNNL_MAX_NDIMS);
    // Accumulator type
    seed = hash_combine(seed, static_cast<size_t>(desc.accum_data_type));
    // Combined hash for (de-)convolution desc
    return seed;
}
```
All the functions that are responsible for computing hash values should be implemented
in the same manner.

To make the functions work with `std::unordered_map` a specialization of
`std::hash` for `key_t` should be injected in the std namespace.

Specialization of `std::hash` for `key_t` is defined as follows:

```cpp
namespace std {
template <>
struct hash<dnnl::impl::primitive_hashing::key_t> {
    using argument_type = dnnl::impl::primitive_hashing::key_t;
    using result_type = std::size_t;
    result_type operator()(const argument_type &key) const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        // Compute hash for primitive_kind_, attr_, impl_id_ and impl_nthr_
        seed = hash_combine(seed,
                hash_combine(0, static_cast<size_t>(key.primitive_kind_)));
        seed = hash_combine(seed, get_attr_hash(key.attr_));
        seed = hash_combine(seed, hash_combine(0, key.impl_id_));
        seed = hash_combine(seed, hash_combine(0, key.impl_nthr_));
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.engine_kind_)));
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.runtime_kind_)));
        seed = hash_combine(
                seed, hash_combine(0, static_cast<size_t>(key.device_id_)));
        // Combine hash for op_desc with the computed hash
        // ...
        seed = get_array_hash(seed, key.mds.data(), (int)key.mds.size());
        return seed;
    }
};
```

Another requirement to the `key_t` is that it should be possible to compare them.

The comparison operator for convolution operation descriptors looks as follows:
```cpp
#define COMPARE_DESC_MEMBERS(m) lhs.m == rhs.m
#define COMPARE_DESC_ARRAY_MEMBERS(m, s) utils::array_cmp(lhs.m, rhs.m, s)

inline bool operator==(
        const convolution_desc_t &lhs, const convolution_desc_t &rhs) {
    bool ret = COMPARE_DESC_MEMBERS(primitive_kind)
            && COMPARE_DESC_MEMBERS(prop_kind)
            && COMPARE_DESC_MEMBERS(alg_kind)
            && COMPARE_DESC_MEMBERS(src_desc)
            && COMPARE_DESC_MEMBERS(diff_src_desc)
            && COMPARE_DESC_MEMBERS(weights_desc)
            && COMPARE_DESC_MEMBERS(diff_weights_desc)
            && COMPARE_DESC_MEMBERS(bias_desc)
            && COMPARE_DESC_MEMBERS(diff_bias_desc)
            && COMPARE_DESC_MEMBERS(dst_desc)
            && COMPARE_DESC_MEMBERS(diff_dst_desc)
            && COMPARE_DESC_ARRAY_MEMBERS(strides, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(dilates, DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[0], DNNL_MAX_NDIMS)
            && COMPARE_DESC_ARRAY_MEMBERS(padding[1], DNNL_MAX_NDIMS)
            && COMPARE_DESC_MEMBERS(accum_data_type);
    return ret;
}
```
All the comparison operators should be implemented in the same manner.


### Primitive Cache Interface
The interface of the cache looks as follows:

```cpp
struct primitive_cache_t  {
    using key_t = primitive_hashing::key_t;
    using value_t = std::shared_ptr<dnnl::impl::primitive_t>;

    virtual int get_capacity() const = 0;
    virtual status_t set_capacity(int capacity) = 0;

    virtual void add(const key_t &key, const value_t &impl) = 0;
    virtual value_t get(const key_t &key) = 0;

    virtual ~primitive_cache_t() = default;
};
```
Basically, cache implementations can use any eviction policy as long as the
implementations provide the aforementioned interface.
The proposal is to use the LRU (Least Recently Used) eviction policy as it seems
more appropriate for the oneDNN needs.

The global cache is maintained as the simplest singleton which can be implemented
as follows:
```cpp
primitive_cache_t &primitive_cache() {
    constexpr int capacity = 1024;
    // LRU based cache implementation
    static lru_primitive_cache_t instance(capacity);
    return instance;
}
```

## User-facing Primitive Abstraction
The user-facing primitive abstraction is represented inside the library as
`primitive_iface_t`.
The responsibility of the abstraction is to keep `dnnl::impl::primitive_t`
(primitive implementation) and `dnnl::impl::scratchpad_t>`. The abstraction
looks as follows:
```cpp
struct primitive_iface_t {
    // ...
    std::shared_ptr<dnnl::impl::primitive_t> primitive_;
    std::unique_ptr<dnnl::impl::scratchpad_t> scratchpad_;
    // ...
};
```
This abstraction will be extended so that the cache can be implemented. This
will be described further in the this RFC.

## Decouple Engine from Primitive Descriptor
The oneDNN primitive descriptor contains a pointer to the engine which it's
associated with. When creating a primitive implementation the primitive descriptor
is copied and stored in the the primitive implementation. The problem is that
the primitive descriptor doesn't own the engine therefore it's required for the
engine to stay alive until the primitive is destroyed.
This means that the primitive implementation cannot be stored in the global cache
because if the engine is destroyed then the primitive implementation should be
destroyed as well.

The proposal is to decouple engine from primitive descriptor by introducing
`primitive_desc_iface_t` so that the primitive implementation can be stored
in the cache.

Currently the primitive descriptor looks as follows:
```cpp
struct primitive_desc_t {
    dnnl::impl::engine_t *engine_;
    dnnl::impl::primitive_attr_t attr_;
    // ...
};
```

The proposed `primitive_desc_iface_t` looks as follows:
```cpp
struct primitive_desc_iface_t {
    dnnl::impl::engine_t *engine_;
    std::shared_ptr<dnnl::impl::primitive_desc_t> pd_;
    // ...
};
```

Once `primitive_desc_iface_t` is introduced the `primitive_iface_t` will be changed
as follows:

```cpp
struct primitive_iface_t {
    // During `primitive_iface_t` creation `primitive_desc_iface_t` is also
    // created.
    primitive_iface_t(const std::shared_ptr<dnnl::impl::primitive_t> &primitive,
            engine_t *engine) : primitive_(primitive),
            pd_(std:make_unique<primitive_desc_iface_t>(primitive_->pd(), engine)) {}

    std::shared_ptr<dnnl::impl::primitive_t> primitive_;
    std::unique_ptr<dnnl::impl::scratchpad_t> scratchpad_;
    // The primitive_desc_iface_t resides here
    std::unique_ptr<primitive_desc_iface_t> pd_;
};
```

The `pd_` and `primitive_` share the same `dnnl::impl::primitive_desc_t` with
`std::shared_ptr`.

Since engine is no longer a part of `dnnl::impl::primitive_desc_t` that affects
`dnnl::impl::primitive_t` and `dnnl::impl::primitive_desc_t` initialization:
```cpp
// All, except reorders
virtual status_t init(); // old interface
virtual status_t init(engine_t *engine); // new interface
// Reorders
virtual status_t init(); // old interface
virtual status_t init(engine_t *engine, engine_t *src_engine, engine_t *dst_engine); // new interface
```
The interface of the method that is responsible for creating `dnnl::impl::primitive_t`
should be modified as follows:

```cpp
// Old interface
virtual dnnl::impl::status_t create_primitive(dnnl::impl::primitive_t **primitive) const = 0;
// New interface
virtual status_t create_primitive(std::shared_ptr<dnnl::impl::primitive_t> &primitive,
        engine_t *engine, bool is_primitive_nested = true) const = 0;
```
The `create_primitive` method with the new interface will be responsible for
interacting with the cache.

The `primitive_desc_iface_t` abstraction will responsible for creating `primitive_iface_t`
with the following method:

```cpp
struct primitive_desc_iface_t {
    // ...
    status_t create_primitive_iface(primitive_iface_t **primitive_iface) const {
        // Step 1: create dnnl::impl::primitive_t or get it from the cache
        std::shared_ptr<dnnl::impl::primitive_t> p;
        auto status = pd_->create_primitive(p, engine(), false);
        if (status != status::success) return status;
        // Step 2: create primitive_iface_t, init it and return to user
        primitive_iface_t *p_iface = nullptr;
        CHECK(safe_ptr_assign<primitive_iface_t>(
                p_iface, new primitive_iface_t(p, engine())));
        status = p_iface->init();
        if (status != status::success) {
            delete p_iface;
            return status;
        }
        (*primitive_iface) = p_iface;
       return status::success;
    }
    // ...
};
```

## Nested Primitives
The next step towards making primitive implementation global cache friendly is
to get rid of `primitive_iface_t` instances inside `dnnl::impl::primitive_t`.
Currently all nested primitives are created and held as `primitive_iface_t`
which contains a pointer to the engine and sometimes a scratchpad.

The proposal is to replace all the nested primitives with `dnnl::impl::primitve_t`
to make it semantically correct as `dnnl::impl::primitive_t` should not contain
`primitive_iface_t`. And scratchpad should be propagated from the top level
primitive implementation to the nested ones.
The nested `dnnl::impl::primitive_t` can also be taken from the cache.


## OpenCL Kernels
GPU primitive implementations contain `cl_kernel` which is `cl_context` dependent
and therefore is engine dependent too. If a user re-creates engine with different
`cl_context` for each primitive creation then the global cache becomes pretty
much useless.

The proposal is to make the GPU primitive `cl_context` independent. OpenCL provides
a way to get a binary representation of the `cl_kernel` which doesn't depend on
the `cl_context` and can be converted to the actual `cl_kernel`. This can be achieved
by replacing all `cl_kernel`s in the GPU primitive implementations with their
binary representations.
The actual kernel will be created at primitive creation time. This will be described
in the next section.

## Primitive Resource
Primitives implementations must be immutable so that they can be used from different
threads (thread-safety).
The `dnnl::impl::primitive_t` provides only one non-constant member function `init`
that is used during its creation. Once the initialization is completed the
primitive implementation cannot be changed.
However each primitive implementation can potentially have a part which cannot
be immutable, for instance some primitives need to have so-called constant memory
which is used for holding scales. For GPU this becomes a problem as the memory
is `cl_context` dependent therefore it cannot be held inside the primitive
implementation.

The proposal is to introduce a primitive resource abstraction. The idea is that
each primitive implementation should be able to create a resource and put there
everything it needs to run, which cannot be stored in the cache as part of the
primitive implementation. To create the resource each primitive implementation
can override function `create_resource`.

Instance of the abstraction should reside inside `primitive_iface_t` therefore
the abstraction cannot be GPU specific and should be common. The common resource
abstraction is defined as follows:

```cpp
struct resource_t {
    virtual ~resource_t() = default;
};
```
Each nested primitive implementation can also require a resource. The proposal is
to introduce a resource mapper abstraction which will be responsible for holding
resources for a particular primitive implementation and providing corresponding
mapping between primitive implementation and its resource.
Interaction with the mapper happens in two steps:
1. Initialization. Each primitive implementation can override `create_resource`
member function that is responsible for creating a certain derived from resource_t
object and for filling it with some content, e.g. memory for scales, OpenCL kernels etc...
2. Passing it to the execution function which extracts needed resources and
uses them at execution time. The mapper is passed to the execution function
with the execution context.

The mapper takes ownership of all resources it has therefore it should be
responsible for destroying them as well.
The resource mapper abstractions is defined as follows:

```cpp
struct resource_mapper_t {
    using key_t = const dnnl::impl::primitive_t;
    using mapped_t = std::unique_ptr<resource_t>;

    resource_mapper_t() = default;

    bool has_resource(key_t *k) const {
        return primitive_to_resource_.count(k);
    }

    void add(key_t *k, mapped_t &&r) {
        assert(primitive_to_resource_.count(k) == 0);
        primitive_to_resource_.emplace(k, std::move(r));
    }

    template <typename T>
    const T *get(key_t *k) const {
        assert(primitive_to_resource_.count(k));
        return static_cast<T *>(primitive_to_resource_.at(k).get());
    }

private:
    resource_mapper_t(const resource_mapper_t &other) = delete;
    resource_mapper_t &operator=(const resource_mapper_t &other) = delete;

    std::unordered_map<key_t *, mapped_t> primitive_to_resource_;
};
```
In the case of asynchronous execution the same implementations will share the
same resource.

Currently only GPU primitives require to have a resource which will be responsible
for holding OpenCL kernels and so-called constant memory. The resource for GPU is
defined as follows:

```cpp
struct gpu_resource_t : public resource_t {
    // compute::kernel_t is an abstraction that holds either:
    // 1. Binary representation of an OpenCL kernel
    // 2. The actual OpenCL kernel
    // Note: this resource holds compute::kernel_t with the actual OpenCL kernel
    using key_kernel_t = compute::kernel_t::id_t;
    using mapped_kernel_t = compute::kernel_t;

    using key_memory_t = int;
    using mapped_memory_t = std::unique_ptr<memory_storage_t>;

    gpu_resource_t() = default;

    status_t add_kernel(compute::kernel_t::id_t kernel_id,
            const compute::kernel_t &kernel) {
        if (!kernel) return status::success;
        assert(kernel_id_to_kernel_.count(kernel_id) == 0);
        kernel_id_to_kernel_.emplace(kernel_id, kernel);
        return status::success;
    }

    const compute::kernel_t &get_kernel(key_kernel_t id) const {
        assert(kernel_id_to_kernel_.count(id));
        const auto &kernel = kernel_id_to_kernel_.at(id);
        assert(kernel);
        return kernel;
    }

    void add_memory_storage(key_memory_t idx, mapped_memory_t &&m) {
        assert(idx_to_memory_storage_.count(idx) == 0);
        if (!m) return;
        idx_to_memory_storage_.emplace(idx, std::move(m));
    }

    const memory_storage_t *get_memory_storage(int idx) const {
        assert(idx_to_memory_storage_.count(idx) != 0);
        return idx_to_memory_storage_.at(idx).get();
    }

    DNNL_DISALLOW_COPY_AND_ASSIGN(gpu_resource_t);

private:
    gpu_resource_t(const gpu_resource_t &other) = delete;
    gpu_resource_t &operator=(const gpu_resource_t &other) = delete;

    std::unordered_map<key_kernel_t, mapped_kernel_t> kernel_id_to_kernel_;
    std::unordered_map<key_memory_t, mapped_memory_t> idx_to_memory_storage_;
};
```

The `compute::kernel_t` encapsulates `compute::kernel_impl_t` which implements
all the logic. The `compute::kernel_impl_t` for OpenCL kernel is defined as follows:

```cpp
class ocl_gpu_kernel_t : public compute::kernel_impl_t {
public:
    // This is a public ctor that is used to create ocl_gpu_kernel_t
    // with the binary state.
    ocl_gpu_kernel_t(const std::vector<unsigned char> &binary,
            const std::string &binary_name)
        : state_(state_t::binary)
        , ocl_kernel_(nullptr)
        , binary_(binary)
        , binary_name_(binary_name) {
        MAYBE_UNUSED(state_);
    }

    ~ocl_gpu_kernel_t() override;

    cl_kernel ocl_kernel() const {
        assert(state_ == state_t::kernel);
        return ocl_kernel_;
    }

    status_t parallel_for(stream_t &stream, const compute::nd_range_t &range,
            const compute::kernel_arg_list_t &arg_list) const override;

    status_t realize(
            compute::kernel_t *kernel, engine_t *engine) const override;

    const char *name() const {
        assert(state_ == state_t::binary);
        return binary_name_.c_str();
    }

    enum class state_t { binary, kernel };

private:
    // This ctor can only be used by `realize` method to create
    // ocl_gpu_kernel_t with the kernel state.
    ocl_gpu_kernel_t(cl_kernel ocl_kernel)
        : state_(state_t::kernel), ocl_kernel_(ocl_kernel) {}

private:
    state_t state_;
    cl_kernel ocl_kernel_;
    std::vector<unsigned char> binary_;
    std::string binary_name_;
};

```

To unify common GPU-specific functionality a common GPU layer should be introduced.
The layer will be responsible for implementing that functionality:
1. `create_resource` - generalized method that creates `gpu_resource_t`
2. `create_kernel(s)` - creates `compute::kernel_t` with a binary representation inside
3. `nested_primitives` - primitive implementation overrides this method to
return a vector of pointers to the nested primitives, if applicable. This is
required to create resource for them in the `create_resource` method.
4. `init_res_storage` - primitive implementation overrides this method to
initialize memory storage with scales or zero points, if applicable. This is required
to create resource for them in the `create_resource` method.
5. `parallel_for` - gets resource for the given implementation and extracts the
`compute::kernel_t` that contains the actual OpenCL kernel and submits it for
execution.

The GPU layer is defined as follows:

```cpp
struct gpu_primitive_t : public primitive_t {
    using primitive_list_t = std::vector<const dnnl::impl::primitive_t *>;

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;
        auto r = std::make_unique<gpu_resource_t>();
        if (!r) return status::out_of_memory;
        for (const auto &rk : registered_kernels_) {
            if (!rk) continue;
            compute::kernel_t realized_kernel;
            // `realize` creates `compute::kernel_t` that contains the actual OpenCL kernel
            // out of the `compute::kernel_t` that contains binary representation
            CHECK(rk.realize(&realized_kernel, engine));
            r->add_kernel(rk.id(), realized_kernel);
        }
        CHECK(init_res_storage(engine, r.get()));
        mapper.add(this, std::move(r));

        for (auto const &np : nested_primitives()) {
            // some nested primitives are created on demand
            if (np) CHECK(np->create_resource(engine, mapper));
        }
        return status::success;
    }

    status_t create_kernels(engine_t *engine,
            std::vector<compute::kernel_t> *kernels,
            const std::vector<const char *> &kernel_names,
            const compute::kernel_ctx_t &kernel_ctx) {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        CHECK(compute_engine->create_kernels(
                kernels, kernel_names, kernel_ctx));
        // Register the `compute::kernel_t` that contains binary representation
        register_kernels(*kernels);
        return status::success;
    }

    status_t create_kernel(engine_t *engine, compute::kernel_t *kernel,
            const char *kernel_name, const compute::kernel_ctx_t &kernel_ctx) {
        std::vector<compute::kernel_t> kernels(1);
        auto status
                = create_kernels(engine, &kernels, {kernel_name}, kernel_ctx);
        if (status == status::success) *kernel = kernels[0];
        return status;
    }

protected:
    virtual primitive_list_t nested_primitives() const { return {}; }

    virtual status_t init_res_storage(
            engine_t *engine, gpu_resource_t *r) const {
        return status::success;
    }

    status_t parallel_for(const gemm_exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) const {
        return parallel_for(ctx.get_resource_mapper(), ctx.stream(), range,
                kernel, arg_list);
    }

    status_t parallel_for(const exec_ctx_t &ctx,
            const compute::nd_range_t &range, const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) const {
        return parallel_for(ctx.get_resource_mapper(), ctx.stream(), range,
                kernel, arg_list);
    }

private:
    void register_kernels(const std::vector<compute::kernel_t> &kernels) {
        for (const auto &k : kernels) {
            registered_kernels_.push_back(k);
        }
    }

    status_t parallel_for(const resource_mapper_t *resource_mapper,
            stream_t *stream, const compute::nd_range_t &range,
            const compute::kernel_t &kernel,
            const compute::kernel_arg_list_t &arg_list) const {

        compute::compute_stream_t *compute_stream
                = utils::downcast<compute::compute_stream_t *>(stream);
        const auto *resource = resource_mapper->get<gpu_resource_t>(this);
        const auto &realized_kernel = resource->get_kernel(kernel.id());

        CHECK(compute_stream->parallel_for(range, realized_kernel, arg_list));
        return status::success;
    }

    std::vector<compute::kernel_t> registered_kernels_;
};
```

## Reorder Primitive Descriptor
To create a primitive descriptor for reorders 3 engines are required:
1. `src` engine
2. `dst` engine
3. The main engine on which the primitive is running. It is one of the aforementioned

The main engine can be held inside the `primitive_desc_iface_t` but the rest of them
should reside somewhere else.

The proposal is to extended `primitive_desc_iface_t` in the following way to hold
those engines:
```cpp
struct reorder_primitive_desc_t : public primitive_desc_iface_t {
    // ...
    engine_t *src_engine_;
    engine_t *dst_engine_;
    engine_t *scratchpad_engine_;
    // ...
};
```

## Multithreading
Some users may want to create oneDNN primitives in parallel to speed-up first
primitive creation. It can be important for GPU primitives because creation of
the GPU primitives is slow, ~300ms.

The proposal is to use functionality providing by `std::promise` and `std::shared_future`.
This will allow to establish efficient communication between threads to reduce
pressure on memory while creating primitives. For instance, if several threads request
the same primitive, then only one of them will create it and others will wait.
Once the primitive creation is completed, this event will be communicated
to the waiting threads so that they can take the created primitive.

The `primitive_cache_t::value_t` should be modified as follows:
```cpp
struct primitive_cache_t {
    struct cache_value_t {
        std::shared_ptr<dnnl::impl::primitive_t> primitive;
        // If status is not `success` then `primitive == nullptr` is true
        status_t status;
    };
    using key_t = primitive_hashing::key_t;
    // The `shared_future` allows multiple primitives to share the same `future`
    // and therefore get a notification and created primitive.
    using value_t = std::shared_future<cache_value_t>;
    // ...
};
```

The common function which will be responsible for creating `dnnl::impl::primitive_t`
is defined as follows:
```cpp
template <typename impl_type, typename pd_t>
static status_t create_primitive_common(
        std::shared_ptr<primitive_t> &primitive, const pd_t *pd,
        engine_t *engine, bool use_global_scratchpad,
        bool is_primitive_nested) {

    auto &global_primitive_cache = primitive_cache();
    primitive_hashing::key_t key(pd, engine, dnnl_get_max_threads());

    std::promise<primitive_cache_t::cache_value_t> p_promise;
    const bool need_lock = !is_primitive_nested;
    // Try to get the shared future from the cache, if it's missing then
    // a shared future with no shared state is returned and the passed
    // shared future is added, otherwise a valid shared future is returned
    // and no insertion is performed.
    auto p_future = global_primitive_cache.get_or_add(
            key, p_promise.get_future(), need_lock);

    bool cache_hit = p_future.valid();

    auto status = status::success;
    std::shared_ptr<primitive_t> p;

    if (cache_hit) {
        // The requested primitive is present in the cache or is being
        // created by another thread.
        p = p_future.get().primitive;
        if (!p) return p_future.get().status;
    } else {
        // The requested primitive is NOT present in the cache therefore
        // we have to create it and notify the waiting threads
        // once the creation is done.
        p = std::make_shared<impl_type>(pd);
        status = p->init(engine, use_global_scratchpad);
        if (status != status::success) {
            // Communicate an error.
            p_promise.set_value({nullptr, status});
            // Remove the shared future from the cache because it's
            // invalidated. An invalidated shared future is the one that
            // stores a nullptr.
            global_primitive_cache.remove_if_invalidated(key, need_lock);
            return status;
        } else {
            // Store the created primitive in the shared future and notify
            // the waiting threads.
            p_promise.set_value({p, status});
        }
    }
    primitive = p;
    return status;
}
```

