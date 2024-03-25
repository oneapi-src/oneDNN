# Refactor GPU Architecture

## Background

The current design of the library architecture responsible for GPU was
built with Intel in mind and therefore is Intel centric. Because of that,
the basic GPU abstractions (e.g. `compute_engine`, etc) are tightly tied
to OpenCL and nGEN and other Intel specifics such as information about
device architecture, stepping, etc. Such design had been working fine
up until support for NVIDIA (and later AMD) GPUs was introduced. Currently,
the NVIDIA and AMD specific abstractions are built on top of the basic GPU
abstractions and therefore have dependencies on OpenCL and nGEN even though
there is no need in them. Furthermore, oneDNN now has generic SYCL
kernels that can be used on a variety of different architectures
that are supported by the SYCL ecosystem. The SYCL kernels also use
abstractions that are built on top of the basic GPU abstractions and
therefore have the same issue.

This RFC proposes a new organization of GPU kernels and abstractions
to clearly separate independent functionality, get rid of unnecessary
dependencies and have a flexible enough architecture to make adding support
for new vendors easier.

## Proposal

Reorganize the GPU kernels and abstractions according the the following
schema: Vendor / Technology

With this schema the GPU directory will have subdirectories that
correspond to the vendors: `intel`, `nvidia`, `amd`, `generic`, etc.
Each of the subdirectories may have technology specific sub-subdirectories:
`sycl`, `ocl`, `jit`, etc.

Pros:
* The schema provides enough flexibility to enable new vendors and extend
the already supported ones
* Clustering the functionality and abstractions around vendors is
convenient as a vendor can share the same functionality/abstractions across
different technologies (e.g. the compute layer for Intel vendor)
* Currently, functionality and abstractions are fully or partially
clustered around vendors therefore the new schema should not cause a lot of
confusion among the developers, it should also positively affect the
implementation cost
* The schema also provides sufficient configurability, e.g
the generic vendor can be enabled along with nvidia or amd, or it can be
enabled individually

```bash
├── common/            # Common files for the library (e.g. API, basic abstractions, etc).
├── xpu/               # xpu stands for Heterogeneous Runtime. All vendor agnostic code for runtimes resides here.
│   ├── sycl/          # Vendor agnostic code for SYCL runtime.
│   ├── ocl/           # Vendor agnostic code for OpenCL runtime.
│   └── ...
├── cpu/               # CPU code.
│   ├── sycl/          # CPU specific SYCL code.
│   └── ...
└── gpu/               # GPU code.
    ├── intel/         # Intel vendor specific code.
    │   ├── compute/   # Compute layer abstractions.
    │   ├── ocl/       # OpenCL code.
    │   ├── jit/       # JIT (nGEN) code.
    │   ├── sycl/      # SYCL code.
    │   │   ├── l0/    # Level zero backend specific code.
    │   │   ├── ocl/   # OpenCL backend specific code.
    │   │   └── ...
    │   └── ...
    ├── nvidia/        # NVIDIA vendor specific code.
    ├── amd/           # AMD vendor specific code.
    └── generic/       # Generic (vendor agnostic) GPU kernels.
        ├── sycl/      # SYCL kernels.
        └── ...
```

### Vendor Macros

oneDNN can be built for a number of GPU vendors, such as Intel, Nvidia and AMD. 
Adding support for generic GPU vendor is in progress. In order to be able to dispatch
between GPU vendor specific code at compile time the corresponding macros will be added.

```cpp
// dnnl_config.hpp.in

/// No vendor (corresponding runtime is disabled)
#define DNNL_VENDOR_NONE 0u

/// Intel vendor
#define DNNL_VENDOR_INTEL 1u

/// NVIDIA vendor
#define DNNL_VENDOR_NVIDIA 2u

/// AMD vendor
#define DNNL_VENDOR_AMD 4u

// oneDNN GPU vendor
#cmakedefine DNNL_GPU_VENDOR DNNL_VENDOR_${DNNL_GPU_VENDOR}

// ...
```

### Namespaces and Prefixes

All vendor-specific code should be enclosed in a namespace that has the vendor's name.
Based on the directory structure described above the following namespaces will be
introduced:
* `dnnl::impl`
    * `dnnl::impl::xpu::sycl`
    * `dnnl::impl::xpu::ocl`
* `dnnl::impl::cpu`
    * `dnnl::impl::cpu::sycl`
* `dnnl::impl::gpu`
    *  `dnnl::impl::gpu::intel`
        * `dnnl::impl::gpu::intel::compute`
        * `dnnl::impl::gpu::intel::ocl`
        * `dnnl::impl::gpu::intel::jit`
    * `dnnl::impl::gpu::nvidia`
    * `dnnl::impl::gpu::amd`
    * `dnnl::impl::gpu::generic`
        * `dnnl::impl::gpu::generic::sycl`

Given that the namespaces already prevent name collisions adding the prefixes
such as `sycl_`, `ocl_`, etc is redundant and therefore the suggestion is to
drop the prefixes. For example, `dnnl::impl::gpu::ocl::ocl_gpu_engine_t` will
be converted to `dnnl::impl::gpu::intel::ocl::engine_t`.

### Implementation List

Currently there are separate implementation lists for each vendor, which
introduces unnecessary redundancy and increases maintenance cost. The proposal
is to unify the implementation lists into a single one. The CPU primitives already
use the approach.

There will be introduced a set of new macros to instantiate vendor specific and
generic implementations.

The macros for INTEL, NVIDIA and AMD vendors assume that all implementations
within a single vendor can be enabled at once regardless of the kernel language.

The macros for the GENERIC vendor can be either truly generic or kernel language specific:
* Truly generic implementation is the one that is not tied to any vendor and kernel language, e.g.
  an implementation of the concat primitive based on reorders
* Runtime specific implementation is the one that is tied to a particular kernel language,
  e.g. SYCL generic implementations (written in generic SYCL)

 The concat, sum and reorder primitives require specialized versions of some of the the
 macros because their `pd_t::create` functions have unique signatures.

The following macros are the helper ones and not expected to be used directly:
* GPU_INSTANCE - primary macro to instantiate a GPU implementation
* GPU_CONCAT_INSTANCE - a specialization for the concat primitive
* GPU_SUM_INSTANCE - a specialization for the sum primitive
* GPU_REORDER_INSTANCE - a specialization for the reorder primitive

The following are the _primary_ vendor specific macros that are used to instantiate vendor specific and generic implementations.
* GPU_INSTANCE_INTEL - instantiate implementations for the Intel vendor
* GPU_INSTANCE_NVIDIA - instantiate implementations for the Nvidia vendor
* GPU_INSTANCE_AMD - instantiate implementations for the AMD vendor
* GPU_INSTANCE_GENERIC_SYCL - instantiate implementations written in SYCL for the generic vendor (currently enabled only for Nvidia vendor)
* GPU_INSTANCE_GENERIC - instantiate implementations that a truly generic. Truly generic
  implementation is the one that is not tied to any vendor and runtime, e.g. an implementation
  of the concat primitive based on reorders

The following are the specializations of the vendor specific macros for the concat primitive:
* GPU_CONCAT_INSTANCE_INTEL
* GPU_CONCAT_INSTANCE_NVIDIA
* GPU_CONCAT_INSTANCE_AMD
* GPU_CONCAT_INSTANCE_GENERIC_SYCL
* GPU_CONCAT_INSTANCE_GENERIC

The following are the specializations of the vendor specific macros for the sum primitive:
* GPU_SUM_INSTANCE_INTEL
* GPU_SUM_INSTANCE_NVIDIA
* GPU_SUM_INSTANCE_AMD
* GPU_SUM_INSTANCE_GENERIC_SYCL
* GPU_SUM_INSTANCE_GENERIC

The following are the specializations of the vendor specific macros for the reorder primitive:
* GPU_REORDER_INSTANCE_INTEL
* GPU_REORDER_INSTANCE_NVIDIA
* GPU_REORDER_INSTANCE_AMD
* GPU_REORDER_INSTANCE_GENERIC_SYCL
* GPU_REORDER_INSTANCE_GENERIC

### Affected Basic Abstractions

The new schema will require moving a lot of parts of the library around
and while most of the changes are probably just an implementation detail
there are a few major changes that are worth describing in this RFC.

#### Engine

The engine abstraction gets affected by the changes the most.

There is a `compute_engine_t` abstraction that serves as a base class for
`sycl_engine_base_t` and `ocl_gpu_engine_t` classes. The problem is that each
vendor specific SYCL engine has to be inherited from `sycl_engine_base_t`and
therefore it cannot be derived from `compute_engine_t`. The solution to the problem
is to decouple the SYCL specific but vendor agnostic part of the `sycl_engine_base_t`
class and move it over to a new class. The `ocl_gpu_engine_t` class doesn't have
this particular problem but it makes sense to also decouple the OpenCL specific but
vendor agnostic part of it and move it over to the new class.

The new class is defined as follows:
```cpp
// Location: src/common
namespace dnnl::impl {

struct engine_impl_t {
    engine_impl_t() = delete;
    // Constructs an impl::engine_impl_t instance using the provided engine kind, runtime kind and index.
    engine_impl_t(engine_kind_t kind, runtime_kind_t runtime_kind, size_t index);
    virtual ~engine_impl_t() = default;

    // Returns engine kind.
    engine_kind_t kind() const;
    // Returns runtime kind.
    runtime_kind_t runtime_kind() const;
    // Returns index.
    size_t index() const;

    // Returns engine ID.
    virtual engine_id_t engine_id() const;

#ifdef ONEDNN_BUILD_GRAPH
    // Returns allocator.
    void *get_allocator() const;
    // Sets allocator.
    void set_allocator(graph::allocator_t *alloc);
#endif
    // Initializes the engine implementation.
    virtual status_t init();
    // Creates a stream implementation.
    virtual status_t create_stream_impl(impl::stream_impl_t **stream_impl, unsigned flags) const;
    // Returns a buffer alignment.
    virtual int get_buffer_alignment() const;

private:
    engine_kind_t kind;
    runtime_kind_t runtime_kind;
    size_t index;

#ifdef ONEDNN_BUILD_GRAPH
    graph::allocator_t allocator;
#endif
};

} // namespace dnnl::impl
```

```cpp
// Location: src/xpu/ocl
namespace dnnl::impl::xpu::ocl {

struct engine_impl_t : public dnnl::impl::engine_impl_t {
    engine_impl_t() = delete;
    // Constructs an xpu::ocl::engine_impl_t instance using the provided OpenCL device, OpenCL context and index
    engine_impl_t(cl_device_id device, cl_context context, size_t index);
    ~engine_impl_t() override = default;
    
    // Initializes the engine implementation.
    status_t init() override;

    // Creates a memory storage.
    status_t create_memory_storage(memory_storage_t **storage, engine_t *engine,
            unsigned flags, size_t size, void *handle) const;
    
    // Creates a memory storage.
    status_t create_stream_impl(
            impl::stream_impl_t **stream_impl, unsigned flags) const override;
            
    // Returns OpenCL device.
    cl_device_id device() const;
    // Returns OpenCL context.
    cl_context context() const;
    // Returns OpenCL platform.
    cl_platform_id platform();
    // Returns engine ID.
    engine_id_t engine_id() const override;
    // Returns device name.
    const std::string &name() const;
    // Returns runtome version.
    const runtime_version_t &runtime_version() const;
    // Returns buffer alignment.
    int get_buffer_alignment() const override;

private:
    xpu::ocl::wrapper_t<cl_device_id> device;
    xpu::ocl::wrapper_t<cl_context> context;
    cl_platform_id platform;
    bool is_user_context;

    std::string name;
    runtime_version_t runtime_version;
};

} // namespace dnnl::impl::xpu::ocl
```

```cpp
// Location: src/xpu/sycl
namespace dnnl::impl::xpu::sycl {

struct engine_impl_t : public dnnl::impl::engine_impl_t {
    engine_impl_t() = delete;
    // Constructs an xpu::sycl::engine_impl_t instance using the provided SYCL device, SYCL context and index.
    engine_impl_t(engine_kind_t kind, const ::sycl::device &device,
            const ::sycl::context &context, size_t index);
    ~engine_impl_t() override = default;

    // Initializes the engine implementation.
    status_t init() override;

    // Creates a memory storage.
    status_t create_memory_storage(memory_storage_t **storage, engine_t *engine,
            unsigned flags, size_t size, void *handle) const;

    // Creates stream implementation.
    status_t create_stream_impl(
            impl::stream_impl_t **stream_impl, unsigned flags) const override;
            
    // Returns SYCL device.
    const ::sycl::device &device() const;
    // Returns SYCL context.
    const ::sycl::context &context() const;
    // Returns backend.
    backend_t backend() const;
    // Returns engine ID.
    engine_id_t engine_id() const override;
    // Returns device name.
    const std::string &name() const;
    // Returns runtime version.
    const runtime_version_t &runtime_version() const;
    // Returns buffer alignment.
    int get_buffer_alignment() const override;
    // Returns true if system memory allocations are supported.
    bool mayiuse_system_memory_allocators() const;

private:
    ::sycl::device device;
    ::sycl::context context;

    backend_t backend;

    std::string name;
    runtime_version_t runtime_version;
};

} // namespace dnnl::impl::xpu::sycl
```

When it comes to the generic SYCL kernels there are 3 scenarios that should be
supported:
* Only generic kernels are enabled
* NVIDIA and generic kernels are enabled
* AMD and generic kernels are enabled

According to the oneDNN programming model only 1 GPU engine can be enabled at a time.
In the second and third scenarios the NVIDIA and AMD specific engines will be used
for the generic SYCL kernels. In order to support the first scenario a separate, new
engine is required. As a result the generic SYCL kernels will be used with different
engines. To make it work there will be added a new `dnnl::impl::gpu::engine_t` class
that will hold `dnnl::impl::engine_impl_t` that will point to `dnnl::impl::sycl::engine_impl_t`
in this particular case. The `dnnl::impl::gpu::engine_t` will also provide
an interface to return a pointer to `dnnl::impl::engine_impl_t` so that the generic
SYCL kernels can query all SYCL specific information they may need (e.g. supported
sub-group sizes).

The base `engine_t` class contains a pointer to the engine implementation and interface
to initialize it:
```cpp
// Location: src/common
namespace dnnl::impl {
struct engine_t : dnnl::impl::c_compatible {
    // Each engine creates an engine implementation and passes it down to the base engine class.
    engine(dnnl::impl::engine_impl_t *impl) : impl(impl), ... {}

    // Returns a pointer to the engine implementation.
    const dnnl::impl::engine_impl_t *impl() const;

protected:
    // Each engine has to call `init_impl()` to initialize the implementation.
    dnnl::impl::status_t init_impl();
private:
    // Points to a particular engine implementation.
    std::unique_ptr<dnnl::impl::engine_impl_t> impl;
};
```

The `dnnl::impl::gpu::engine_t` is defined as follows:
```cpp
// Location: src/gpu
namespace dnnl::impl::gpu {

struct engine_t : public dnnl::impl::engine_t {
    // Interfaces to get implementation lists.
    const impl_list_item_t *get_reorder_implementation_list(
            const memory_desc_t *src_md, const memory_desc_t *dst_md) const override;
    const impl_list_item_t *get_concat_implementation_list() const override;
    const impl_list_item_t *get_sum_implementation_list() const override;
    const impl_list_item_t *get_implementation_list(const op_desc_t *desc) const override;

    // Returns an alignment that should be used for memory allocations on the GPU.
    int get_buffer_alignment() const;

    // Return a service stream.
    status_t get_service_stream(impl::stream_t *&stream) override;
        
private:
    std::unique_ptr<impl::stream_t> service_stream;
};

} // namespace dnnl::impl::gpu
```

The following vendor specific engine classes will replace the currently
implemented ones.

This class takes over responsibility of `compute_engine_t` class.
```cpp
// Location: src/gpu/intel
namespace dnnl::impl::gpu::intel {

struct engine_t : public dnnl::impl::gpu::engine_t {};

} // namespace dnnl::impl::gpu::intel
```

This class takes over responsibility of `sycl_engine_base_t` class.
```cpp
// Location: src/gpu/intel/sycl
namespace dnnl::impl::gpu::intel::sycl {

struct engine_t : public dnnl::impl::gpu::intel::engine_t {
protected:
    // Convenience interface to simplify access to `impl` within the class.
    const dnnl::impl::xpu::sycl::engine_impl_t *impl() const;
};

} // namespace dnnl::impl::gpu::intel::sycl
```

This class takes over responsibility of `ocl_gpu_engine_t` class.
```cpp
// Location: src/gpu/intel/ocl
namespace dnnl::impl::gpu::intel::ocl {

struct engine_t : public dnnl::impl::gpu::intel::engine_t {
protected:
    // Convenience interface to simplify access to `impl` within the class.
    const dnnl::impl::xpu::ocl::engine_impl_t *impl() const;
};

} // namespace dnnl::impl::gpu::intel::ocl
```

This class takes over responsibility of `sycl_cuda_engine_t` and `sycl_hip_engine_t` classes.
```cpp
// Location: src/gpu/nvidia
namespace dnnl::impl::gpu::nvidia {

struct engine_t : public dnnl::impl::gpu::engine_t {
protected:
    // Convenience interface to simplify access to `impl` within the class.
    const dnnl::impl::xpu::sycl::engine_impl_t *impl() const;
};

} // namespace dnnl::impl::gpu::nvidia


// Location: src/gpu/amd
namespace dnnl::impl::gpu::amd {

struct engine_t : public dnnl::impl::gpu::engine_t {
protected:
    // Convenience interface to simplify access to `impl` within the class.
    const dnnl::impl::xpu::sycl::engine_impl_t *impl() const;
};

} // namespace dnnl::impl::gpu::amd
```

This class will be used when only generic SYCL kernels are enabled.
```cpp
// Location: src/gpu/generic
namespace dnnl::impl::gpu::generic {

struct engine_t : public dnnl::impl::gpu::engine_t {
protected:
    // Convenience interface to simplify access to `impl` within the class.
    const dnnl::impl::xpu::sycl::engine_impl_t *impl() const;
};

} // namespace dnnl::impl::gpu::generic

```

The present inheritance chains for SYCL and OpenCL GPU engines are
the following:
* SYCL Intel and generic: `engine_t` -> `compute_engine_t` -> `sycl_engine_base_t` -> `sycl_gpu_engine_t`
* SYCL NVIDIA: `engine_t` -> `compute_engine_t` -> `sycl_engine_base_t` -> `sycl_cuda_engine_t`
* SYCL AMD: `engine_t` -> `compute_engine_t` -> `sycl_engine_base_t` -> `sycl_hip_engine_t`
* OpenCL (only Intel): `engine_t` -> `compute_engine_t` -> `ocl_gpu_engine_t`

The new inheritance chains for SYCL and OpenCL GPU engines are
the following:
* SYCL Generic: `engine_t` -> `gpu::engine_t` -> `gpu::generic::engine_t`
* SYCL Intel: `engine_t` -> `gpu::engine_t` -> `gpu::intel::engine_t` -> `gpu::intel::sycl::engine_t`
* SYCL NVIDIA: `engine_t` -> `gpu::engine_t` -> `gpu::nvidia::engine_t`
* SYCL AMD: `engine_t` -> `gpu::engine_t` -> `gpu::amd::engine_t`
* OpenCL: `engine_t` -> `gpu::engine_t` ->  `gpu::intel::engine_t` -> `gpu::intel::ocl::engine_t`

Reminder: the `gpu::engine_t` holds an `engine_impl_t` pointer that points to either
`sycl::engine_impl_t` or `ocl::engine_impl_t`.

#### Stream

Similar to the engine, the following stream abstractions will be introduced.

```cpp
// Location: src/common
namespace dnnl::impl {

struct stream_impl_t {
    stream_impl_t() = delete;
    // Constructs an impl::stream_impl_t instance using the provided stream flags.
    stream_impl_t(unsigned flags);
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    // Constructs an impl::stream_impl_t instance using the provided threadpool.
    stream_impl_t(threadpool_interop::threadpool_iface *threadpool);
#endif
    virtual ~stream_impl_t() = default;

    // Returns stream flags.
    unsigned flags() const;
    // Returns true if the profiling flag is set.
    bool is_profiling_enabled() const;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    // Returns threadpool.
    status_t get_threadpool(threadpool_interop::threadpool_iface **threadpool) const;
#endif

private:
    unsigned flags;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    threadpool_interop::threadpool_iface *threadpool;
#endif
};

} // namespace dnnl::impl
```
```cpp
// Location: src/xpu/ocl
namespace dnnl::impl::xpu::ocl {

struct stream_impl_t : public dnnl::impl::stream_impl_t {
    stream_impl_t() = delete;
    // Constructs an xpu::ocl::stream_impl_t instance using the provided stream flags.
    stream_impl_t(unsigned flags);
    // Constructs an xpu::ocl::stream_impl_t instance using the provided OpenCL queue and stream flags.
    stream_impl_t(cl_command_queue queue, unsigned flags);
    ~stream_impl_t() override;

    // Sets OpenCL queue.
    status_t set_queue(cl_command_queue queue);

    // Returns OpenCL queue.
    cl_command_queue queue();

    // Wait on the queue.
    status_t wait();

    // Copies data from src to dst. 
    status_t copy(impl::stream_t *stream, const memory_storage_t &src,
            const memory_storage_t &dst, size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep,
            xpu::stream_profiler_t *stream_profiler = nullptr);

    // Fills dst with pattern.
    status_t fill(impl::stream_t *stream, const memory_storage_t &dst,
            uint8_t pattern, size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep,
            xpu::stream_profiler_t *stream_profiler = nullptr);

    // Returns oneDNN OpenCL context.
    const xpu::ocl::context_t &ocl_ctx() const;
    // Returns oneDNN OpenCL context.
    xpu::ocl::context_t &ocl_ctx();
    // Returns oneDNN XPU context.
    xpu::context_t &ctx();
    // Returns oneDNN XPU context.
    const xpu::context_t &ctx() const;

    // Returns output OpenCL event.
    const xpu::ocl::wrapper_t<cl_event> &get_output_event() const;

    // Initializes flags.
    static status_t init_flags(unsigned *flags, cl_command_queue queue);

private:
    cl_command_queue queue;
    mutable utils::thread_local_storage_t<xpu::ocl::context_t> ctx;
};

} // namespace dnnl::impl::xpu::ocl
```
```cpp
// Location: src/xpu/sycl
namespace dnnl::impl::xpu::sycl {

struct stream_impl_t : public dnnl::impl::stream_impl_t {
    stream_impl_t() = delete;
    // Constructs an xpu::sycl::stream_impl_t instance using the provided stream flags.
    stream_impl_t(unsigned flags);
    // Constructs an xpu::sycl::stream_impl_t instance using the provided SYCL queue and stream flags.
    stream_impl_t(const ::sycl::queue &queue, unsigned flags);
    ~stream_impl_t() override = default;

    // Sets SYCL queue. 
    status_t set_queue(::sycl::queue queue);

    // Returns SYCL queue.
    ::sycl::queue *queue();

    // Wait on the queue.
    status_t wait();

    // Copies data from src to dst. 
    status_t copy(impl::stream_t *stream, const memory_storage_t &src, const memory_storage_t &dst,
            size_t size, const xpu::event_t &deps, xpu::event_t &out_dep,
            xpu::stream_profiler_t *stream_profiler = nullptr);
    // Fills dst with pattern.
    status_t fill(const memory_storage_t &dst, uint8_t pattern, size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep, xpu::stream_profiler_t *stream_profiler = nullptr);

    // Returns oneDNN SYCL context.
    const xpu::sycl::context_t &sycl_ctx() const;
    // Returns oneDNN SYCL context.
    xpu::sycl::context_t &sycl_ctx();
    // Returns oneDNN XPU context.
    xpu::context_t &ctx();
    // Returns oneDNN XPU context.
    const xpu::context_t &ctx() const;

    // Returns output SYCL event.
    ::sycl::event get_output_event();

    // Registers dependency.
    void register_deps(::sycl::handler &cgh) const;

    // Initializes flags.
    static status_t init_flags(unsigned *flags, ::sycl::queue &queue);

    // Returns a dummy accessor.
    template <::sycl::access_mode mode>
    ::sycl::accessor<uint8_t, 1, mode> get_dummy_accessor(
            ::sycl::handler &cgh);

private:
    std::unique_ptr<::sycl::queue> queue;
    mutable utils::thread_local_storage_t<xpu::sycl::context_t> ctx;
    // XXX: this is a temporary solution to make sycl_memory_arg_t
    // default constructible.
    xpu::sycl::buffer_u8_t dummy_buffer = xpu::sycl::buffer_u8_t(1);
};

} // namespace dnnl::impl::xpu::sycl
```
The `dnnl::impl::stream_t` is modified to contain a pointer to `impl::stream_impl_t`:
```cpp
// Location: src/common
namespace dnnl::impl {

struct stream_t : public impl::c_compatible {
    // Constructs an impl::stream_t instance using the provided engine and stream implementation.
    // This constructor substitutes the existing one.
    stream_t(dnnl::impl::engine_t *engine, dnnl::impl::stream_impl_t *impl);

    // Returns a non-const pointer to the stream implementation.
    impl::stream_impl_t *impl() { return impl.get(); }
protected:
    std::unique_ptr<dnnl::impl::stream_impl_t> impl;
};

} // namespace dnnl::impl
```

The `dnnl::impl::gpu::stream_t` is defined as follows:

```cpp
// Location: src/gpu
namespace dnnl::impl::gpu {

struct stream_t : public dnnl::impl::stream_t {
    // Copies data from src to dst.
    virtual status_t copy(const memory_storage_t &src,
            const memory_storage_t &dst, size_t size, const xpu::event_t &dep,
            xpu::event_t &out_dep)
            = 0;
    // Fills dst with pattern.
    virtual status_t fill(const memory_storage_t &dst, uint8_t pattern,
            size_t size, const xpu::event_t &deps, xpu::event_t &out_dep)
            = 0;

    // Returns oneDNN XPU context.
    virtual xpu::context_t &ctx() = 0;
    // Returns oneDNN XPU context.
    virtual const xpu::context_t &ctx() const = 0;

    // Returns stream profiler.
    virtual const xpu::stream_profiler_t &profiler() const;
    // Returns stream profiler.
    xpu::stream_profiler_t &profiler();

    // Returns frequency.
    virtual double get_freq(const xpu::event_t &event) const { return 0.0; }

protected:
    std::unique_ptr<xpu::stream_profiler_t> profiler;
};

} // namespace dnnl::impl::gpu
```

The following vendor specific stream classes will replace the currently
implemented ones. The vendor specific streams are responsible for initializing
the stream implementation.

This class takes over responsibility of `compute_stream_t` class.
```cpp
// Location: src/gpu/intel
namespace dnnl::impl::gpu::intel {

struct stream_t : public dnnl::impl::gpu::stream_t {};

} // namespace dnnl::impl::gpu::intel
```

This class takes over responsibility of `sycl_stream_t` class.
```cpp
// Location: src/gpu/intel/sycl
namespace dnnl::impl::gpu::intel::sycl {

struct stream_t : public dnnl::impl::gpu::intel::stream_t {};

} // namespace dnnl::impl::gpu::intel::sycl
```

This class takes over responsibility of `ocl_stream_t` class.
```cpp
// Location: src/gpu/intel/ocl
namespace dnnl::impl::gpu::intel::ocl {

struct stream_t : public dnnl::impl::gpu::intel::stream_t {};

} // namespace dnnl::impl::gpu::intel::ocl
```

This class takes over responsibility of `sycl_cuda_stream_t` and `sycl_hip_stream_t` classes.
```cpp
// Location: src/gpu/nvidia
namespace dnnl::impl::gpu::nvidia {

struct stream_t : public dnnl::impl::gpu::stream_t {};

} // namespace dnnl::impl::gpu::nvidia


// Location: src/gpu/amd
namespace dnnl::impl::gpu::amd {

struct stream_t : public dnnl::impl::gpu::stream_t {};

} // namespace dnnl::impl::gpu::amd
```

This class will be used when only generic SYCL kernels are enabled.
```cpp
// Location: src/gpu/generic
namespace dnnl::impl::gpu::generic {

struct stream_t : public dnnl::impl::gpu::stream_t {};

} // namespace dnnl::impl::gpu::generic

```

The current inheritance chains for SYCL and OpenCL GPU streams are the following:
* SYCL Intel and genetic: `stream_t` -> `compute_stream_t` -> `sycl_stream_t`
* SYCL NVIDIA: `stream_t` -> `compute_stream_t` -> `sycl_stream_t` -> `sycl_cuda_stream_t`
* SYCL AMD: `stream_t` -> `compute_stream_t` -> `sycl_stream_t` -> `sycl_hip_stream_t`
* OpenCL: `stream_t` -> `compute_stream_t` -> `ocl_stream_t`

The new inheritance chains for SYCL and OpenCL GPU streams are
the following:
* SYCL Generic: `stream_t` -> `gpu::stream_t` -> `gpu::generic::stream_t`
* SYCL Intel: `stream_t` -> `gpu::stream_t` -> `gpu::intel::stream_t` -> `gpu::intel::sycl::stream_t`
* SYCL NVIDIA: `stream_t` -> `gpu::stream_t` -> `gpu::nvidia::stream_t`
* SYCL AMD: `stream_t` -> `gpu::stream_t` -> `gpu::amd::stream_t`
* OpenCL: `stream_t` -> `gpu::stream_t` ->  `gpu::intel::stream_t` -> `gpu::intel::ocl::stream_t`

Reminder: the `gpu::stream_t` holds an `stream_impl_t` pointer that points to either
`sycl::stream_impl_t` or `ocl::stream_impl_t`.

