# Support for DPCPP Primitives in oneDNN

## Introduction
As part of the oneAPI strategy oneDNN is going to migrate from OpenCL to
DPCPP kernels in the nearest future. Currently oneDNN doesn't provide
a mechanism for writing DPCPP kernels, this RFC describes a proposal on
introducing it into the library.


## Proposal
The proposal is to implement support for DPCPP kernels based on SYCL 1.2.1
specification taking into account the upcoming changes in SYCL 2020 to make
the transition smooth.

The goal of this RFC is to cover the main technical aspects of enabling DPCPP
kernels support in the library.
The implementation of this proposal should allow to start the development of
the DPCPP primitives.

### Memory Models
SYCL supports two memory models, USM and buffer.
1. USM (Unified shared memory) is a pointer-based model that provides a flexible
way to manage memory. Memory allocations are represented as pointers therefore
all memory arguments can be stored in an allocated USM array and passed to a
DPCPP kernel. Basically, this allows to implement a DPCPP kernel which works for
an arbitrary number of memory arguments.
2. Buffer is a more restrictive model compared to the USM one because the
buffers can only be accessed with a special SYCL abstraction - accessor.
The DPCPP kernel can only take accessors as memory arguments and the number of
accessors must be known at compile time.

The proposal is to introduce `sycl_memory_arg_t` abstraction that will make it
possible to implement a memory model agnostic kernel. The abstraction
encapsulates both a USM pointer and accessor but only one of them can be
active at the same time.

Requirements to `sycl_memory_arg_t`:
1. Must be a trivially copyable type
2. Must be a standard layout type (until SYCL 2020)

Each implementation of SYCL memory storage (USM and buffer) will be responsible
for creating `sycl_memory_arg_t` with `get_memory_arg` function. The interface
of the function and implementation of `sycl_memory_arg_t` are shown below.

```cpp
class sycl_memory_storage_base_t : public memory_storage_t {
    // ...
     virtual sycl_memory_arg_t get_memory_arg(cl::sycl::handler &cgh);
    // ...
 };
```

```cpp
struct sycl_memory_arg_t {
    static constexpr auto rw_mode = cl::sycl::access::mode::read_write;
    using acc_t = cl::sycl::accessor<uint8_t, 1, rw_mode,
            cl::sycl::access::target::global_buffer>;

    sycl_memory_arg_t(void *usm, cl::sycl::handler &cgh)
        : usm(usm), acc(/* dummy accessor */) {}
    sycl_memory_arg_t(const acc_t &acc) : usm(nullptr), acc(acc) {}
    // This method must be called only from inside a kernel.
    void *get_pointer() { return usm ? usm : acc.get_pointert().get(); }

    void *usm;
    acc_t acc;
};
```
In SYCL 1.2.1 accessor is not a default constructible type therefore a dummy
accessor is required to be able to construct `sycl_memory_arg_t` for USM.
This problem will be solved in SYCL 2020 as the default constructor will be
added.

There are a few option on how to get the dummy accessor:
1. Introduce a simple buffer singleton that can be used to create the dummy accessor.
    * Pros
        * An isolated solution, doesn't require to modify key oneDNN
        abstractions.
    * Cons
        * Serialized execution when 2 or more queues are used due to the
        common dependency.
        * OpenCL may not work well when its objects are stored globally which
        may result in a crash or hang due to the order of unloading oneDNN and
        OpenCL libraries.
2.  Create a dummy buffer in SYCL stream.
    * Pros
        * Parallel execution when 2 or more queues are used.
        * Doesn't introduce a global buffer object therefore no potential
        problems with OpenCL
    * Cons
        * The workaround becomes part of the key oneDNN abstraction

The proposal is to go with the second option because the trade-offs are better
and the workaround will be removed after transition to SYCL 2020.
For the second option the interface of `get_memory_arg` should be extended to
take a stream.

Conceptually the `sycl_memory_arg_t` approach has an issue with a redundant
data field. SYCL has a limitation on the size of the arguments that can be
passed to a kernel, currently the limitation is 1024 bytes. The USM pointer
always takes 8 bytes and accessor currently takes 32 bytes. However, this is the
only way to have a single kernel for different memory models, which justifies this
approach.

### DPCPP Kernel
In SYCL a kernel can be defined in three different ways:
1. Defining the kernel as a cl_kernel and use OpenCL interop API to pass
arguments to the kernel, this is the way that is currently used in the library.
2. Defining the kernel as a lambda function that implicitly captures all the
kernel arguments.
3. Defining the kernel as a function object that takes all the arguments with
a constructor and stores them as data fields.

In general, the 2nd and 3rd ways can be used interchangeably however, the
proposal is to use the 3rd one in oneDNN as it brings more to the table:
1. Inheritance
2. Templates (assuming that we stick with C++11)
3. Modularity

The proposed DPCPP primitive skeleton looks as follows:

```cpp
using namespace cl::sycl;

// DPCPP function object.
template <typename src_data_t, typename wei_data_t, typename dst_data_t>
struct ref_conv_kernel_t {
    ref_conv_kernel_t(const sycl_conv_conf_t &conf, sycl_memory_arg_t &src,
            sycl_memory_arg_t &wei, sycl_memory_arg_t &dst)
        : conf(conf)
        , src(src)
        , wei(wei)
        , dst(dst) {
    }

    void operator()(sycl::nd_item<3> nd_item) {
        auto *src_p = (src_data_t *)src.get_pointer();
        auto *wei_p = (wei_data_t *)wei.get_pointer();
        auto *dst_p = (dst_data_t *)dst.get_pointer();

        auto md_src = conf.md_src;
        auto md_wei = conf.md_wei;
        auto md_dst = conf.md_dst;
        // ...
        // Kernel code.
    }

private:
    // DPCPP kernel arguments.
    sycl_conv_conf_t conf;
    sycl_memory_arg_t src;
    sycl_memory_arg_t wei;
    sycl_memory_arg_t dst;
};

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
struct ref_conv_primitive : public gpu_primitive_t {
    struct pd {
        status_t init(engine_t *engine) {
            // Initialize conf.
            // ...
        }
        // The conf abstraction will be described in the specialization
        // constants section.
        sycl_conv_conf_t conf;
        // ...
    };

    status_t init(engine_t *engine) {
        // Create compute::kernel_t that contains binary representation of the
        // DPCPP kernel.
        // ...
        // This will be described in the primitive cache section.
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        // CTX_{IN|OUT}_SYCL_STORAGE casts memory_storage_t to
        // sycl_memory_storage_base_t.
        auto *src = CTX_IN_SYCL_STORAGE(DNNL_ARG_SRC);
        auto *wei = CTX_IN_SYCL_STORAGE(DNNL_ARG_WEIGHTS);
        auto *dst = CTX_OUT_SYCL_STORAGE(DNNL_ARG_DST);

        auto *sycl_stream = utils::downcast<sycl_stream_t *>(ctx.stream());
        auto &queue = sycl_stream->queue();

        // Ideally, this should be abstracted away with the `compute` abstractions.
        // But perhaps it makes sense to be more verbose at the very beginning
        // of DPCPP kernels development to better understand the use cases.
        auto event = queue.submit([&](handler &cgh) {
            cgh.depends_on(sycl_stream->get_deps());

            sycl_memory_arg_t src_memory_arg = src->get_memory_arg(cgh);
            sycl_memory_arg_t wei_memory_arg = wei->get_memory_arg(cgh);
            sycl_memory_arg_t dst_memory_arg = dst->get_memory_arg(cgh);
            // Optionally create local accessors.
            // ...

            // Get the compute kernel from resource and query the sycl_kernel
            // from it.
            compute::kernel_t k = gpu_primitive_t::get_kernel(ctx, kernel));
            auto sycl_kernel = downcast<sycl_gpu_kernel_t<ref_conv_kernel_t> *>(
                    k.impl())->sycl_kernel();
            // The API requires to pass the kernel functor along with the
            // sycl_kernel.
            // Basically, the functor acts as a container for kernel arguments.
            ref_conv_kernel_t<src_data_t, wei_data_t, dst_data_t> functor(
                pd()->conf, src_memory_arg, wei_memory_arg, dst_memory_arg);

            cgh.parallel_for(sycl_kernel, nd_range, functor);
        });

        sycl_stream->set_deps({event});
        return status::success;
    }

    using src_data_t = typename prec_traits<src_type>::type;
    using wei_data_t = typename prec_traits<wei_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    compute::kernel_t kernel;
};
```
The skeleton doesn't contain a couple of parts namely the code related to
primitive cache and specialization constants that was omitted for simplicity.
The omitted parts will be described in the corresponding sections of this RFC.

One of the drawbacks of this DPCPP primitive design is that the primitive must
be explicitly instantiated for each data type configuration which increases the
size of the library. It's difficult to accurately quantify the impact because
there is no real, well optimized DPCPP kernel yet, but an instantiation of a
reference binary primitive for one data type configuration increases the library
size by 35 kilobytes.
One of the ways to get rid of the templates in DPCPP primitives is to use
`if`/`switch` statements for performing computations for a particular data type
within a block. However, even if this approach works the code maintainability
will deteriorate.
Another way to get rid of the templates is to use a jitter but it would be a
completely different way of writing kernels and it has nothing to with writing
DPCPP kernels.
The proposal is to use templates for DPCPP primitives.

### Specialization Constants
One of the OpenCL advantages is an ability to compile OpenCL kernels online.
This allows to use a preprocessor to specify some values at runtime which is
essential for implementing highly optimized kernels as it gives the compiler more
room for optimization. The oneDNN OpenCL kernels largely depend on that.
SYCL programming model doesn't provide a way to specify runtime values with macros.
The DPCPP kernels are compiled with an offline compiler that outputs an
intermediate representation - SPIR-V, which will be finalized for a particular
device at runtime.
The specialization constants are intended to compensate the lack of macros
support in SYCL, this is not an equal replacement though, e.g. macros can be
used to specify data types while the specialization constants cannot.

The specialization constants represent constant variables whose values are not
known at offline compilation time but they will be specialized at runtime
providing more room for optimization during finalization of the SPIR-V
(JIT generation stage).

* In SYCL 1.2.1 the specialization constant type must meet POD requirements.
Currently the support is limited to `int` and `float` data types (there is
a request from oneDNN to support POD types)
* In SYCL 2020 the specification doesn't explicitly state the requirements.
But based on how the type is used in the SYCL API it must be a literal type.

The proposal is to introduce a `sycl_<primitive name>_conf_t` abstraction
that will be responsible for passing all non-memory arguments to a
DPCPP kernel.


The specialization constants will be used to specify the entire
`sycl_<primitive name>_conf_t` object at runtime. The compiler doesn't generate
kernel arguments for specialization constants therefore the limitation on
the total size of all kernel arguments is not applicable to the object. However,
the objects have to meet the requirement for the total size of the kernel
arguments because of the reasons described below.

There will be two categories of the DPCPP primitives:
1. Reference primitives - primitives that work on any device: Intel, Nvidia, etc.
A primitive from this category is expected to meet the following requirements:
    * Don't use specialization constants, at least until SYCL 2020 because
    in SYCL 2020 specialization constants will be emulated by the SYCL Runtime for
    the devices that do not support them (e.g. Nvidia).
    * Total size of the kernel arguments is less than 1024 bytes.
    * (Optional) If it's not possible to meet the requirement (1024 bytes) then
    there is an option to use scratchpad for passing non-memory arguments to a
    kernel. This is a very suboptimal solution which should be used only if there
    are no other options.
2. Optimized primitives - primitives that use special features to achieve best
possible performance. A primitive from this category is expected to meet the
following requirements:
    * Should use specialization constants for `sycl_<primitive name>_conf_t`
    * Should use vendor specific extensions

An example of the `sycl_<primitive name>_conf_t` abstraction is shown below.

```cpp
struct sycl_conv_conf_t {
    sycl_md_t md_src;
    sycl_md_t md_wei;
    sycl_md_t md_dst;

    prop_kind_t prop_kind;

    bool with_bias;
    bool with_groups;
    // ...
};
```
The native oneDNN memory descriptor is well generalized and hence is large and
contains a lot of redundant information. To meet the requirement for the total
size of the kernel arguments the memory descriptor should be trimmed down. Because
it should be used in both reference and optimized primitives. The proposal is to
introduce `sycl_md_t`.
This also gives the following benefits:
1. Reduced size of the generated SPIR-V.
2. Reduced memory descriptor will be less general therefore offset calculation
can probably be implemented more efficiently.

It's worth mentioning that similar abstractions are already implemented for
the OpenCL kernels. Although they can probably be modified for reuse, the
suggestion is to isolate DPCPP related abstractions from the OpenCL ones to
facilitate transition from OpenCL to DPCPP in the future and to avoid modifying
the OpenCL primitives.

The specialization constants are not part of the SYCL 1.2.1 specification.
However DPCPP implementation provides an experimental support for specialization
constants for trivial data types (`int` and `float`), and there is a request
from oneDNN to support POD types. The support can be implemented before the Gold
release because it can be re-used in DPCPP implementation based on SYCL 2020.

Once the support for POD types is implemented the specialization constants
can be used in the library.

oneDNN also provides support for runtime primitive arguments. Since
`sycl_<primitive name>_conf_t` is initialized at primitive descriptor creation
time it doesn't have the actual arguments, also a DPCPP kernel is built during
primitive creation meaning that the specialization constants are evaluated to
wildcards. The proposal is to introduce `sycl_<primitive name>_rt_conf_t` which
will be responsible for passing runtime arguments to the DPCPP kernel. It will
be passed as an additional kernel argument. The modified DPCPP primitive
skeleton (<> denotes omitted code) with support for specialization constants
looks as follows:

```cpp
using namespace cl::sycl;
using namespace cl::sycl::experimental;

template <typename src_data_t, typename wei_data_t, typename dst_data_t>
struct ref_conv_kernel_t {
    // This is allowed to use the name of the function object as a specialization
    // constant ID.
    using sc_conf_t = spec_constant<sycl_conv_conf_t, class ref_conv_kernel_t>;

    // The `sycl_conv_rt_conf_t` is always passed to the primitive that supports
    // runtime arguments.
    ref_conv_kernel_t(const sc_conf_t &sc_conf, const sycl_conv_rt_conf_t &rt_conf,
            const sycl_memory_arg_t &src, sycl_memory_arg_t &wei, sycl_memory_arg_t &dst) {
        : sc_conf(sc_conf)
        , src(src)
        , wei(wei)
        , dst(dst) {}

    void operator()(sycl::nd_item<3> nd_item) {
        // <getting pointers from memory arguments>

        // These values can be known at JIT generation time (finalizing the
        // SPIR-V for a particular device). Or they can be runtime values.
        const auto &conf = sc_conf.get();
        auto md_src = conf.md_src.is_runtime() ? rt_conf.md_src : conf.md_src;
        auto md_src = conf.md_wei.is_runtime() ? rt_conf.md_wei : conf.md_wei;
        auto md_src = conf.md_dst.is_runtime() ? rt_conf.md_dst : conf.md_dst;
        // ...
        // Kernel code.
    }

private:
    // DPCPP implementation based on SYCL 1.2.1 requires to pass specialization
    // constants as kernel arguments.
    sc_conf_t sc_conf;

    // <memory arguments>
};

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
struct ref_conv_primitive : public gpu_primitive_t {
    // <pd implementation>

    status_t init(engine_t *engine) {
        // Note: the verbosity is intentional to show the steps in one place,
        // these steps will be hidden in the `compute` abstractions.
        auto *sycl_engine = utils::downcast<sycl_engine_base_t *>(engine);
        program p(sycl_engine.context());
        p.set_spec_constant<class ref_conv_kernel_t>(pd()->conf)
        // The built program contains evaluated specialization constants.
        p.build_with_kernel_type<ref_conv_kernel_t>();
        assert(p.get_state() == program_state::linked);
        // ...
        // Create compute::kernel_t that contains binary representation of the
        // DPCPP kernel.
        // ...
        // This will be described in the primitive cache section.
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        // <getting memory storages>

        auto *sycl_stream = utils::downcast<sycl_stream_t *>(ctx.stream());
        auto &queue = sycl_stream->queue();

        auto event = queue.submit([&](handler &cgh) {
            cgh.depends_on(sycl_stream->get_deps());
            // <getting memory arguments from memory storages>
            // <getting sycl kernel from resource>

            // Since SYCL 1.2.1 doesn't natively support specialization constants
            // they may not play well with some SYCL API.
            // Particularly for this case, the API requires to pass the
            // specialization constants to the kernel even if the kernel has
            // been created from binary and there is no need in this.
            // To work-around it, a dummy specialization constant is passed to
            // the kernel because it has no effect.

            // Create the dummy specialization constant.
            program p_dummy(queue.get_context());
            auto sc_dummy = p.set_spec_constant<class ref_conv_kernel_t>(pd()->conf);

            ref_conv_kernel_t<src_data_t, wei_data_t, dst_data_t> functor(
                sc_dummy, src_memory_arg, wei_memory_arg, dst_memory_arg);

            cgh.parallel_for(sycl_kernel, nd_range, functor);
        });

        sycl_stream->set_deps({event});
        return status::success;
    }

// ...
};
```

### Primitive Cache
The SYCL specification states that SYCL implementations are free to cache state
globally, but it's not required. The DPCPP implementation implements a sort of
cache for the generated kernels. The generated kernel is tied to the context
it's associated with meaning that the kernel is available as long as the
context is alive.

The DPCPP cache has the following drawbacks:
1. The cache capacity and eviction policy are not specified
2. The cache doesn't provide any API for controlling it
3. The cache is context dependent

The compiler team confirmed that this is an internal optimization and there is
no plan to expose any control over it to users or specify its behavior unless
they get a lot of requests and strong justification.

Because of the aforementioned details oneDNN cannot rely on the DPCPP cache.
The proposal is to stick with the oneDNN primitive cache.

To be primitive cache friendly the DPCPP primitives must meet the same
requirements as OpenCL primitives:
1. Context dependent parts of the primitive implementation must not reside in
the primitive implementation. The primitive resource mechanism should be used
to handle that.
2. There must be a way to get binary representation of the kernel
3. There must be a way to create the kernel out of the binary representation

The modified DPCPP primitive skeleton that illustrates how to fulfil
requirements looks as follows:

```cpp
using namespace cl::sycl;

template <typename kernel_name_t>
struct sycl_gpu_kernel_t : public compute::kernel_impl_t {
    // This is a public ctor that is used to create sycl_gpu_kernel_t with
    // binary state.
    sycl_gpu_kernel_t(const std::vector<char> &binary);

    // Creates a new sycl_gpu_kernel_t with kernel state.
    status_t realize(compute::kernel_t *kernel, engine_t *engine) const override {
        // SYCL 1.2.1 doesn't provide any API to create a kernel out of its
        // binary representation therefore backend API should be used directly.

        // ...
        // Requirement #1: the DPCPP kernel is a context dependent part,
        // compute::kernel_t that is returned by this function is stored in the
        // primitive resource.
        // Requirement #3: this function creates a DPCPP kernel out of the
        // binary using backend specific API.
        auto *sycl_engine = downcast<sycl_gpu_engine_t *>(engine);
        const unsigned char *b = (const unsigned char *)binary.data();
        size_t b_sz = binary.size();

        // Dispatching between backends.

        sycl::kernel *k = nullptr;
        void *handle_to_destroy = nullptr;

        // OpenCL backend:
        cl_device_id ocl_device = sycl_engine->ocl_device();
        cl_context ocl_context = sycl_engine->ocl_context();
        cl_int err;
        cl_program ocl_program = clCreateProgramWithBinary(ocl_context, 1,
                &ocl_device, &b_sz, &b, nullptr, &err);
        OCL_CHECK(err);
        err = clBuildProgram(ocl_program, 1, &ocl_device, nullptr, nullptr,
                nullptr);
        OCL_CHECK(err);
        sycl::program sycl_program(sycl_engine->context(), ocl_program);
        // SYCL program retains a reference to ocl program.
        OCL_CHECK(clReleaseProgram(program));
        assert(sycl_program.get_state() == program_state::linked);
        k = new sycl::kernel(sycl_program.get_kernel<kernel_name_t>))

        // Level0 backend (if applicable):
        ze_module_desc_t desc {ZE_MODULE_DESC_VERSION_CURRENT};
        desc.format = ZE_MODULE_FORMAT_NATIVE;
        desc.inputSize = b_sz;
        desc.pInputModule = b;
        desc.pBuildFlags = "";
        desc.pConstants = nullptr;
        auto ze_device = (ze_device_handle_t)sycl_engine->device().get();
        ze_module_handle_t ze_module;
        handle_to_destroy = ze_module;
        CHECK(func_zeModuleCreate(ze_device, &desc, &ze_module, nullptr));
        sycl::program sycl_program = level0::make<sycl::program>(
                sycl_engine->context(), ze_module);
        k = new sycl::kernel(sycl_program.get_kernel<kernel_name_t>());

        // Backend independent part.
        (*kernel) = compute::kernel_t(
                new sycl_gpu_kernel_t<kernel_name_t>(k, handle_to_destroy));
        return status::success;
    }

    ~sycl_gpu_kernel_t() {
        // The module can be deleted only after a kernel submission.
        if (handle_to_destroy)
            auto ze_module = reinterpret_cast<ze_module_handle_t>(handle_to_destroy);
            func_zeModuleDestroy(ze_module);
        }
    }
private:
    sycl_gpu_kernel_t(sycl::kernel *k, void *handle_to_destroy)
        : sycl_kernel(k), handle_to_destroy(handle_to_destroy) {}

    std::unique_ptr<sycl::kernel> sycl_kernel;
    void *handle_to_destroy = nullptr;

    std::vector<char> binary;
    // ...
};
```

```cpp
// <ref_conv_kernel_t implementation>

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
struct ref_conv_primitive : public gpu_primitive_t {
    // <pd implementation>

    status_t init(engine_t *engine) {
        // Note: the verbosity is intentional to show the steps in one place,
        // these steps will be hidden in the `compute` abstractions.

        // <create a program and build it with kernel type>

        // The binary contains the specialization constants that have been
        // evaluated to the runtime values.
        auto kernel_binary = p.get_binaries()[0];
        // Requirement #2: the binary is context independent therefore
        // storing it in the primitive implementation is allowed.
        kernel = compute::kernel_t(new sycl_gpu_kernel_t<ref_conv_kernel_t>(kernel_binary));
        // ...
    }

    // <execute>

    compute::kernel_t kernel;
// ...
};
```
### Post Ops
Support for post ops is essential functionality in the library. The library
supports three post op kinds that can be enabled in DPCPP primitive as follows:
1. Eltwise/binary: there are two options to support it:
    * Create post ops as separate DPCPP kernels and submit them separately.
    This is not a performant solution but is general enough to support many
    use cases and it can be used as a fallback.
    * Create an injector class that fulfil the requirements for DPCPP kernel
    arguments.
        * For the binary post op it will require to pass an array of
        `sycl_memory_arg_t` to the kernel. This will also require to make
        `sycl_memory_arg_t` default constructible and add one more
        field: `bool is_empty`.
2. Sum: this post op can be implemented with `sycl_<primitive name>_conf_t`,
that is it can carry the the scale factor.
3. Depthwise: the suggestion is to postpone it until it's proven that it gives
performance on GPU.

To support an arbitrary number of post ops `sycl_<primitive name>_conf_t` will
contain an array (`primitive_kind_t po_kinds[max_pos]`) describing the chain of
the post ops.
The maximum number of post ops should be limited to a reasonable number.

The injector class can be implemented as follows:
```cpp
// The math utils must be adapted to DPCPP, most likely dedicated DPCPP math
// utils will be a better solution.
// According to the SYCL specification it supports many different built-in math
// function.
template <typename T, typename A,
        typename U = typename std::remove_reference<T>::type>
inline typename std::enable_if<!std::is_integral<U>::value, U>::type
relu_fwd(T s, A alpha) {
    return s > 0 ? s : (U)(s * alpha);
}

// With specialization constants these values are know at JIT generation time.
struct ref_conv_conf_t {
    // Assuming that convolution can support up to 4 eltwise post ops
    // (this is subject to change).
    static constexpr max_pos = 4;
    ref_eltwise_scalar_fwd_t eltwise_pos[max_pos];
    primitive_kind_t po_kinds[max_pos];
    // Indicate the actual number of post ops.
    int n_pos;

    float sum_scale;
    // ...
};

// This is taken from the CPU implementation.
// DPCPP primitive can either re-use this one or implement a new one.
// This class must meet the requirements for DPCPP kernel arguments, basically
// it should stay POD.
// The methods must also meet the requirements for functions that can be called
// from a DPCPP kernel. E.g. no asserts are allowed.
struct ref_eltwise_scalar_fwd_t {
    ref_eltwise_scalar_fwd_t() = default;

    ref_eltwise_scalar_fwd_t(alg_kind_t alg, float alpha, float beta, float scale);
    ref_eltwise_scalar_fwd_t(const post_ops_t::entry_t::eltwise_t &eltwise);

    float compute_scalar(float s) {
        return compute_eltwise_scalar_fwd(alg_, s, alpha_, beta_) * scale_;
    }

    alg_kind_t alg_;
    float alpha_;
    float beta_;
    float scale_;

private:
    float compute_eltwise_scalar_fwd(alg_kind_t alg, float s, float alpha, float beta) {
        float d = 0.f;
        switch (alg) {
           case eltwise_relu: d = relu_fwd(s, alpha); break;
           // ...
        }
        return d;
    }
};

// Kernel operator of the ref_conv_kernel_t.
void operator()(sycl::nd_item<3> nd_item) {

    for (int e = 0; e < conf_.n_pos; e++) {
        // ...
        int eltwise_pos_idx = 0
        switch (po_kinds[e]) {
            // acc and dst_f are expected to be float.
            // SFINAE can be used to implement a special case for `int` data types.
            case sum: acc += conf_.sum_scale * dst_f; break;
            case eltwise: acc = eltwise_pos[eltwise_pos_idx++].compute_scalar(acc); break;
            // ...
        };
    }
}
```

### Auxiliary Functionality
All math and other utils functions, q10n functions, offset calculation must
be adapted to DPCPP.

### Directory Structure (taken from Eugene's proposal)
```
onednn
└── src
    ├── common
    ├── cpu
    ├── gpu
    │   ├── compute
    │   ├── ocl                                 # OpenCL GPU service layer and primitives
    │   │   ├── gen9_conv_fwd_f32.cl
    │   │   ├── gen9_convolution.cpp
    │   │   ├── gen9_convolution.hpp
    │   │   ├── ref_shuffle.cl
    │   │   ├── ref_shuffle.cpp
    │   │   └── ref_shuffle.hpp
    │   └── sycl                                # SYCL GPU service layer and primitives
    │       ├── gen9_pooling.cpp
    │       └── gen9_pooling.hpp
    └── sycl                                    # SYCL service layer (common between CPU and GPU)
```
According to the proposal some refactoring is required to move GPU specific
code from `src/sycl` to `src/gpu/sycl`.


### Implementation List
The idea is to have a common implementation list for OpenCL and DPCPP primitives.
Since the implementations that come first are expected to be most performant, there
should be a way to test DPCPP primitives until performance of the DPCPP primitives
is (at least) on par with the OpenCL ones.
There are two options:
1. Introduce a build time option `DNNL_USE_DPCPP_PRIMITIVES`.
2. Introduce an environment variable `DNNL_USE_DPCPP_PRIMITIVES`.

The environment variable option seems more appealing because it doesn't require
to re-build the library.
When the environment variable/build time option is specified the non-DPCPP
primitives are filtered out so that the DPCPP ones can be tested. This also
requires to modify tests to skip unimplemented cases.

### Transition to SYCL 2020
According to the SYCL 2020 specification it brings a lot of new features and
makes SYCL API more convenient. However, to migrate from SYCL 1.2.1 to SYCL 2020
some changes to the library architecture will be required. The kernels are not
expected to be affected.

The main changes in SYCL 2020:
1. Removed `sycl::program` and introduced `sycl::module` which is similar to
`sycl::program` but not the same. It will affect mostly the primitive cache.
2. The specialization constants became a part of the standard and have slightly
different API.

When the time comes there will be another RFC describing the transition.

