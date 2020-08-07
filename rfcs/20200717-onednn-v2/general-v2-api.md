# General v2 API

The discussion here will mostly focus on C++ API (`dnnl*.hpp`). Where important
the C API will be mentioned explicitly. Otherwise, the assumption that
statements made relate to both C and C++ APIs.


## 1. API Mock-up

Please check simplified [API directory](api-simplified/), which focuses on
C++ API only:
- Current version [`dnnl.hpp`](api-simplified/v1_dnnl.hpp)
- Suggested v2 version:
  - [`dnnl.hpp`](api-simplified/v2_dnnl.hpp)
  - [`dnnl_sycl.hpp`](api-simplified/v2_dnnl_sycl.hpp)
  - [`dnnl_threadpool.hpp`](api-simplified/v2_dnnl_threadpool.hpp)


## 2. Basis Principals

1. Currently the main `dnnl.hpp` _dynamically_ adds or modifies the functions
   depending on the API/RT the library was built with, by looking at
   `dnnl_config.h`. We suggest to split the headers into API/RT agnostic and
   API/RT dependent parts. A user will be responsible to include and call
   special functions that provide interoperability API between DNNL and API/RT,
   e.g. SYCL.
   - The library shipped could always include all of the headers, which reduces
     the divergence between the configurations.
   - The approach is generally less fragile.
   - Better alignment with potential future support for plugin-based model
     (see also C++-API over C-API).

2. C++ API is fully based on C API. DPC++ objects are passed by address casted
   to a `(const) void *`.

   Rationale:
   - The approach makes it more difficult to break ABI, as the only entry point
     to the library will be C-based (living in `dnnl*.h` files).
   - (weak argument) The approach is better aligned with plugin-model, if ever
     decide to go this route.
   - We could put all symbols to the library regardless of the API/RT it was
     built with. In this case even if user tries to call OpenCL-specific API
     with the library that was built w/o its support, the function with
     gracefully return `status::unimplemented` or so. This will make different
     application to "work" with arbitrary oneDNN library (meaning that they
     won't crash at load time due to undefined symbols). However, I think
     seeing the issue at run time make it easier to debug, so suggest to not
     implement this, at least at the beginning.

   This will require proper and thorough documenting the C-API (mostly for the
   DPC++ C++ counterpart), as the types would be erased. E.g. instead of an
   address on a SYCL buffer the API will take `void *`. As C-API for DPC++ is
   merely an implementation detail (entry point), we should encourage our users
   to avoid using it (at least for DPC++).

   Example:
   ``` cpp
   // ===========
   // dnnl_sycl.h
   // ===========

   // ...
   // @param sycl_dev_ptr Address of a SYCL device object casted to `void *`.
   // @param sycl_ctx_ptr Address of a SYCL context object casted to `void *`.
   // ...
   // @note
   //     Since the function signature takes `void *` instead of some of the
   //     parameters (type erasure), the compiler won't be able to check that
   //     the arguments are passed correctly. The users are encouraged to use
   //     the corresponding C++ API instead.
   dnnl_status_t dnnl_sycl_engine_create(dnnl_engine_t *engine,
           void *sycl_dev_ptr, void *sycl_ctx_ptr); // <-- type erasure

   // =============
   // dnnl_sycl.hpp
   // =============
   namespace dnnl {
   namespace sycl {
   inline engine engine_create(cl::sycl::device dev, cl::sycl::context ctx) {
     dnnl_engine_t e = nullptr;
     dnnl::error(
         dnnl_sycl_engine_create(&e, /* type erasure */ &dev, /* type erasure */ &ctx),
         "cannot create engine"
     );
     return dnnl::engine(e);
   }
   } // namespace sycl
   } // namespace dnnl
   ```


## 3. Suggested API

As it is mentioned above, it is suggested to split oneDNN API into
API/RT-agnostic that would continue living in `dnnl.hpp`, and interoperability
API going to `dnnl_$api.hpp` header file. Only API-specific headers are allowed
to make API-specific includes, e.g. only `dnnl_sycl.hpp` is allowed to include
`CL/sycl.hpp`. Luckily, the interoperability API is limited to creating few
oneDNN objects, such as engine, stream, and memory, and querying those.


### 3.1. Engine

There would be 2 ways to create an engine:

``` cpp
// =================
// dnnl_$runtime.hpp
// =================

// 1. The most advance, runtime-specific way of creating an engine that accepts
// API/RT-specific input arguments. Example below for DPC++ (similar for OCL).
dnnl::engine dnnl::sycl::engine_create(
        const cl::sycl::device &dev, const cl::sycl::context &ctx);

// ========
// dnnl.hpp
// ========

// 2. oneDNN v1.x compatible way
auto eng = engine(engine::kind, index);

// 2.a. In oneDNN v2.x the default value for index is 0:
auto eng = engine(engine::kind); // index = 0
```

Just like in oneDNN v1.x and `dev-v2` branch, the engine created with the
second (API/RT-agnostic way) would depend on the library configuration. Say,
for `cpu_dpcpp_gpu_dpcpp` configuration the engine will be backed up by DPC++.

Notice, there is no `engine::kind` argument in `dnnl::sycl::engine_create()`
(same for OpenCL). The assumption is that the kind could be derived from the
input device.


### 3.2. Stream

1. The stream story is pretty similar to engine's one: the `dnnl_$runtime.hpp`
   header will contain the constructors that could accept the runtime-specific
   objects. Examples:
   ``` cpp
   // =============
   // dnnl_sycl.hpp
   // =============
   stream stream_create(const engine &e, cl::sycl::queue &queue);

   // ===================
   // dnnl_threadpool.hpp
   // ===================

   stream stream_create(const engine &e, threadpool_iface *threadpool);
   ```

2. Since thread pool now deserves a separate header file we suggest to get rid
   of stream attributes, and pass the thread pool pointer directly to the
   stream constructor.

   - The biggest concern is a broken API, though the hope is that no one
     actually uses `stream_attr`, except for TF with Eigen thread pool. But
     given that the proposal anyways break the thread pool, that should be
     fine.

3. The stream flags are also dropped from the RT-specific constructors, as they
   could be derived from the corresponding RT objects (e.g. `sycl::queue`).
   - For DPC++ and OpenCL the order could be derived from the queue properties.
     The assumption that if the queue is `out_of_order`, no one would want to
     _simulate_ it is in-order, by passing us the corresponding flag. If they
     do, they could call `stream::wait()` on their own after each API call.
   - For Thread Pool the assumption that (just like regular C++ API) only
     in-order streams will be supported.
   - If really necessary we extend the constructors by returning back the flags
     with default value equals `stream::flags::default_flags` which will
     preserve API/ABI backwards compatibility.
   - On the other hand, since flags could contain (in future) extra meaning,
     such as enabling profiling (not quite relevant as this flag cannot be used
     for the already created queue), it could be safer to include them in the
     API. We also could make a semantic check on the flags and input queue:
     - `flags::out_of_order` and out-of-order queue is a valid combination
     - `flags::out_of_order` and in-order queue is a valid combination
     - `flags::in_order` and out-of-order queue is *not* a valid combination

   The suggestion is to drop the stream flags for RT-specific API. Of course,
   the flags stay in RT-agnostic constructor (`dnnl.hpp`).


### 3.3. Memory

1. Memory now has an extra constructor
   ``` cpp
   // dnnl.hpp
   dnnl::memory(const desc &md, const stream &astream, void *handle);
   ```
   That uses stream to do zero padding, if necessary. This could also be handy
   to properly _pin_ the memory.

2. Similarly to engine and stream objects, RT-specific headers will provide the
   interoperable API. For instance, to create a memory with sycl buffers, one
   will use on of the following functions:
   ``` cpp
    template <typename T, int ndims>
    memory memory_create(const desc &md, const engine &aengine,
            cl::sycl::buffer<T, ndims> buf, stream &s);

    template <typename T, int ndims>
    void memory_set_data_handle(memory &m, cl::sycl::buffer<T, ndims> b);
   ```

3. For DPC++ the [USM and Buffers Support](usm-and-buffer-support.md) discusses
   the support for buffers and USM. Assuming we choose to go with the suggested
   approach (that is supporting both USM and buffers at memory creation,
   allowing user to mix those at a primitive execution), the DPC++ specific
   header defines `dnnl::sycl::memory_kind` enum class, that declares different
   kinds of memory (device USM, shared USM, and buffers).
   - A memory object created through RT-agnostic API (`dnnl.hpp`) will use
     _device USM_ as an option that gives the best performance and usability.


The relevant piece of the `dnnl_sycl.hpp`:

``` cpp
namespace dnnl {
namespace sycl {

enum class memory_kind { // maybe memory_model is better name?
    usm_device, // default for SYCL-agnostic engine creation
    usm_shared,
    buffer,
};

// for mkind == buffer, handle could only be DNNL_MEMORY_{ALLOCATE,NONE}
memory memory_create(const desc &md, const engine &aengine, memory_kind mkind,
        void *handle = DNNL_MEMORY_ALLOCATE);
memory memory_create(const desc &md, const engine &astream, memory_kind mkind,
        void *handle = DNNL_MEMORY_ALLOCATE);

// API for sycl buffers The `memory_kind == buffer` is implied.
template <typename T, int ndims>
memory memory_create(const desc &md, const engine &aengine,
        cl::sycl::buffer<T, ndims> buf, stream &s);

// memory_kind could be changed during the lifetime, by setting the USM handle
// or SYCL buffer
memory_kind memory_get_memory_kind(const memory &amemory);

} // namespace sycl
} // namespace dnnl
```


### 3.4. Primitive Execution

DPC++ and OpenCL potentially might need to support events at the execution API.
For this purpose, the interoperability API is used:
``` cpp
// =============
// dnnl_sycl.hpp
// =============
cl::sycl::event dnnl::sycl::primitive_execute(const primitive &p,
        const stream &s, const std::unordered_map<int, memory> &args,
        const std::vector<cl::sycl::event> &dependencies = {});

// Similarly for dnnl_ocl.hpp, which we currently don't have, as only support
// in-order queues. Could add later, if required.
```

The `dnnl::primitive::execute()` function that lives in `dnnl.hpp` will assume
that there are no dependencies for the given primitive, and that the resulting
event is unused. That should work fine for the in-order streams or with
manual `stream::wait()` calls from a user side.


### 3.5 API Classes (NOT TO BE IMPLEMENTED)

> This is a brief introduction to the idea that is **not** suggested to be
> implemented and included here mostly due to the significance of it. More
> implementation details could be read in
> [Dropped Ideas: API Classes](dropped-ideas-api-classes.md).

It was suggested to introduce a new enum `dnnl::api`, which names the API
classes the library supports. For native C/C++ API the enum contains `api::c`.
Exactly this API class is fully defined in `dnnl.hpp`, and only engine kind
supported by this API is CPU. It was also suggested that for DPC++ library
configuration (`dnnl_cpu_dpcpp_gpu_dpcpp`) it is allowed to create engine with
native C API.  This will allow supporting native C++ applications as well as
DPC++ applications within a single configuration.

The list of `dnnl::api` values would be:

| `dnnl::api`       | API to work with | Memory buffer representation                        |
| :--               | :--              | :--                                                 |
| `api::c`          | Native C/C++     | `void *`                                            |
| `api::sycl`       | DPC++            | Further dispatch based on `dnnl::sycl::memory_kind` |
| `api::ocl`        | OpenCL           | `cl_mem`                                            |

The engine constructor in `dnnl.hpp` is to be extended with a new parameter:
``` cpp
struct engine {
    engine(engine::kind akind, dnnl::api aapi, size_t index);

    dnnl::api get_api() const;
};
```

As mentioned above, that would allow creating native C++ CPU engine for the
DPC++ library configuration. With such API we could easily enable plugin system
in the library: `api` indicates what plugin to load.

However, after some discussions we decided that:
1. We are not going to enable plugin system any time soon;
2. The `api` doesn't bring any value except for allowing creating native C++
   CPU engine for DPC++ build. In turn, we see no much value in having this
   ability. It seems that users would either use only DPC++ on CPU or only
   native C++. For the latter group (which is, e.g. some frameworks) we need to
   provide the corresponding configuration anyways, say `cpu_gomp_gpu_dpcpp`.

With that in mind we decided to not pursue adding the `api` enum and keep it
for future extensions, if we get the corresponding requests. The plugin system,
which seems to be much more valuable and possible in the future, requires much
more than just adding the `api` enum anyways, so there would be very little
value in adding it right away.


## 4. Discussion


### 4.1. Library Configurations

The proposed API is very similar to the current API (except for small technical
details). In particular, the API doesn't change anything with respect to the
way the library is shipped in the binary forms.

Given we received multiple asks to support Native C++ CPU along with DPC++ GPU,
we anticipate the need to ship the following configuration:
- `cpu_gomp`
- `cpu_iomp`
- `cpu_tbb`
- `cpu_iomp_gpu_dpcpp`
- `cpu_dpcpp_gpu_dpcpp`

Frameworks are also welcome to build the library from sources (just like they
do now), in which case an arbitrary configuration could be chosen.


### 4.2. Trade-offs

1. Multi-GPU systems. The API doesn't help with specifying the type of the GPU
   that will be picked when engine is created. For instance, if a system has
   both iGPU and dGPU, what `dnnl::engine(engine::kind::gpu, 0)` will pick?
   Internally we probably could create a priority list, e.g. make dGPU be more
   preferable than iGPU. That could be handy for the testing. Alternatively, we
   could just rely on `gpu_selector{}` mechanism that DPC++ has, and maybe ask
   to provide an environment control over the default selector to be able to
   control what oneDNN chooses. Finally, we could just make our tests more
   advanced to pick a proper GPU (via option, or just implicitly), and ask
   users to use the advanced API where our engine is created with SYCL device.


### 4.3. Incompatible Changes: v1.x and v2.x

1. **Native C++** API
   - Stream attributes are dropped as unused. Mostly affects TF thread pool
     integration, as it is unlikely anyone else uses stream attributes.

2. **ThreadPool** API
   - Same as above + the way to create a stream with thread pool attached
     happens by including `dnnl_threadpool.hpp` header file and calling
     ``` cpp
     dnnl::threadpool::create_stream(const engine &e, threadpool_iface *thread_pool);
     ```

3. **OpenCL** API
   - Same as bullet 1, and the way engine, stream, and memory are created with
     the corresponding CL objects: instead of constructors a free functions
     that live in a separate `dnnl_ocl.hpp` header file should be used.


### 4.4. Incompatible Changes: `dev-v2` (v2.0-betaX) and v2.x

Since `dev-v2` essentially based on v1.x the incompatible changes repeat the
one above.

1. **DPC++** API
   - Constructors with SYCL objects are migrated to `dnnl_sycl.hpp` into a free
     functions, like `dnnl::sycl::engine_create(dev, ctx)` with slightly
     different parameters.
   - The default memory kind switched from buffers to device USM.
   - There is no more control between buffers and USM via
     `DNNL_USE_SYCL_BUFFERS` macro. The control happens through passing the
     desired memory kind to memory creation: `dnnl::sycl::memory_create(...)`.
   - Primitive execution that has dependencies and returns a SYCL event is
     moved to a separate function `dnnl::sycl::primitive_execute(...)`.


### 4.5. Feedback from Frameworks

TBA.


---

EOD
