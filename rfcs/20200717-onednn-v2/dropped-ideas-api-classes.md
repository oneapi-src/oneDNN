# Dropped Ideas: API Classes

The proposed library API could be found in
[history/api-simplified-rev0](history/api-simplified-rev0/) directory.

## 1. API Classes

It was suggested to bring a new enum `dnnl::api` (or, maybe better
`dnnl::api_class`, and maybe make it `engine` sub-class), which names the API
classes we support. For native C/C++ API the enum contains `api::c`. Exactly
this API class is fully defined in `dnnl.hpp`, and only engine kind supported
by this API is CPU. It was also suggested that for DPC++ library configuration
(`dnnl_cpu_dpcpp_gpu_dpcpp`) it is allowed to create engine with native C API.
This will allow supporting native C++ applications as well as DPC++
applications within a single configuration.

The list of `dnnl::api` values:

| `dnnl::api`       | API to work with | Memory buffer representation                        |
| :--               | :--              | :--                                                 |
| `api::c`          | Native C/C++     | `void *`                                            |
| `api::sycl`       | DPC++            | Further dispatch based on `dnnl::sycl::memory_kind` |
| `api::ocl`        | OpenCL           | `cl_mem`                                            |

The API class is closely related to the engine. Engine class would then have a
new query: `dnnl::api engine::get_api() const;`. This is also the reason why
`api` should probably belong to the `engine` class.

## 2. Engine

There would be 3 ways to create an engine:

``` cpp
// =============
// dnnl_$api.hpp
// =============

// 1. The most advance, API-specific way of creating an engine that accepts
// API/RT-specific input arguments. Example below for DPC++ (similar of OCL):
dnnl::engine dnnl::sycl::engine_create(engine::kind, memory_kind,
        const cl::sycl::device &dev, const cl::sycl::context &ctx);
// The memory_kind is discussed below.

// =============
// dnnl.hpp
// =============

// 2. New way of creating engine. If `api` != `api::c`, the extra parameters
// appeared in `dnnl_$api.hpp` will used some default parameters, that should
// be documented on the corresponding API-related page.
auto eng = engine(engine::kind, dnnl::api, index);

// Example:
// engine(engine::kind::cpu, dnnl::api::sycl, 0) will use
// `memory_kind::usm_device` and create device and context underneath.

// 3. oneDNN v1.x compatible way
auto eng = engine(engine::kind, index); // same as engine(engine::kind, api::c, index);
```

DPC++ engine will have (mostly for completeness) one extra constructor:
``` cpp
// =============
// dnnl_sycl.hpp
// =============

dnnl::engine dnnl::sycl::engine_create(engine::kind, memory_kind = memory_kind::usm_device);
```

Which will create device and context underneath. Exactly this version will be
used by `dnnl::engine(engine::kind::xpu, api::sycl, index);` that lives in
`dnnl.hpp`.

The proposed API breaks backwards compatibility with OpenCL.
``` cpp
auto eng = engine(engine::kind::gpu, 0); // which is the same as
auto eng = engine(engine::kind::gpu, api::c, 0); // will always fail
```

One could create OpenCL engine in two ways:
``` cpp
auto eng = engine(engine::kind::gpu, api::ocl, 0); // dnnl.hpp
auto eng = dnnl::ocl::engine_create(engine::kind::gpu, ocl_dev, ocl_ctx); // dnnl_ocl.hpp
```

## Discussion: Default For Engine Constructor w/o API

As it is mentioned above the RFC suggests making
`engine(engine::kind::xpu, index)` constructor be an equivalent of
`engine(engine::kind::xpu, api::c, index);`, which in particularly means:
- No matter what, `engine(engine::kind::gpu, index)` will always fail
- ThreadPool, OpenCL and DPC++ users will always have to explicitly pass the
  desired API to the constructor.

While this behavior is very explicit, it might be inconvenient for non-C++-API
users.

Eugene suggests the alternative approach: make the API-agnostic engine
constructor to pick the first suitable version that is satisfies the
requirements. For instance, if the library is built with OpenCL support,
the `engine(engine::kind::gpu, 0)` will create OpenCL engine. User can verify
that by querying the `api` with `engine::get_api()` method.

There are 3 obvious benefits:
1. The API is "more" backwards compatible with v1.x version
2. For the well known configurations (e.g. `cpu_omp_gpu_ocl`) a user gets what
   they want by default (simpler API).
3. It would be very convenient for the testing, as we don't need to care much
   about properly picked runtime (anyways we don't have multiple API at the
   same time in our current building system).

The drawbacks:
1. If the wrong configuration is loaded, the behavior would be unexpected
   (could be mitigated by user, if desired, by explicitly setting the API in
   the constructor, i.e. `engine(engine::kind::xpu, desired_api, 0);` instead
   of `engine(engine::kind::xpu, 0)`).
2. Less explicit behavior wrt multiple APIs if supported.
   - For instance, if we support both C++ and DPC++ APIs for CPU (say, for
     `cpu_dpcpp_gpu_dpcpp` configuration), it is unclear whether DPC++ is
     prioritized of C++ or vice versa.
3. Probably, this approach doesn't scale well if we are to support more APIs.
   Though we probably envision any beyond what we currently have.

If we go this route an open question if we should enumerate one engine kind
(say, CPU) with different API with this API-agnostic constructor, or we should
just allow creating one. I.e.:

``` cpp
engine(engine::kind::cpu, 0) --> engine(engine::kind::xpu, api::dpcpp, 0)
engine(engine::kind::cpu, 1) --> engine(engine::kind::xpu, api::c, 0)

// vs

engine(engine::kind::cpu, 0) --> engine(engine::kind::xpu, api::dpcpp, 0)
engine(engine::kind::cpu, 1) --> error, not such engine
// to create CPU engine with C, one needs explicitly call:
// engine(engine::kind::xpu, api::c, 0)
```

One of the concerns about the alternative approach is that when people use
oneDNN they typically know what API they are using (as they use all the
interoperability features). So, with that, I think, the clarity / explicitness
of choosing the API outweighs the usefulness of the default dispatching.

## Discussion: Alternative to API Enum Class -- Runtime Enum Class

Eugene suggested to avoid using overloaded word API (or API class) and reuse
`runtime_kind` that we use internally in the library:

``` cpp
enum class runtime_kind {
    native = 0x1000,
    ocl = 0x2000,
    sycl = 0x4000,
    threadpool = 0x8000,
    seq = native | 0x1,
    omp = native | 0x2,
    tbb = native | 0x4,
};
```

After a second thought I think this could be a possible and actually quite nice
solution. The sketch of the API and its semantics is shown below.

``` cpp
namespace dnnl {

enum class runtime_kind {
    // interfaces
    c = 0x1000, // instead of native
    ocl = 0x2000,
    sycl = 0x4000,

    // extra runtime specifications, not a part of specification
    seq = 0x1,
    omp = 0x2,
    tbb = 0x4,
    threadpool = 0x8,
};

struct engine {
    // rkind must specify the interface {c, ocl, sycl}, and could or could not
    // specify extra runtime specification. The portable code should not make
    // any assumption on the extra runtime specifications.
    engine(engine::kind ekind, runtime_kind rkind, ...);

    // returns the runtime kind. For `rkind` that was used at engine creation,
    // it is guaranteed that `get_kind() & rkind == rkind`.
    // However, the returned runtime_kind could contain additional information,
    // e.g. for engine created with `runtime_kind::c`, the returned value
    // could be `runtime_kind::c | runtime_kind::omp` (in future, even
    // `runtime_kind::c | runtime_kind::omp | runtime_kind::omp_rt_intel`).
    runtime_kind get_runtime_kind() const;
};

} // namespace dnnl
```

Consider few use cases:

1. Portable way to create CPU engine for C/C++ API
   ``` cpp
   auto eng = engine(engine::kind::cpu, runtime_kind::c, 0);
   assert(eng.get_runtime_kind() & runtime_kind::c == runtime_kind::c);
   // not portable code, but could be handy
   if (eng.get_runtime_kind() & runtime_kind::omp) {
       // omp specific code
   }
   ```

2. Non-portable way to create an engine that will use OMP RT. The creation may
   fail if the library was built with TBB:
   ``` cpp
   // a strong desire to use OMP, ok to fail...
   auto eng = engine(engine::kind::cpu, runtime_kind::c | runtime_kind::omp, 0);
   ```

3. Could be possible now if we want to support that:
   ``` cpp
   // even though the configuration is dnnl_cpu_omp we could support external
   // thread pool, if we really want to.
   auto eng = engine(engine::kind::cpu,
           runtime_kind::c | runtime_kind::threadpool, 0);
    ```

4. There would be a way to use sequential oneDNN even if it comes with OMP:
   ``` cpp
   // configuration cpu_iomp
   // a user wants sequential version
   auto eng = engine(engine::kind::cpu, runtime_kind::c | runtime_kind::seq, 0);
   ```

5. Get what is underlying parallelization runtime for DPC++:
   ``` cpp
   auto eng = engine(engine::kind::cpu, runtime_kind::sycl, 0);

   print(eng.get_runtime_kind());
   // cpu_dpcpp_tbb will print: runtime_kind::sycl | runtime_kind::tbb
   // cpu_dpcpp_omp will print: runtime_kind::sycl | runtime_kind::omp
   ```

I think as of today, there is little value in exposing these extra flags
(compared to just exposing the _API_). However, the approach gives the
following advantages:
1. We could (at least in theory) support more configurations in a single
   bundle, like forcing:
   - sequential version for OMP or TBB configurations,
   - having threadpool for OMP or TBB configurations;
2. Use can query extra information if that would be of any help. For instance,
   even though the API is DPC++, a user can find out that the runtime used for
   parallelization is OMP.
3. This approach will allow us support plugin model, where depending on the
   chosen runtime flags oneDNN common (core) library will load the proper
   engine plugin.

In my opinion, however, implementing all this will be an overkill, and probably
not feasible at this moment (for instance, I am not sure how easy it would be
to support sequential mode for the library built with OMP). With that, I still
vote for having an `api` (maybe better api_class) enum class, and extend the
when we decide to proceed with plugin model.

- The intermediate solution will be to use `runtime_kind` name and implement
  only meaningful part, e.g.:
  ``` cpp
  enum class runtime_kind {
      // interfaces
      c = 0x1000, // instead of native
      ocl = 0x2000,
      sycl = 0x4000,

      // extra runtime specifications, not a part of specification
      threadpool = 0x8, // should it be an interface actually?..
  };
  ```
  I.e. do not bring OMP, TBB, SEQ to the list, and extend it when time comes.
  The downside is that with this definition users could be confused with our
  cmake options `-DDNNL_CPU_RUNTIME=OMP` and `-DDNNL_GPU_RUNTIME=DPCPP`.


---

EOD
