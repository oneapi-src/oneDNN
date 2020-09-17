# oneDNN v2.0 Changes

The objective of this RFC is to discuss the changes in the upcoming oneDNN
v2.0, which will introduce the support for DPC++ programming model.
The plan is to merge
[`dev-v2`](https://github.com/oneapi-src/oneDNN/tree/dev-v2) branch into
[`master`](https://github.com/oneapi-src/oneDNN/tree/master). The exact date
is TBD, but would probably happen around October 2020.

One of the goals is to make v2.0 API backwards compatible with v1.x as much as
possible, ideally not breaking the existing code at all. There is no such goal
with respect to the backwards compatibility with `dev-v2` branch though
(`2021.1-betaXX` releases).

## 1. Executive Summary Of The Changes

1. The header files are split into runtime-agnostic and runtime-specific
   versions. The interoperability API with RT objects is only possible through
   runtime-specific header files. The changes are **incompatible** with the
   existing integration of Thread Pool, OpenCL, and DPC++.

   | Headers                                             | API For                | Namespace (applicable for C++ header files only) |
   | :--                                                 | :--                    | :--                                              |
   | `dnnl_types.h`, `dnnl.h`, `dnnl.hpp`                | Common and RT-agnostic | `dnnl`                                           |
   | `dnnl_sycl_types.h`, `dnnl_sycl.h`, `dnnl_sycl.hpp` | SYCL-specific          | `dnnl::sycl_interop`                             |
   | `dnnl_ocl.h`, `dnnl_ocl.hpp`                        | OpenCL specific        | `dnnl::ocl_interop`                              |
   | `dnnl_threadpool.h`, `dnnl_threadpool.hpp`          | Thread pool specific   | `dnnl::threadpool_interop`                       |

2. Stream attributes are removed. This should affect only the integration with
   Thread Pool.

3. Stream flag `stream::flags::default_order` is removed.

4. Some simplifications for RT-specific object constructors. For instance,
   there is no need to pass `engine::kind` to engine constructor that takes
   OpenCL device as an argument.

5. For DPC++ users:
   - Default buffer-based API is replaced with device USM;
   - Memory kind (USM or buffers) is now controlled using memory constructor
     and not `DNNL_USE_SYCL_BUFFERS`;

No changes are expected to the distribution model: there will be still many
configurations. We anticipate the need to add one more configuration:
`cpu_omp_gpu_dpcpp`.

### 1.1. API Preview

On can make a quick comparison of the current (v1.x and `dev-v2`) API and the
proposed one by checking the [api-simplified](api-simplified/) directory.

## 2. Discussion

For simplicity, the discussions on different aspects of the changes are split
in the following files:

1. [General v2 API](general-v2-api.md)
   - The most interesting part the describes the changes to the general API in
     details.
2. [USM and Buffer Support](usm-and-buffer-support.md)
   - Options and challenges in supporting DPC++ USM and Buffers.
3. [Changing `stream::flags::default_order` Behavior](changing-stream-flags-default-order-behavior.md)
   - Just as the name says.

## 3. Dropped Ideas

The discussion on the API change took quite some time. During it we considered
multiple options, reflected on them, and changed our decisions. The following
documents discuss some of the options that we decided not to proceed with:

1. [Dropped Ideas: API Classes](dropped-ideas-api-classes.md)
   - Initially we wanted to introduce `api` enum class to the API, to be able
     to choose between different runtimes within a fixed configuration. For
     instance, for DPC++ CPU configuration one could still create an enigne
     with Native C++ API. However, after quite some discussions we decided this
     idea brings very small value and should wait till better times (e.g. when
     oneDNN decides to enalbe plugin system, in which case a user needs a way
     to specify the runtime during engine construction).

2. [API Mockups](history/api-mockups/)
   - Mostly made by Roma (@rsdubtso). He considered several different
     approaches how organize the interoperability API (e.g. oneDNN <-> SYCL).
     The option 1 is the most simple one and essentially used in this RFC.
     Options 2 - 4, are probably more C++-like, but seem overcomplicated and
     less flexible for future (and even current) changes. For instance,
     `dnnl::sycl::memory_kind` memory parameter doesn't interplay nicely with
     `dnnl::interop::memory` structure. However, it is worth admitting that
     the [option 3a](history/api-mockups/option3a/) enforces different API/RT
     to be quite consistent.

## 4. Updates To This RFC

- 20/08/06. After the discussion between Eugene (@echeresh) and Evarist
  (@emfomenk), the RFC changed as follows:
  - Decided to disallow supporting multiple runtimes for one library
    configuration. Specifically, before RFC suggested to support native (C++)
    CPU along with DPC++ CPU runtime for `cpu_dpcpp_gpu_dpcpp`. The rationale
    is that the usefulness of this feature is questionable at this moment,
    given that we don't support plugin model and there are very low chances
    people would like to mix two runtimes for CPU.
  - As a consequence, `api` enum is dropped altogether -- there won't be any
    benefit of having it exposed in the API. That essentially means that the
    API corresponds to the configuration the library was built with. This is
    exactly what oneDNN v1.x and `dev-v2` were about.
  - This also solves the problem with the "default" api/runtime in engine
    constructor and with the name of api/runtime enum too.
  - The previous definition of the API where there was an `api` enum is
    documented in [Dropped Ideas: API Classes](dropped-ideas-api-classes.md).

- 20/08/05. After a discussion with Eugene (@echeresh), Mourad (@mgouicem),
  Denis (@densamoilov), Igor (@igorsafo), and Evarist (@emfomenk), the RFC
  changed as follows:
  - The engine constructor without `api` parameter was default to the runtime
    according the priorities instead of always assuming `api::c`. For DPC++
    configuration with native C++ CPU support, the suggestion was to default to
    `api::c` and not `api::sycl` for backward compatibility reasons. This
    approach allows creating GPU engine with `engine(kind::gpu, 0)`, as before
    RFC prohibited this (because the `api` was assumed `c`).
  - Thread-pool was suggested to be dropped from `api` enum as it doesn't bring
    any value.
  - There was a discussion about name for `api` enum: whether it should be
    named `api` or `runtime`.
    - The name `runtime` better reflects the reality, and has future
      extensibility if oneDNN will switch to plug-in based system. Runtime
      also is primary, while API is secondary (depends on runtime).
    - The name `api` with less values seemed more user-friendly.
    - We didn't come to a conclusion, those the majority leaned towards
      `api` mostly due to its simplicity from user's point of view.
  - Briefly discussed removing `memory_kind` from engine creation to memory
    creation, allowing mixing different kinds of memory for a single primitive.
    Agreed on proceeding this way. This simplifies the API, its concepts, it is
    more flexible, and in future, if buffers are de-prioritized / dropped,
    there would be minimum changes to the API (essentially 0).

- 20/09/09. After we discovered the issue with namespace clashing, we renamed
  `dnnl::sycl` (and other) namespaces to `dnnl::sycl_interop`. Along with that
  we also simplified the names of the interoperability functions. For instance,
  instead of `dnnl::sycl_interop::memory_get_memory_kind()` the function is now
  called `dnnl::sycl_interop::get_memory_kind()` as subject can be guessed by
  the first argument name.

- 20/09/17. Remove `usm_device` and `usm_shared` in favor of simple `usm` for
  `sycl_interop::memory_kind`. Justification: simpler API and in most of the
  cases the type of USM is not that important. For the cases where the type of
  USM is important users could use `sycl::get_pointer_type()`.


---

EOD
