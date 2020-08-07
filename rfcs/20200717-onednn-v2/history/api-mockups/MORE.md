# Additions To The Main [README.md](README.md)

## 1. Core API: Universal vs Narrow

> **Assumption** We move the RT-specific interop API into separate headers,
>  e.g. `dnnl_sycl.hpp`, `dnnl_ocl.h`.

The RT-agnostic API is called Core API. Essentially, this is `dnnl.h{,pp}` and
`dnnl_types.h`.

### Question

Should interop API interact with the library through the C Core API (by using
type-erased generic-enough functions) or should directly go to the library?

#### Option 1. Interop API (and C++ Core API) goes through C Core API

``` cpp
// dnnl_sycl.hpp
#include "dnnl.h" // for dnnl_memory_create_generic()
namespace dnnl {
memory make_memory(dnnl::engine e, cl::sycl::buffer buf) {
    unsigned flags = dnnl_sycl_memory_kind_usm;
    dnnl_memory_create_generic(e.get(), flags, (void*)&buf);
    ...
    return mem;
}
} // namespace dnnl
```

Advantages:
1. If a "wrong" `libdnnl.so` was loaded, the application won't crash, but
   gracefully reports an issue on creating an engine (for that case when
   initially an application was built against oneDNN w/ SYCL, but then was run
   with oneDNN w/o SYCL).
   - This advantage, however, could be seen as disadvantage too, as linking or
     runtime issue could help understanding earlier that something goes wrong.
2. Assuming the C Core API is "stable" (i.e. doesn't often change) this model
   more or less guaranties that the final user application will use only one
   oneDNN library (even if several copies are loaded).
3. Minor. Less chance to break backwards compatibility, as oneDNN C++ API
   doesn't directly use the library, but through C API: the ABI surface is
   tangible.
4. If we decide to implement plugin model in future, this approach, in theory,
   would not require users to link against plugins themselves, as the whole
   communication will happen through the main API (code library).

#### Option 2. Interop API can directly go to the library

   ``` cpp
// dnnl_sycl.hpp
namespace dnnl {
memory make_memory(dnnl::engine e, cl::sycl::buffer buf);
} // namespace dnnl

// src/sycl/cxxapi/memory.cpp
dnnl::memory dnnl::make_memory(dnnl::engine e, cl::sycl::buffer buf) {
    ...
}
```

Advantages:
1. See the 1st advantage of the Option 1.
2. Future proof: there is more flexibility to add more RT-specific stuff in
   future as there is no need to adjust the Core API.


## More Questions

1. We have `dnnl_cpu_omp` configuration and code that works with it, running
   with the assumption that default stream is in-order (even more, actually,
   **blocking**) and `data_handle` for memory is just a pointer. We now replace
   the library with `dnnl_cpu_dpcpp`. Do we expect the same source code to
   continue working? Without rebuilding? With rebuilding?
   - If we do, that might make us require from user to create SYCL engine not
     as `engine(kind::cpu, 0)` as it is now, but with something like
     `engine(api_kind::sycl, kind::cpu, 0)`. This is, however, doesn't aligned
     well with GPU and OCL...
   - This also will make the testing more complex, as we need to specify the
     RT. We may have something line `DNNL_OVERRIDE_DEFAULT_RT={CPP,OCL,SYCL}`
     and set this environment variable during our testing, but still.


---

EOD
