oneDNN GPU Implementation
=========================

The GPU source code is organized around vendors. The code specific to a vendor
is independent of the code specific to other vendors. The only exception is the
generic vendor that can be enabled in combination with any other vendor.
* The common GPU code resides in the `gpu/` directory
* The vendor-specific code resides in the `gpu/<vendor>` sub-directories
* The `gpu/<vendor>` sub-directories may have kernel language sub-sub-directories
`gpu/<vendor>/<kernel language>` to accommodate the corresponding code

## Directory structure

```
gpu
├── intel/               # Intel vendor-specific code
│   ├── compute/         # Compute layer abstractions
│   ├── ocl/             # OpenCL-specific code
│   ├── jit/             # JIT(nGEN)-specific code
│   ├── sycl/            # SYCL-specific code
│   │   ├── l0/          # Level-Zero backend-specific code
│   │   └── ...
│   └── ...
├── nvidia/              # NVIDIA vendor-specific code
├── amd/                 # AMD vendor-specific code
└── generic/             # Generic (vendor agnostic)-specific code
    ├── sycl/            # Generic SYCL kernels
    ├── ref_concat.hpp   # Generic kernel language agnostic implementations
    └── ...
```

NOTE: There is also XPU-specific code that resides outside of the GPU directory but
is used by the GPU-specific code.

See additional information on the XPU code [oneDNN XPU Implementation](../xpu/README.md).

## Vendors

Currently, oneDNN supports the following GPU vendors:
* INTEL: SYCL and OpenCL runtimes
* NVIDIA, AMD: SYCL runtime

The vendor-specific code can easily access the XPU or common GPU code, but the
opposite should be limited as much as possible. However, sometimes it is
absolutely necessary for the XPU or common GPU code to access the vendor-specific
one. For example, the lists of implementations that live in `gpu/*_list.cpp`
should conditionally include the specific implementations for the corresponding
vendor. The macro `DNNL_GPU_VENDOR` can be defined to dispatch vendor-specific
code at compile time. The following values are
possible:
* `DNNL_VENDOR_NONE`
* `DNNL_VENDOR_INTEL`
* `DNNL_VENDOR_NVIDIA`
* `DNNL_VENDOR_AMD`

The macros are defined in `dnnl_config.h`.

Usage example:

``` cpp
#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/foo.hpp"
#elif DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/nvidia/foo.hpp"
#endif

int foo() {
#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
    return gpu::intel::foo();
#elif DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
    return gpu::nvidia::foo();
#endif
}
```

Additionally, there is `DNNL_GPU_<VENDOR>_ONLY` macro that expands to its
parameters only for the corresponding vendors. Hence, the following
code has the same behavior as the example above:

``` cpp
#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/foo.hpp"
#elif DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/nvidia/foo.hpp"
#endif

int foo() {
DNNL_GPU_INTEL_ONLY(return gpu::intel::foo());
DNNL_GPU_NVIDIA_ONLY(return gpu::nvidia::foo());
}
```
