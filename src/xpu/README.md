oneDNN XPU Implementation
=========================

The XPU source code is organized around heterogeneous runtimes. The XPU code is
vendor agnostic.
* The common XPU code resides in the `xpu/` directory
* The runtime-specific code resides in the `xpu/<runtime>` sub-directories

## Directory structure
```
xpu/       # Vendor agnostic code for heterogeneous runtimes
├── sycl/  # Vendor agnostic code for SYCL runtime
├── ocl/   # Vendor agnostic code for OpenCL runtime
└── ...
```

The vendor-specific code can easily access the XPU code, but the
opposite should be limited as much as possible. However, sometimes it is
absolutely necessary for the XPU code to access the vendor-specific one. For
example, `xpu/<runtime>/engine_factory.hpp` should conditionally include the
vendor-specific engines. The macro `DNNL_GPU_VENDOR` can be defined to dispatch vendor-specific
code at compile time.

See more details on the conditional dispatching in [oneDNN GPU Implementation](../gpu/README.md).
