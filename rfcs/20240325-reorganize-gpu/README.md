# Reorganize the GPU Abstractions

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

Eeorganize th

Reorganize the GPU kernels and abstractions according the the following
schema: Vendor / Technology

With this schema the GPU directory will have subdirectories that
correspond to the vendors: `intel`, `nvidia`, `amd`, `generic`, etc.
Each of the subdirectories will have technology specific sub-subdirectories:
`sycl`, `ocl`, `jit`, etc.

Pros:
* The schema provides enough flexibility to enable new vendor and extend
the supported ones
* Clustering functionality and abstractions around vendors is very
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
├── cpu/ # CPU code
├── sycl/ # A common place for basic CPU and GPU SYCL abstractions, e.g. `sycl_cpu_engine_t`, `sycl_gpu_engine_t`, etc.
└── gpu/ # GPU code. Basic GPU code resides in the GPU directory directly, e.g. `gpu_engine_t`.
    ├── intel/ # Everything that is related to Intel
    │   ├── compute/ # Compute layer abstractions
    │   ├── ocl/ # OpenCL kernels and abstractions (very similar to the current state of `gpu/ocl` directory)
    │   ├── jit/ # JIT kernels/generators and abstractions (very similar to the current state of `gpu/jit` directory)
    │   ├── sycl/ # Intel related SYCL functionality/abstractions (e.g. SYCL kernels that use eSIMD, Intel specific extensions, etc)
    │   └── ...
    ├── nvidia/ # Everything that is related to NVIDIA
    │   ├── sycl # NVIDIA related functionality/abstractions (cuDNN/cuBLAS based and CUDA specific SYCL kernels)
    │   └── ...
    ├── amd/ # Everything that is related to AMD
    │   ├── sycl/ # AMD related functionality/abstractions (MIOpen/rocBLAS based and HIP specific SYCL kernels)
    │   └── ...
    └── generic/ # Everything that is related to generic kernels
        └── sycl/ # generic SYCL kernels and related abstractions

```

### Affected Basic Abstractions

The new schema will require moving a lot of parts of the library around
and while most of the changes are probably just an implementation detail
there are a few major changes that have to be described in this RFC.

#### Engine

Engine is the most heavily affected abstraction by the changes.
There is currently no `gpu_engine_t` class as `compute_engine_t` is used as
an Intel centric alternative of it. In order to have a common basic
abstraction at the GPU level the `gpu_engine_t` class will be introduced.

The current inheritance chains for SYCL and OpenCL GPU engines are
the following:
* SYCL Intel and generic: `engine_t` -> `compute_engine_t` -> `sycl_engine_base_t` -> `sycl_gpu_engine_t`
* SYCL NVIDIA: `engine_t` -> `compute_engine_t` -> `sycl_engine_base_t` -> `sycl_cuda_engine_t`
* SYCL AMD: `engine_t` -> `compute_engine_t` -> `sycl_engine_base_t` -> `sycl_hip_engine_t`
* OpenCL (only Intel): `engine_t` -> `compute_engine_t` -> `ocl_gpu_engine_t`

With the new schema they will be changed to:
* SYCL Generic: `engine_t` -> `gpu_engine_t` -> `gpu_generic_engine_t` -> `sycl_engine_base_t` -> `sycl_gpu_generic_engine_t`
* SYCL Intel: `engine_t` -> `gpu_engine_t` -> `gpu_intel_engine_t` -> `compute_engine_t` -> `sycl_engine_base_t` -> `sycl_gpu_intel_engine_t`
* SYCL NVIDIA: `engine_t` -> `gpu_engine_t` -> `gpu_nvidia_engine_t` -> `sycl_engine_base_t` -> `sycl_gpu_nvidia_engine_t`
* SYCL AMD: `engine_t` -> `gpu_engine_t` -> `gpu_amd_engine_t` -> `sycl_engine_base_t` -> `sycl_gpu_amd_engine_t`
* OpenCL: `engine_t` -> `gpu_engine_t` ->  `gpu_intel_engine_t` -> `compute_engine_t` -> `ocl_gpu_engine_t`
    * There is currently no plan to extend OpenCL support to other vendors
    therefore the name of the final class (`ocl_gpu_engine_t`) doesn't
    include the vendor's name.

#### Stream

The current inheritance chains for SYCL and OpenCL GPU streams are the following:
* SYCL Intel and genetic: `stream_t` -> `compute_stream_t` -> `sycl_stream_t`
* SYCL NVIDIA: `stream_t` -> `compute_stream_t` -> `sycl_stream_t` -> `sycl_cuda_stream_t`
* SYCL AMD: `stream_t` -> `compute_stream_t` -> `sycl_stream_t` -> `sycl_hip_stream_t`
* OpenCL: `stream_t` -> `compute_stream_t` -> `ocl_stream_t`

With the new schema they will be changed to:
The current inheritance chains for SYCL and OpenCL GPU streams are the following:
* SYCL Genetic: `stream_t` -> `gpu_stream_t` -> `gpu_generic_stream_t` -> `sycl_gpu_generic_stream_t`
* SYCL Intel: `stream_t` -> `gpu_stream_t` -> `gpu_intel_stream_t` -> `compute_stream_t` -> `sycl_gpu_intel_stream_t`
* SYCL NVIDIA: `stream_t` -> `gpu_stream_t` -> `gpu_nvidia_stream_t` -> `sycl_gpu_nvidia_stream_t`
* SYCL AMD: `stream_t` -> `gpu_stream_t` -> `gpu_amd_stream_t` -> `sycl_gpu_amd_stream_t`
* OpenCL: `stream_t` -> `gpu_stream_t` -> `gpu_intel_stream_t` -> `compute_stream_t` -> `ocl_stream_t`
