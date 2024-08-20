# Enable Generic GPU Vendor

## Motivation

oneDNN project is part of the Unified Acceleration Foundation (UXLF) ecosystem,
and as part of the commitment to the UXLF strategy oneDNN will introduce primitive
implementations that can work with multiple different GPU and accelerator vendors.


## Proposal

In this RFC the primitive implementations will be called "generic".
The generic implementations (kernels) will be implemented in SYCL. oneDNN already has a
set of such kernels that have extensive coverage and are already enabled for
NVIDIA and AMD vendors. This proposal is about enabling the kernels for a "generic"
vendor that could be used to run the kernels on different hardware.

SYCL specification defines the following device types:
* CPU
* GPU
* Accelerator

Each of the aforementioned device types is independent from the others and therefore
SYCL devices have to be created specifically for particular types.

oneDNN supports two engine kinds: CPU and GPU. According to the UXFL strategy the
generic kernels are expected to run on different GPUs and accelerators.

The proposal is to enable the new accelerator device type with the existing GPU
Runtime.

A new engine kind `accelerator` will be introduced to enable runtime agnostic API
for creating an engine of that kind.
```cpp
dnnl::engine engine(dnnl::engine::kind::accelerator, 0);
```
Below is a matrix of supported GPU and accelerator engine kinds for different
GPU vendors.

| GPU Vendor \ Engine kind | GPU | Accelerator |
|--------------------------|-----|-------------|
|         Intel            |  +  |      -      |
|         NVIDIA           |  +  |      -      |
|         AMD              |  +  |      -      |
|         Generic          |  +  |      +      |

oneDNN library that is built with `DNNL_GPU_RUNTIME=SYCL` and `DNNL_GPU_VENDOR=GENERIC`
options will support multiple GPUs and accelerators.

Pros:
* The only change in the API is adding a new engine kind: `accelerator`.
* There is no need to complicate the configuration management by introducing
a new runtime for the new engine kind.
* Generic vendor can be used in a combination with any CPU runtime.

Cons:
* The GPU runtime may be a confusing name to cover accelerators. We may consider to
rename it if that becomes a problem.
* Minor semantic change. Currently, there are two runtimes: CPU and GPU that map
to the corresponding engine kinds, which changes after introducing the accelerator
 engine kind.


### Summary

* The `DNNL_GPU_RUNTIME` name can be changed in the future to something more
appropriate, e.g. `DNNL_ACC_RUNTIME`, where `ACC` would be short for `ACCELERATOR`.
That could be a better name as GPU could also be considered an accelerator. But
given that it is just a beginning of the journey the recommendation is to keep the
current name and just document this peculiarity for the time being.
* When a new vendor decides to add vendor specific kernels for a GPU or accelerator
or both they will need to document what engine kinds the vendor supports.
* The generic SYCL kernels can be used for any device type, including CPU so it may
seem to be a good idea to avoid associating them exclusively with GPUs and
accelerators. However, the current strategy for enabling generic kernels for CPU is to
use reference CPU kernels implemented in C++. It is also possible to use them within
the SYCL programming model because oneDNN supports SYCL runtime for CPU. All things considered, it doesn't make much sense to enable generic SYCL (or any heterogenous)
kernels for CPU.