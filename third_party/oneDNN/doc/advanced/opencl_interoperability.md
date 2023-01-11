OpenCL Interoperability {#dev_guide_opencl_interoperability}
============================================================

> [API Reference](@ref dnnl_api_ocl_interop)

## Overview

oneDNN uses the OpenCL runtime for GPU engines to interact with the GPU. Users
may need to use oneDNN with other code that uses OpenCL. For that purpose, the
library provides API extensions to interoperate with underlying OpenCL objects.
This interoperability API is defined in the `dnnl_ocl.hpp` header.

The interoperability API is provided for two scenarios:
- Construction of oneDNN objects based on existing OpenCL objects
- Accessing OpenCL objects for existing oneDNN objects

The mapping between oneDNN and OpenCL objects is provided in the following
table:

| oneDNN object        | OpenCL object(s)                    |
| :------------------- | :---------------------------------- |
| Engine               | `cl_device_id` and `cl_context`     |
| Stream               | `cl_command_queue`                  |
| Memory (Buffer-based)| `cl_mem`                            |
| Memory (USM-based)   | Unified Shared Memory (USM) pointer |

The table below summarizes how to construct oneDNN objects based on OpenCL
objects and how to query underlying OpenCL objects for existing oneDNN objects.

| oneDNN object        | API to construct oneDNN object                                                                          | API to access OpenCL object(s)                                                                    |
| :------------------- | :------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------ |
| Engine               | dnnl::ocl_interop::make_engine(cl_device_id, cl_context)                                                | dnnl::ocl_interop::get_device(const engine &) <br> dnnl::ocl_interop::get_context(const engine &) |
| Stream               | dnnl::ocl_interop::make_stream(const engine &, cl_command_queue)                                        | dnnl::ocl_interop::get_command_queue(const stream &)                                              |
| Memory (Buffer-based)| dnnl::memory(const memory::desc &, const engine &, cl_mem)                                              | dnnl::ocl_interop::get_mem_object(const memory &)                                                 |
| Memory (USM-based)   | dnnl::ocl_interop::make_memory(const memory::desc &, const engine &, ocl_interop::memory_kind, void \*) | dnnl::memory::get_data_handle()                                                                   |

## OpenCL Buffers and USM Interfaces for Memory Objects

The memory model in OpenCL is based on OpenCL buffers. [Intel extension](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/cl_intel_unified_shared_memory.asciidoc)
further extends the programming model with a Unified Shared Memory (USM)
alternative, which provides the ability to allocate and use memory in a uniform
way on host and OpenCL devices.

oneDNN supports both buffer and USM memory models. The buffer model is
the default. The USM model requires using the interoperability API.

To construct a oneDNN memory object, use one of the following interfaces:

- dnnl::ocl_interop::make_memory(const memory::desc &, const engine &, ocl_interop::memory_kind kind, void \*handle)

    Constructs a USM-based or buffer-based memory object depending on memory
    allocation kind `kind`. The `handle` could be one of special values
    #DNNL_MEMORY_ALLOCATE or #DNNL_MEMORY_NONE, or it could be a user-provided
    USM pointer. The latter works only when `kind` is dnnl::ocl_interop::memory_kind::usm.

- dnnl::memory(const memory::desc &, const engine &, void \*)

    Constructs a buffer-based memory object. The call is equivalent to calling the
    function above with with `kind` equal to dnnl::ocl_interop::memory_kind::buffer.

- dnnl::ocl_interop::make_memory(const memory::desc &, const engine &, cl_mem)

    Constructs a buffer-based memory object based on a user-provided OpenCL
    buffer.

To identify whether a memory object is USM-based or buffer-based,
dnnl::ocl_interop::get_memory_kind() query can be used.

@note oneDNN follows retain/release OpenCL semantics when using OpenCL objects
during construction. An OpenCL object is retained on construction and released
on destruction. This ensures that the OpenCL object will not be destroyed while
the oneDNN object stores a reference to it.

@note The access interfaces do not retain the OpenCL object. It is the user's
responsibility to retain the returned OpenCL object if necessary.

@note USM memory doesn't support retain/release OpenCL semantics. When
constructing a oneDNN memory object using a user-provided USM pointer oneDNN
doesn't own the provided memory. It's user's responsibility to manage its
lifetime.
