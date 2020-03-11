OpenCL Interoperability {#dev_guide_opencl_interoperability}
===============================================================

oneDNN uses the OpenCL runtime for GPU engines to interact with the GPU.
Users may need to use oneDNN with other code that uses OpenCL. For that
purpose, the library provides API extensions to interoperate with underlying
OpenCL objects.

The interoperability API is provided for two scenarios:
- Construction of oneDNN objects based on existing OpenCL objects
- Accessing OpenCL objects for existing oneDNN objects

The mapping between oneDNN and OpenCL objects is provided in the
following table:

| oneDNN object        | OpenCL object(s)                |
| :------------------- | :------------------------------ |
| Engine               | `cl_device_id` and `cl_context` |
| Stream               | `cl_command_queue`              |
| Memory               | `cl_mem`                        |

## C++ API Extensions for Interoperability with OpenCL

### API to Construct oneDNN Objects

| oneDNN object        | API to construct oneDNN object                                   |
| :------------------- | :--------------------------------------------------------------- |
| Engine               | [dnnl::engine(kind, ocl_dev, ocl_ctx)](@ref dnnl::engine)        |
| Stream               | [dnnl::stream(engine, ocl_queue)](@ref dnnl::stream)             |
| Memory               | [dnnl::memory(memory_desc, engine, ocl_mem)](@ref dnnl::memory)  |

@note oneDNN follows retain/release OpenCL semantics when using OpenCL
objects during construction. An OpenCL object is retained on construction and
released on destruction - that ensures that the OpenCL object will not be
destroyed while the oneDNN object stores a reference to it.

### API to Access OpenCL Objects

| oneDNN object        | API to access OpenCL object(s)                                     |
| :------------------- | :----------------------------------------------------------------- |
| Engine               | dnnl::engine::get_ocl_device() and dnnl::engine::get_ocl_context() |
| Stream               | dnnl::stream::get_ocl_command_queue()                              |
| Memory               | dnnl::memory::get_ocl_mem_object()                                 |

@note The access interfaces do not retain the OpenCL object. It is the user's
responsibility to retain the returned OpenCL object if necessary.

## C API Extensions for Interoperability with OpenCL

### API to Construct oneDNN Objects

| oneDNN object        | API to construct oneDNN object                                                         |
| :------------------- | :------------------------------------------------------------------------------------- |
| Engine               | [dnnl_engine_create_ocl(&engine, kind, ocl_dev, ocl_ctx)](@ref dnnl_engine_create_ocl) |
| Stream               | [dnnl_stream_create_ocl(&stream, engine, ocl_queue)](@ref dnnl_stream_create_ocl)      |
| Memory               | [dnnl_memory_create(&memory, memory_desc, engine, &ocl_mem)](@ref dnnl_memory_create)  |

### API to Access OpenCL Objects

| oneDNN object        | API to access OpenCL object(s)                                 |
| :------------------- | :------------------------------------------------------------- |
| Engine               | dnnl_engine_get_ocl_device() and dnnl_engine_get_ocl_context() |
| Stream               | dnnl_stream_get_ocl_command_queue()                            |
| Memory               | dnnl_memory_get_ocl_mem_object()                               |
