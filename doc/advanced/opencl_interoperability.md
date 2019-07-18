OpenCL Interoperability {#dev_guide_opencl_interoperability}
===============================================================

Intel MKL-DNN uses the OpenCL runtime for GPU engines to interact with the GPU.
Users may need to use Intel MKL-DNN with other code that uses OpenCL. For that
purpose, the library provides API extensions to interoperate with underlying
OpenCL objects.

The interoperability API is provided for two scenarios:
- Construction of Intel MKL-DNN objects based on existing OpenCL objects
- Accessing OpenCL objects for existing Intel MKL-DNN objects

The mapping between Intel MKL-DNN and OpenCL objects is provided in the
following table:

| Intel MKL-DNN object | OpenCL object(s)                |
| :------------------- | :------------------------------ |
| Engine               | `cl_device_id` and `cl_context` |
| Stream               | `cl_command_queue`              |
| Memory               | `cl_mem`                        |

## C++ API Extensions for Interoperability with OpenCL

### API to Construct Intel MKL-DNN Objects

| Intel MKL-DNN object | API to construct Intel MKL-DNN object                                |
| :------------------- | :------------------------------------------------------------------- |
| Engine               | [mkldnn::engine(kind, ocl_dev, ocl_ctx)](@ref mkldnn::engine)        |
| Stream               | [mkldnn::stream(engine, ocl_queue)](@ref mkldnn::stream)             |
| Memory               | [mkldnn::memory(memory_desc, engine, &ocl_mem)](@ref mkldnn::memory) |

@note Intel MKL-DNN follows retain/release OpenCL semantics when using OpenCL
objects during construction. An OpenCL object is retained on construction and
released on destruction - that ensures that the OpenCL object will not be
destroyed while the Intel MKL-DNN object stores a reference to it.

### API to Access OpenCL Objects

| Intel MKL-DNN object | API to access OpenCL object(s)                                         |
| :------------------- | :--------------------------------------------------------------------- |
| Engine               | mkldnn::engine::get_ocl_device() and mkldnn::engine::get_ocl_context() |
| Stream               | mkldnn::stream::get_ocl_command_queue()                                |
| Memory               | mkldnn::memory::get_ocl_mem_object()                                   |

@note The access interfaces do not retain the OpenCL object. It is the user's
responsibility to retain the returned OpenCL object if necessary.

## C API Extensions for Interoperability with OpenCL

### API to Construct Intel MKL-DNN Objects

| Intel MKL-DNN object | API to construct Intel MKL-DNN object                                                      |
| :------------------- | :----------------------------------------------------------------------------------------- |
| Engine               | [mkldnn_engine_create_ocl(&engine, kind, ocl_dev, ocl_ctx)](@ref mkldnn_engine_create_ocl) |
| Stream               | [mkldnn_stream_create_ocl(&stream, engine, ocl_queue)](@ref mkldnn_stream_create_ocl)      |
| Memory               | [mkldnn_memory_create(&memory, memory_desc, engine, &ocl_mem)](@ref mkldnn_memory_create)  |

### API to Access OpenCL Objects

| Intel MKL-DNN object | API to access OpenCL object(s)                                     |
| :------------------- | :----------------------------------------------------------------- |
| Engine               | mkldnn_engine_get_ocl_device() and mkldnn_engine_get_ocl_context() |
| Stream               | mkldnn_stream_get_ocl_command_queue()                              |
| Memory               | mkldnn_memory_get_ocl_mem_object()                                 |
