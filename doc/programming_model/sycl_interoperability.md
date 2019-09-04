SYCL Interoperability {#dev_guide_sycl_interoperability}
===============================================================

DNNL may use the SYCL runtime for CPU and GPU engines to interact with
the hardware. Users may need to use DNNL with other code that uses
SYCL. For that purpose, the library provides API extensions to interoperate
with underlying SYCL objects.

One of the possible scenarios is executing a SYCL kernel for a custom
operation not provided by DNNL. In this case, the library provides
all necessary API to "seamlessly" submit a kernel, sharing the execution
context with DNNL: using the same device and queue.

The interoperability API is provided for two scenarios:
- Construction of DNNL objects based on existing SYCL objects
- Accessing SYCL objects for existing DNNL objects

The mapping between DNNL and SYCL objects is provided in the
following table:

| DNNL object          | SYCL object(s)                             |
| :------------------- | :----------------------------------------- |
| Engine               | `cl::sycl::device` and `cl::sycl::context` |
| Stream               | `cl::sycl::queue`                          |
| Memory               | `cl::sycl::buffer<uint8_t, 1>`             |

@note Internally, library memory objects use 1D `uint8_t` SYCL buffers,
however user may initialize and access memory using SYCL buffers of a
different type, in this case buffers will be reinterpreted to the underlying
type `cl::sycl::buffer<uint8_t, 1>`.

## C++ API Extensions for Interoperability with SYCL

### API to Construct DNNL Objects

| DNNL object          | API to construct DNNL object                                          |
| :------------------- | :-------------------------------------------------------------------- |
| Engine               | [dnnl::engine(kind, sycl_dev, sycl_ctx)](@ref dnnl::engine)           |
| Stream               | [dnnl::stream(engine, sycl_queue)](@ref dnnl::stream)                 |
| Memory               | [dnnl::memory(memory_desc, engine, sycl_buf)](@ref dnnl::memory)      |

### API to Access SYCL Objects

| DNNL object          | API to access SYCL object(s)                                             |
| :------------------- | :----------------------------------------------------------------------- |
| Engine               | dnnl::engine::get_sycl_device() and dnnl::engine::get_sycl_context()     |
| Stream               | dnnl::stream::get_sycl_queue()                                           |
| Memory               | dnnl::memory::get_sycl_buffer<T>()                                       |
