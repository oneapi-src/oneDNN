SYCL Interoperability {#dev_guide_sycl_interoperability}
===============================================================

Intel MKL-DNN may use the SYCL runtime for CPU and GPU engines to interact with
the hardware. Users may need to use Intel MKL-DNN with other code that uses
SYCL. For that purpose, the library provides API extensions to interoperate
with underlying SYCL objects.

One of the possible scenarios is executing a SYCL kernel for a custom
operation not provided by Intel MKL-DNN. In this case, the library provides
all necessary API to "seamlessly" submit a kernel, sharing the execution
context with Intel MKL-DNN: using the same device and queue.

The interoperability API is provided for two scenarios:
- Construction of Intel MKL-DNN objects based on existing SYCL objects
- Accessing SYCL objects for existing Intel MKL-DNN objects

The mapping between Intel MKL-DNN and SYCL objects is provided in the
following table:

| Intel MKL-DNN object | SYCL object(s)                             |
| :------------------- | :----------------------------------------- |
| Engine               | `cl::sycl::device` and `cl::sycl::context` |
| Stream               | `cl::sycl::queue`                          |
| Memory               | `cl::sycl::buffer<uint8_t, 1>`             |

@note Internally, library memory objects use 1D `uint8_t` SYCL buffers,
however user may initialize and access memory using SYCL buffers of a
different type, in this case buffers will be reinterpreted to the underlying
type `cl::sycl::buffer<uint8_t, 1>`.

## C++ API Extensions for Interoperability with SYCL

### API to Construct Intel MKL-DNN Objects

| Intel MKL-DNN object | API to construct Intel MKL-DNN object                                 |
| :------------------- | :-------------------------------------------------------------------- |
| Engine               | [mkldnn::engine(kind, sycl_dev, sycl_ctx)](@ref mkldnn::engine)       |
| Stream               | [mkldnn::stream(engine, sycl_queue)](@ref mkldnn::stream)             |
| Memory               | [mkldnn::memory(memory_desc, engine, sycl_buf)](@ref mkldnn::memory)  |

### API to Access SYCL Objects

| Intel MKL-DNN object | API to access SYCL object(s)                                             |
| :------------------- | :----------------------------------------------------------------------- |
| Engine               | mkldnn::engine::get_sycl_device() and mkldnn::engine::get_sycl_context() |
| Stream               | mkldnn::stream::get_sycl_queue()                                         |
| Memory               | mkldnn::memory::get_sycl_buffer<T>()                                     |
