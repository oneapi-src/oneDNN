DPC++ Unified Shared Memory Support {#dev_guide_dpcpp_usm}
===============================================================

Unified Shared Memory (USM) is a DPC++ capability that provides the ability
to allocate and use memory in a uniform way on host and DPC++ devices.

oneDNN enables you to use two sets of interfaces with memory: SYCL buffers-based
and USM-based. This is controlled by the special macros: `DNNL_USE_SYCL_BUFFERS`
for SYCL buffers (which is the default) and `DNNL_USE_DPCPP_USM` for USM. The
corresponding macro must be defined to enable SYCL buffers or USM interfaces.

For example, to enable USM interfaces in oneDNN, the user must pass the
`DNNL_USE_DPCPP_USM` macro:

~~~sh
${CXX} application.cpp -DDNNL_USE_DPCPP_USM ...
~~~

@note SYCL buffers and USM interfaces cannot be used together in a single
application. With the `DNNL_USE_SYCL_BUFFERS` macro defined, the library
provides interfaces working with SYCL buffers only and oneDNN memory objects are
always constructed with SYCL buffers. With the `DNNL_USE_DPCPP_USM` macro
defined, the library provides interfaces working with USM only and oneDNN memory
objects are always constructed with USM memory.

## API

oneDNN provides the following interfaces to work with USM memory:

- `memory mem(md, eng);`

    Constructs a memory object from an internally allocated USM memory using
    `cl::sycl::malloc_shared()`. The constructed object owns the allocated USM
    memory, which is deallocated on destruction of the object.

- `memory mem(md, eng, usm_ptr);`

    Constructs a memory object from a USM pointer. The constructed object does
    not own the passed USM memory. The user is reponsible for deallocation of
    `usm_ptr`.

- `usm_ptr = mem.get_data_handle();`

    Returns the USM pointer, associated with a memory object.

- `mem.set_data_handle(usm_ptr);`

    Sets the USM pointer, associated with a memory object. Any previously owned
    USM memory will be deallocated by the memory object.

## Handling Dependencies with USM

SYCL queues from the SYCL 1.2.1 specification are inherently out-of-order. That
means that the order of execution is defined by the dependencies between SYCL
tasks. The runtime tracks dependencies based on accessors created for SYCL
buffers. USM pointers cannot be used to create accessors and users must handle
dependencies on their own using SYCL events.

oneDNN provides two mechanisms to handle dependencies when USM memory is used:

1. `primitive::execute_sycl()` interface

    This interface enables you to pass dependencies between primitives using
    SYCL events. In this case, the user is responsible for passing proper
    dependencies for every primitive execution.

2. In-order oneDNN stream

    oneDNN enables you to create in-order streams when submitted primitives are
    executed in the order they were submitted. Using in-order streams prevents
    possible read-before-write or concurrent read/write issues.

    @note Performance with in-order oneDNN streams with the DPC++ runtime may be
    suboptimal due to the lack of in-order queue support in the SYCL 1.2.1
    specification.

## SYCL Buffers and DPC++ USM Interfaces

The tables below summarize the behavior of the oneDNN interfaces for working with
memory.

### For constructors:

| Macro defined                     | `memory mem(md, eng)`                 | `memory mem(md, eng, ptr)`                                                          |
| :-------------------------------- | :------------------------------------ | :---------------------------------------------------------------------------------- |
| `DNNL_USE_SYCL_BUFFERS` (default) | Create memory object with SYCL buffer | Can be used only if `ptr` is either `DNNL_MEMORY_NONE` or `DNNL_MEMORY_ALLOCATE`    |
| `DNNL_USE_DPCPP_USM`              | Create memory object with USM memory  | Use user-provided USM pointer to construct memory object                            |


### For get and set functions:

| Macro defined                     | `ptr = mem.get_data_handle()`     | `mem.set_data_handle(ptr)`                     |
| :-------------------------------- | :-------------------------------- | :--------------------------------------------- |
| `DNNL_USE_SYCL_BUFFERS` (default) | Return pointer to the SYCL buffer | Dereference `ptr` as a SYCL buffer and copy it |
| `DNNL_USE_DPCPP_USM`              | Return the USM pointer            | Set the USM pointer                            |
