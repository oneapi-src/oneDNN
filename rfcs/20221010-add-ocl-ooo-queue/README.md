# Out-of-Order Queue for OpenCL Runtime

## Motivation
Using an out-of-order queue allows users to fine tune kernel execution by managing
dependencies explicitly that can help to reduce undesired overhead.

The main driver for adding this feature is OpenVINO. Based on the analysis conducted
by the OpenVINO team, using an out-of-order queue in certain workloads can significantly
(up to 10%) improve their performance.

## Proposal
The proposal is to enable creating an out-of-order stream for OpenCL runtime and add
a new interop execution API that takes a list of dependencies and returns an event.

### Out-of-Order Stream
oneDNN stream abstraction that encapsulates an OpenCL queue, in the case of OpenCL runtime,
supports flags that controls stream's behavior. The flags include `in_order` and `out_of_order`
values that can be used to create an out-of-order stream. Currently, oneDNN returns a status
`unimplemented` when a user tries to create an out-of-order stream for OpenCL runtime.
The out-of-order stream will be made available for creation in the case of OpenCL runtime.

### OpenCL Interop API
A new interoperability API for execution a primitive will be introduced. The API is an
extension of the generic execute API and has two additional responsibilities:
1. Take a vector of events for dependencies
2. Return an event for the primitive that was submitted for execution

#### C API

```c
// dnnl_ocl.h

/// Executes computations specified by the primitive in a specified stream and
/// returns an OpenCL event.
///
/// @param primitive Primitive to execute.
/// @param stream Stream to use.
/// @param nargs Number of arguments.
/// @param args Array of arguments. Each argument is an
///     <index, #dnnl_memory_t> pair. The index is one of the `DNNL_ARG_*`
///     values such as `DNNL_ARG_SRC`. Unless runtime shapes are used (see
///     #DNNL_RUNTIME_DIM_VAL), the memory object must have the same memory
///     descriptor as that returned by
///     #dnnl_primitive_desc_query_md(#dnnl_query_exec_arg_md, index).
/// @param deps A pointer to a vector of size @p ndeps that contains
///     dependencies.
/// @param ndeps Number of dependencies.
/// @param return_event Output event. It's the user's responsibility to
///     manage lifetime of the event. Can be NULL. When @p stream is in-order
///     NULL will be returned.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_ocl_interop_primitive_execute(
        const_dnnl_primitive_t primitive, dnnl_stream_t stream, int nargs,
        const dnnl_exec_arg_t *args, const cl_event *deps, int ndeps,
        cl_event *return_event);
```

#### C++ API
```cpp
// dnnl_ocl.hpp

/// Executes computations specified by the primitive in a specified stream and
/// returns a SYCL event.
///
/// Arguments are passed via an arguments map containing
/// <index, memory object> pairs. The index must be one of the `DNNL_ARG_*`
/// values such as `DNNL_ARG_SRC`, and the memory must have a memory descriptor
/// matching the one returned by
/// #dnnl::primitive_desc::query_md(#query::exec_arg_md, index) unless using
/// dynamic shapes (see #DNNL_RUNTIME_DIM_VAL).
///
/// @param aprimitive Primitive to execute.
/// @param astream Stream object. The stream must belong to the same engine
///     as the primitive.
/// @param args Arguments map.
/// @param deps Optional vector with `cl_event` dependencies.
///
/// @returns Output event. It's the user's responsibility to manage lifetime
///     of the event.
inline cl_event execute(const dnnl::primitive &aprimitive,
        const stream &astream, const std::unordered_map<int, memory> &args,
        const std::vector<cl_event> &deps = {});
```

### Generic API
The generic API for primitive execution is available for either stream in-order or out-of-order.
Creating an out-of-order stream with the generic API will be made available for the OpenCL runtime.
