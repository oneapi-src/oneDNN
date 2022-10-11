# Out-of-Order Queue for OpenCL Runtime

## Motivation
Using an out-of-order queue allows users to fine tune kernel execution by managing
dependencies explicitly that can help to reduce undesired overhead.

The main driver for adding this feature is OpenVINO. Based on their analysis, using
an out-of-order queue in some of their workloads significantly (up to 10%) improves
performance.

## Proposal
The proposal is to enable creating an out-of-order stream for OpenCL runtime and add
a new interop execution API that takes a list of dependencies and returns an event.

### Out-of-Order Stream
oneDNN stream abstraction that encapsulates an OpenCL queue, in the case of OpenCL runtime,
supports flags that controls stream behavior. The flags include `in_order` and `out_of_order`
values that can be used to create an out-of-order stream. Currently, oneDNN returns a status
`unimplemented` when a user tries to create an out-of-order stream for OpenCL runtime
therefore no new API is required. The out-of-order stream will be made available for creation
in the case of OpenCL runtime.

### OpenCL Interop API
The API is an extension of the generic execute API and has two additional responsibilities:
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
/// @param deps A pointer to std::vector<cl_event> that contains
///     dependencies.
/// @param return_event Output event. The library doesn't own the event.
///     It is the user's responsibility to manage lifetime of the event.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_ocl_interop_primitive_execute(
        const_dnnl_primitive_t primitive, dnnl_stream_t stream, int nargs,
        const dnnl_exec_arg_t *args, const cl_event *deps, int ndeps,
        cl_event *return_event);
```

#### C++ API
```cpp
// dnnl_ocl.hpp

/// Executes computations specified by the primitive in a specified stream and
/// returns an OpenCL event.
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
/// @param deps Optional vector with `cl::sycl::event` dependencies.
///
/// @returns Output event. The library doesn't own the event. It is the user's
///     responsibility to manage lifetime of the event.
inline cl_event execute(const dnnl::primitive &aprimitive,
        const stream &astream, const std::unordered_map<int, memory> &args,
        const std::vector<cl_event> &deps = {});
```

### Generic API
Technically, the generic API for primitive execution is available for any stream. However, using
such API for an out-of-order stream doesn't make much sense because a user cannot specify
input dependencies and get an output event to embed oneDNN primitives in a larger
dependencies graph.

There is a need to specify behavior of the generic execute API for an out-of-order stream.
There are a couple of options.

#### Option 1: Prohibit Using Generic Execute API
The idea is to check whether stream is in-order or out-of-order in the generic execute API and
return an `invalid_arguments` status in the latter case.

Pros:
* It is clearly defined that the generic API is not suitable for out-of-order execution model
* Less error prone implementation because there is no need to handle generic and ocl interop APIs
differently inside the library

Cons:
* Discrepancy in behavior of SYCL and OCL runtimes. The former one allows to use the generic
execute API

#### Option 2: Allow to Use Only Generic or Only Interop Execute API
The idea is to allow users to use both APIs but document that mixing them will result in
undefined behavior. In order to handle dependencies between primitives submitted via the generic
execute API (i.e. emulate in-order behavior) the stream will be used to transfer dependencies
between execute calls. In the case of the interop API it will be the user's responsibility to
manage dependencies.

Pros:
* User doesn't have to manage dependencies when they use only the generic API

Cons:
* Applicability is questionable. oneDNN will support both in-order and out-of-order execution
models. If the user doesn't want to manage dependencies then they should use in-order execution
model
* Error prone approach. User can use both APIs for the same stream that will result in undefined
behavior and the library cannot ensure that the mixing will not happen

Proposal: go with the [option #1](#option-1-prohibit-using-generic-execute-api) because use
cases for the option #2 are not clear. Also, the main user of the feature will be OpenVINO that
won't use the generic API. The second option can be implemented later if there is a request.