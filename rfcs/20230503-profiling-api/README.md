# Profiling API

## Motivation

OpenVINO requested to add profiling capabilities to oneDNN that would allow
it to collect necessary profiling data. The profiling data includes the
execution time of the kernels encapsulated in a primitive. The primary target
for the feature is GPU engine and OpenCL runtime as this is the configuration
in which OpenVINO is interested therefore the RFC will be focused on that.

## Proposal

The proposal is to introduce a set of APIs that could be used for getting any
profiling information but enable it only for getting the execution time of the
encapsulated kernels for a GPU engine and OpenCL runtime.

The API will be experimental because we do not have sufficient amount of use
cases to reliably define the semantics. The API can be enabled with a build
time option `ONEDNN_EXPERIMENTAL_PROFILING`.

The profiling API will have the following requirements:
* In order to enable profiling capabilities the stream should be either created with
a queue that was created with enabled profiling capabilities or created with a new
`stream::flags::profiling` flag
* Explicit synchronization after submitting a primitive to the queue is required
to query the profiling data for the primitive

In order to support profiling for multiple streams each stream will have its own
profiler, this is why the profiling API will have a stream parameter.

A typical workflow looks as follows:
* Create a oneDNN stream with a queue that was created with enabled profiling
capabilities or with using `stream::flags::profiling` flag
* Reset the profiler's state before executing the primitive (optional)
* Execute the primitive and explicitly wait for it to complete
* Use the corresponding API to query profiling data
* Reset the profiler's state

### API

The stream flags will get a new value `profiling`. Passing this flag upon a stream
creation will instruct the library to enable profiling capabilities for the stream.
```c
typedef enum {
    /* ... */
    
    /// Enables profiling capabilities.
    dnnl_stream_profiling = 0x4U,
} dnnl_stream_flags_t;
```

```cpp
struct stream {
/* ... */
    enum class flags : unsigned {
        /* ... */
        
        /// Enables profiling capabilities.
        profiling = dnnl_stream_profiling,
    };
```

The API for resetting the profiler's state:

#### C
```c
/// Resets a profiler's state.
///
/// @param stream Stream associated with the profiler.
///
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_reset_profiling(dnnl_stream_t stream);
```

#### C++
```cpp
/// Resets a profiler's state.
///
/// @param stream Stream associated with the profiler.
inline void reset_profiling(stream &stream);
```

The API to query profiling data.

#### C

```c
/// Profiling data kind.
typedef enum {
    /// Undefined profiling data kind.
    dnnl_profiling_data_kind_undef = 0,
    /// Data kind to query an execution time in nanoseconds.
    dnnl_profiling_data_kind_time,
} dnnl_profiling_data_kind_t;

/// Queries profiling data. The profiling data accumulates for each primitive
/// execution. The @p num_entries will be equal to the number of executions
/// since the last `dnnl_reset_profiling` call. In order to query the
/// @p num_entries the @p data parameter should be NULL. When @p data is NULL
/// then the @p data_kind parameter is ignored.
///
/// The profiling data can be reset by calling #dnnl_reset_profiling.
///
/// @param stream Stream that was used for executing a primitive that
/// is being profiled.
/// @param data_kind Profiling data kind to query.
/// @param data Profiling data.
///
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_query_profiling_data(dnnl_stream_t stream,
        dnnl_profiling_data_kind_t data_kind, int *num_entries,
        uint64_t *data);
```

#### C++
```cpp
/// Profiling data kind.
enum class profiling_data_kind {
    /// Undefined profiling data kind.
    undef = profiling_data_kind_undef,
    /// Data kind to query an execution time in nanoseconds.
    time = profiling_data_kind_time,
};

/// Returns requested profiling data. The profiling data accumulates for each
/// primitive execution. The size of the vector will be equal to the number
/// of executions since the last `dnnl::reset_profiling` call.
///
/// The profiling data can be reset by calling #dnnl::reset_profiling.
///
/// @param stream Stream that was used for executing a primitive that
///     is being profiled.
/// @param data_kind Profiling data kind to query.
///
/// @returns A vector with the requested profiling data.
inline std::vector<uint64_t> get_profiling_data(
        stream &stream, profiling_data_kind data_kind);
```

### Example

Below is pseudo-code that demonstrates the use of the C++ profiling API.
```cpp
    dnnl::engine engine(engine::kind::gpu, 0);
    // Create a queue with enabled profiling mode.
    cl_command_queue ocl_queue {};
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    ocl_queue = clCreateCommandQueueWithProperties(ocl_interop::get_context(engine),
        ocl_interop::get_device(engine), props, ...);
    // Create dnnl::stream with the queue.
    dnnl::stream stream = ocl_interop::make_stream(engine, ocl_queue);
    // Create a convolution primitive ... //
    // Reset profiler's state.
    dnnl::reset_profiling(stream);
    // Execute a primitive twice and wait for both executions to complete.
    conv_prim.execute(stream, ...)
    conv_prim.execute(stream, ...)
    stream.wait();
    // Query profiling data. The vector size will be equal to the number of
    // executions happend on the stream since the last `dnnl::reset_profiling`
    // call.
    std::vector<uint64_t> nsecs = dnnl::get_profiling_data(stream, profiling_data_kind::time);
    assert(nsecs.size() == 2);
    // Reset profiler's state.
    dnnl::reset_profiling(stream);
```
