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
* The profiling mechanism will have to be enabled and disabled explicitly with
the corresponding API
* Enabling of the mechanism should be done prior to the first engine creation
* When the mechanism is enabled the oneDNN stream is expected to contain a queue
that was created with enabled profiling capabilities
* Explicit synchronization after submitting a primitive to the queue is required
to query the profiling data for the primitive

In order to support profiling for multiple streams each stream will have its own
profiler, this is why some profiling API will have a stream as a parameter.

A typical workflow looks as follows:
* Enable profiling
* Create a oneDNN stream with a queue that was created with enabled profiling
capabilities
* Reset the profiler's state before executing the primitive
* Execute the primitive and explicitly wait for it to complete
* Use the corresponding API to query profiling data
* Reset the profiler's state
* Disable profiling

### API

The API for enabling and disabling profiling:

#### C
```c
/// Enable profiling capabilities.
///
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_enable_profiling(void);

/// Disable profiling capabilities.
///
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_disable_profiling(void);
```

#### C++
```cpp
/// @copydoc dnnl_enable_profiling()
status enable_profiling();

/// @copydoc dnnl_disable_profiling()
status disable_profiling();
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
/// @copydoc dnnl_reset_profiling()
status reset_profiling(stream &stream);
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

/// Profiling mode.
typedef enum {
    /// Undefined profiling mode.
    dnnl_profiling_mode_undef = 0,
    /// The profiling data for each kernel will be added up. For example,
    /// if the profiling data kind is time and there are n kernels then the
    /// result will be timeK0 + timeK1 + timeKn-1.
    dnnl_profiling_mode_sum,
} dnnl_profiling_mode_t;


/// Queries profiling data.
///
/// @param stream Stream that was used for executing a primitive that
/// is being profiled.
/// @param data_kind Profiling data kind to query.
/// @param mode Profiling mode.
/// @param data Profiling data.
///
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_query_profiling_data(dnnl_stream_t stream,
        dnnl_profiling_data_kind_t data_kind,
        dnnl_profiling_mode_t mode, uint64_t *data);
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

/// Profiling mode.
enum class profiling_mode {
    /// Undefined profiling mode.
    undef = dnnl_profiling_mode_undef,
    /// The profiling data for each kernel will be added up. For example,
    /// if the profiling data kind is time and there are n kernels then the
    /// result will be timeK0 + timeK1 + timeKn-1.
    sum = dnnl_profiling_mode_sum,
};

/// Returns requested profiling data.
/// @param stream Stream that was used for executing a primitive that
/// is being profiled.
/// @param data_kind Profiling data kind to query.
/// @param mode Profiling mode.
///
/// @returns A value that corresponds to the requested @p data_kind.
uint64_t get_profiling_data(stream &stream, profiling_data_kind data_kind,
        profiling_mode mode);
```

### Example

Below is pseudo-code that demonstrates the use of the C++ profiling API.
```cpp
    // Enable profiling before the first engine creation.
    dnnl::enable_profiling();

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

    // Execute a primitive and wait for it to complete.
    conv_prim.execute(stream, ...)
    stream.wait();

    // Query profiling data.
    uint64_t nsec = dnnl::get_profiling_data(stream, profiling_data_kind::time,
            profiling_mode::sum);

    // Reset profiler's state.
    dnnl::reset_profiling(stream);
    // Disable profiling.
    dnnl::disable_profiling();
```