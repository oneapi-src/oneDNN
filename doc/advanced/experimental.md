Experimental features {#dev_guide_experimental}
===============================================

To test aggressive performance optimizations that might affect accuracy or new
API and functionality without an impact to regular users, oneDNN provides
experimental features.

## Build-time Controls

There are two kinds of experimental features:
1. Features that can be enabled at runtime with an environment variable.
To enable such experimental features, the library should be built with a CMake
option `ONEDNN_EXPERIMENTAL=ON`. Each experimental feature has to be
individually selected using environment variables.
2. Features that can be enabled only with a build time option. To enable such
experimental features, the library should be built with a CMake option that
corresponds to a particular feature.

Both kinds of experimental features can be enabled simultaneously.

## Experimental features

| Environment variable                     | Description                                                                                                                                                    |
|:-----------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ONEDNN_EXPERIMENTAL_BNORM_STATS_ONE_PASS | Calculate mean and variance in batch normalization(BN) in single pass ([RFC](https://github.com/uxlfoundation/oneDNN/tree/rfcs/rfcs/20210519-single-pass-bnorm)). |
| ONEDNN_EXPERIMENTAL_GPU_CONV_V2          | Enable shapeless GPU convolution implementation (the feature is under development).                                                                            |

| Build time option                          | Description                                                        |
|:-------------------------------------------|:-------------------------------------------------------------------|
| ONEDNN_EXPERIMENTAL_UKERNEL                | Enable experimental microkernel APIs and functionalities.          |
| ONEDNN_EXPERIMENTAL_PROFILING              | Enable experimental profiling API.                                 |
| ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND | Enable experimental graph compiler backend of the graph component. |
| ONEDNN_EXPERIMENTAL_LOGGING                | Enable experimental logging support for oneDNN verbose mode.       |

## Features details

### ONEDNN_EXPERIMENTAL_UKERNEL

This option enables a new set of CPU-only APIs to support block-level
functionalities. By composing these low-level, sequential operations, users can
implement their own custom operations/fusions, and tailor blocking/threading
logic to their applications.

More details on this API are available in the [Microkernel APIs
section](@ref dev_guide_ukernel_basic_concepts).


### ONEDNN_EXPERIMENTAL_PROFILING
This option enables profiling API that can be used to query different
profiling data.

There are two ways to use the profiling capabilities:
* Create a queue with enabled profiling capabilities and use the
interoperability API to create a oneDNN stream with the queue. The library
will identify that the queue supports profiling and will collect profiling data
* Create a oneDNN stream using runtime agnostic API and enable
profiling capabilities using the stream flag `stream::flags::profiling`

Below is a pseudo-code that demonstrates the profiling API usage with a
user-provided queue.

~~~cpp
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
    // Enqueue same primitive twice and wait for both executions to complete.
    conv_prim.execute(stream, ...)
    conv_prim.execute(stream, ...)
    stream.wait();
    // Query profiling data. The vector size will be equal to the number of
    // executions happened on the stream since the last `dnnl::reset_profiling`
    // call.
    std::vector<uint64_t> nsecs = dnnl::get_profiling_data(stream, profiling_data_kind::time);
    assert(nsecs.size() == 2);
    // Reset profiler's state.
    dnnl::reset_profiling(stream);
~~~

@warning
- When the stream is created with enabled profiling capabilities it will
  collect profiling data for each primitive execution. It is the user's
  responsibility to reset the profiler's state to avoid consuming all
  memory resources in the system.


#### Limitations

* Only GPU engines with OpenCL and SYCL runtimes are supported
* Only Intel vendor is supported for SYCL runtime
* Out-of-order queue is not supported

### ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND
This option extends the coverage scope of the graph API to cover larger fusion
patterns apart from primitive patterns. Refer to
[Graph Compiler](@ref dev_guide_graph_compiler) for more details.

@warning
- Enabling some experimental features does not guarantee that the library will utilize them
- Enabling some experimental features might change the accuracy of oneDNN primitives

### ONEDNN_EXPERIMENTAL_LOGGING
This option introduces logging support in oneDNN which allows one to save the 
verbose outputs generated by oneDNN applications to user-specified logfiles.
By setting `ONEDNN_EXPERIMENTAL_LOGGING=ON`, a logging mechanism is built into
oneDNN using the third-party [spdlog](https://github.com/gabime/spdlog) 
library. Logging can then be enabled while running different applications by 
specifying the logfile path using `ONEDNN_VERBOSE_LOGFILE`:

~~~bash
$ ONEDNN_VERBOSE=all ONEDNN_VERBOSE_LOGFILE=./logs/cnn_test_logger.log ./examples/cnn-inference-f32-cpp
~~~

When logging is enabled while running an application, it also requires that
the verbose mode be enabled for the run using `ONEDNN_VERBOSE`. 
When no logfile is specified, logging is automatically disabled and 
the verbose output is printed only to the console. 
For the specified logfile path, the logger creates the base directory and the 
logfile if they do not already exist.
When the specified logfile already exists, the output is appended to the 
existing file until it reaches the maximum file size. 
**Note:** Multiple instances using the same filepath for `DNNL_VERBOSE_LOGFILE`
will write to the same file during the API run. 
The spdlog mechanism supports handling multiple instances concurrently 
if they write to the same logfile but the expectation is to specify different 
logfiles for different instances via the runtime variables.

By default, logging is disabled in oneDNN and any verbose output generated by 
oneDNN is printed only to `stdout`. The API is executed as a rotating lazy 
logger with a file size specified by 
`ONEDNN_VERBOSE_LOGFILE_SIZE(=1024*1024*50)`.
When logging is enabled, the user has the option to print verbose output to 
both `stdout` and the logfile by setting `ONEDNN_VERBOSE_LOG_WITH_CONSOLE=1`.
The runtime controls for oneDNN logging are listed as follows:

| Runtime variable                | Description                                                        |
|:--------------------------------|:-------------------------------------------------------------------|
| ONEDNN_VERBOSE_LOGFILE          | Enables verbose logging and specifies logfile path.                |
| ONEDNN_VERBOSE_LOGFILE_SIZE     | Specifies maximum size for the logfile.                            |
| ONEDNN_VERBOSE_NUM_LOGFILES     | Number of rotating logfiles for the logger.                        |
| ONEDNN_VERBOSE_LOG_WITH_CONSOLE | Enables printing to both stdout and the logfile.                   |
