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
| ONEDNN_EXPERIMENTAL_BNORM_STATS_ONE_PASS | Calculate mean and variance in batch normalization(BN) in single pass ([RFC](https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20210519-single-pass-bnorm)). |

| Build time option                          | Description                                                        |
|:-------------------------------------------|:-------------------------------------------------------------------|
| ONEDNN_EXPERIMENTAL_SPARSE                 | Enable experimental API and functionality for sparse domain.       |
| ONEDNN_EXPERIMENTAL_PROFILING              | Enable experimental profiling API.
| ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND | Enable experimental graph compiler backend of the graph component. |

## Features details

### ONEDNN_EXPERIMENTAL_SPARSE
This option extends the existing API and adds a new one to support sparse
functionality in oneDNN.

#### API

The main change is in oneDNN memory object semantics. Now, the memory object can
have multiple underlying buffers. In the case of regular dense computations, the
memory object always contains a single buffer. But in the case of sparse
computations, the memory object always contains one buffer for values and an
arbitrary number of additional buffers for metadata.

The underlying buffers are enumerated starting with 0, meaning that each buffer
has its own number. The buffer with values always has index 0.

In most cases, the API that works with underlying buffers takes a buffer index. The
exception is the API for creating a memory object. In that case, the API takes a vector
of buffers. The order of the buffers in the vector matters and should correspond to
the buffers' indices.

oneDNN also introduces a new format kind dnnl::memory::format_kind::sparse.
Sparse encoding (a.k.a. sparse format) is an
enumeration type that specifies how data is encoded. Currently, oneDNN
supports CSR (Compressed Sparse Row) and PACKED sparse encodings
(dnnl::memory::sparse_encoding::csr, dnnl::memory::sparse_encoding_packed).

The memory descriptor has dedicated static member functions for creating memory
descriptors for different sparse encodings.

Each encoding defines the number and meaning of the buffers.

| Sparse encoding | Buffers                                 |
|:----------------|:----------------------------------------|
| CSR             | 0 - values, 1 - indices, 2 - pointers   |
| PACKED          | The meaning and content are unspecified |

The pseudo-code below demonstrates how to create a memory object
for CSR sparse encoding and use the new API to work with the
underlying handles.

~~~cpp
    using namespace dnnl;
    const memory::dim M = 4, N = 6;
    const memory::dim nnz = 5;
    const auto values_dt = memory::data_type::f32;
    const auto indices_dt = memory::data_type::s32;
    const auto pointers_dt = memory::data_type::s32;

    // Create a memory descriptor for CSR sparse encoding.
    const auto csr_md = memory::desc::csr(
            {M, N}, // Dimensions
            values_dt, // Data type of values
            nnz, // Number of non-zero entries
            indices_dt, // Data type of indices (metadata)
            pointers_dt); // Data type of pointers (metadata)

    // A sparse matrix represented in the CSR format.
    std::vector<float> csr_values = {2.5f, 1.5f, 1.5f, 2.5f, 2.0f};
    std::vector<int32_t> csr_indices = {0, 2, 0, 5, 1};
    std::vector<int32_t> csr_pointers = {0, 1, 2, 4, 5, 5};

    // Create a memory object for the given buffers with values and metadata.
    memory csr_mem(csr_md, engine, {
        csr_values.data(), // Buffer with values
        csr_indices.data(), // Buffer with indices (metadata)
        csr_pointers.data() // Buffer with pointers (metadata)
        });

    const auto values_sz = csr_mem.get_size(0);
    const auto indices_sz = csr_mem.get_size(1);
    const auto pointers_sz = csr_mem.get_size(2);

    assert(values_sz == csr_values.size() * sizeof(float));
    assert(indices_sz == csr_indices.size() * sizeof(int32_t));
    assert(pointers_sz == csr_pointers.size() * sizeof(int32_t));

    void *values_handle = csr_mem.get_data_handle(0);
    void *indices_handle = csr_mem.get_data_handle(1);
    void *pointers_handle = csr_mem.get_data_handle(2);

    assert(values_handle == (void *)csr_values.data());
    assert(indices_handle == (void *)csr_indices.data());
    assert(pointers_handle == (void *)csr_pointers.data());
~~~

A memory descriptor created for the sparse encoding PACKED cannot
be used to create a memory object. It can only be used to create
a primitive descriptor to query the actual memory descriptor
(similar to the format tag `any`).

#### Primitives

##### Matrix Multiplication

This option enables the matmul primitive that can work with
sparse input tensors.

###### CSR encoding
Only one of the input tensors is allowed to be sparse. The
output tensor is always dense.

The following data types combinations are supported:

| Values | Indices | Pointers |
|:-------|:--------|:---------|
| f32    | s32     | s32      |

The following format tags are supported for dense input/output
tensors:

* ab

See the example [here](@ref cpu_matmul_csr_cpp).

Benchdnn can be used to test matmul with a CSR input tensor as follows:
`./benchdnn --matmul --encoding=csr+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128`

For the case above, the number of non-zero elements for the source tensor is
calculated as max(4 * 1000000 * (1 - 0.99), 1).

###### PACKED encoding

Only the weights tensor is allowed to be sparse. The other tensors
are always dense.

In general, it is expected that all matmul related functionality (e.g. post-ops,
scales, zero-points, etc) that is supported for the dense weights should
also work for the sparse weights.

Currently, matmul has the following limitations for the PACKED encoding:
* Only Intel Advanced Matrix Extensions (Intel AMX) instruction set
architecture (ISA) is supported
* Only `s8` data type for the weights is supported
* Only 1 batch dimension is supported

See the example [here](@ref cpu_matmul_weights_compression_cpp).

Benchdnn can be used to test matmul with the PACKED weights tensor as follows:
`./benchdnn --matmul --dt=s8:s8:s32 --encoding=:packed+0.99: 3x512x1024:1x1024x512`

For the case above, the number of non-zero elements for the weights tensor is
calculated as max(1024 * 512 * (1 - 0.99), 1).

##### Reorder

Currently, there is only one reorder for packing a dense tensor, i.e. converting
a dense tensor that is in `ab` format to a sparse tensor that is encoded with
the `PACKED` encoding.

In general, it is expected that all reorder-related functionality
(e.g. scales, zero-points, etc) that is supported for the dense
destination tensor should also work for the sparse one.

#### Common Limitations
* This functionality is not supported for SYCL and OpenCL runtimes
* The interoperability API for sparse memory is not provided
* Sparse memory and memory descriptor can only be used with the Matrix
Multiplication and Reorder primitives
* Sparse memory can be created only for a CPU engine

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
