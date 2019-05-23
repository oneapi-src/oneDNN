Getting Started with Intel(R) MKL-DNN with GPU support {#getting_started_gpu}
=============================================================================

This is an introduction to Intel MKL-DNN with GPU support.
We are going to walk through a simple example to demonstrate OpenCL\* extensions API in Intel MKL-DNN.

## Intel MKL-DNN basic workflow

A very simple workflow in Intel MKL-DNN includes the following steps:

- Engine creation
- Input/output memory objects creation
    - Memory descriptors creation
    - Memory objects creation
- Operation primitive creation
    - Operation descriptor creation
    - Operation primitive descriptor creation
    - Primitive creation
- Stream object creation
- Primitive submission for execution to a stream

## Create engine and memory object

Let's create a GPU engine object. The second parameter specifies the index of the requested engine.

~~~cpp
auto eng = engine(engine::kind::gpu, 0);
~~~

Then, we create a memory object. We need to specify dimensions of our memory by passing `memory::dims` object.
Then we create a memory descriptor with these dimensions, with `f32` data type and `nchw` memory format.
Finally, we construct a memory object and pass the memory descriptor. The library allocates memory internally.

~~~cpp
auto tz_dims = memory::dims{2, 3, 4, 5};
memory::desc mem_d(tz_dims, memory::data_type::f32, memory::format_tag::nchw);
memory mem(mem_d, eng);
~~~

## Initialize the data executing a custom OpenCL kernel

We are going to create an OpenCL kernel that will initialize our data.
It requires writing a bit of C code to create an OpenCL program from a string literal source, build it and extract the kernel.
The kernel initializes the data by the `0, -1, 2, -3, ...` sequence: `data[i] = (-1)^i * i`.

~~~cpp
const char *ocl_code
        = "__kernel void init(__global float *data) {"
          "    int id = get_global_id(0);"
          "    data[id] = (id % 2) ? -id : id;"
          "}";
const char *kernel_name = "init";
cl_kernel ocl_init_kernel = create_init_opencl_kernel(
        eng.get_ocl_context(), kernel_name, ocl_code);
~~~

Refer to the full code example for the code of `create_init_opencl_kernel()` function.
The next step is to execute our OpenCL kernel: set its arguments and enqueue to an OpenCL queue.
The underlying OpenCL buffer can be extracted from the memory object using
the interoperability interface: `memory::get_ocl_mem_object()`.
For simplicity we can just construct a stream, extract the underlying OpenCL queue and enqueue the kernel to this queue:

~~~cpp
cl_mem ocl_buf = mem.get_ocl_mem_object();
clSetKernelArg(ocl_init_kernel, 0, sizeof(ocl_buf), &ocl_buf);

mkldnn::stream strm(eng);
cl_command_queue ocl_queue = strm.get_ocl_command_queue();
clEnqueueNDRangeKernel(ocl_queue, ocl_init_kernel, 1, nullptr, &N, nullptr, 0,
                       nullptr, nullptr);
~~~

## Create and execute a primitive

There are 3 steps to create an operation primitive in Intel MKL-DNN:

- Create an operation descriptor
- Create a primitive descriptor
- Create a primitive

Let's create the primitive to perform ReLU (rectified linear unit) operation: `x = max(0, x)`.

~~~cpp
auto relu_d = eltwise_forward::desc(prop_kind::forward, algorithm::eltwise_relu,
                                    mem_d, 0.0f);
auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
auto relu = eltwise_forward(relu_pd);
~~~

From the code above we see that an operation descriptor has no dependency on a specific engine - it just describes some operation.
On the contrary, primitive descriptors are attached to a specific engine and represent some implementation for this engine.
A primitive object is realization of a primitive descriptor and its construction is usually much "heavier".

Note that for our primitive `mem` serves as both input and output parameter.

Next, execute the primitive:

~~~cpp
relu.execute(strm, { { MKLDNN_ARG_SRC, mem }, { MKLDNN_ARG_DST, mem } });
~~~

Note, primitive submission on GPU is asynchronous.
But user can call `stream::wait()` to synchronize the stream and ensure that all previously submitted primitives are completed.

## Validating the results

The simplest way to access the OpenCL memory is to map it to the host using `memory::map_data()` and `memory::unmap_data()` APIs.
After mapping this data is directly accessible (reading/writing) on the host. While the data is mapped, any GPU-side operations on this data are not allowed.
The data should be unmapped to release all resources associated with mapping.

~~~cpp
float *mapped_data = mem.map_data<float>();
for (size_t i = 0; i < N; i++) {
    float expected = (i % 2) ? 0.0f : (float)i;
    assert(mapped_data[i] == expected);
}
mem.unmap_data(mapped_data);
~~~

---

The full code example is listed below:

~~~cpp
#include <CL/cl.h>
#include <mkldnn.hpp>

#include <cassert>
#include <iostream>
#include <numeric>

using namespace mkldnn;

#define OCL_CHECK(x)                                                      \
    do {                                                                  \
        cl_int s = (x);                                                   \
        if (s != CL_SUCCESS) {                                            \
            printf("OpenCL error: %d at %s:%d\n", s, __FILE__, __LINE__); \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

cl_kernel create_init_opencl_kernel(
        cl_context ocl_ctx, const char *kernel_name, const char *ocl_code) {
    cl_int err;
    const char *sources[] = { ocl_code };
    cl_program ocl_program
            = clCreateProgramWithSource(ocl_ctx, 1, sources, nullptr, &err);
    OCL_CHECK(err);

    OCL_CHECK(
            clBuildProgram(ocl_program, 0, nullptr, nullptr, nullptr, nullptr));

    cl_kernel ocl_kernel = clCreateKernel(ocl_program, kernel_name, &err);
    OCL_CHECK(err);

    OCL_CHECK(clReleaseProgram(ocl_program));
    return ocl_kernel;
}

int main() {
    memory::dims tz_dims = { 2, 3, 4, 5 };
    const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
            std::multiplies<size_t>());

    memory::desc mem_d(tz_dims, memory::data_type::f32,
            memory::format_tag::nchw);

    engine eng(engine::kind::gpu, 0);
    memory mem(mem_d, eng);

    // Extract OpenCL buffer from memory object
    cl_mem ocl_buf = mem.get_ocl_mem_object();

    // Create stream
    mkldnn::stream strm(eng);

    // Create custom OpenCL kernel to initialize the data
    const char *ocl_code
            = "__kernel void init(__global float *data) {"
              "    int id = get_global_id(0);"
              "    data[id] = (id % 2) ? -id : id;"
              "}";
    const char *kernel_name = "init";
    cl_kernel ocl_init_kernel = create_init_opencl_kernel(
            eng.get_ocl_context(), kernel_name, ocl_code);

    // Execute the custom OpenCL kernel
    OCL_CHECK(clSetKernelArg(ocl_init_kernel, 0, sizeof(ocl_buf), &ocl_buf));

    cl_command_queue ocl_queue = strm.get_ocl_command_queue();
    OCL_CHECK(clEnqueueNDRangeKernel(ocl_queue, ocl_init_kernel, 1, nullptr, &N,
            nullptr, 0, nullptr, nullptr));

    // Perform ReLU operation by executing the primitive
    auto relu_d = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, mem_d, 0.0f);
    auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
    auto relu = eltwise_forward(relu_pd);
    relu.execute(strm, { { MKLDNN_ARG_SRC, mem }, { MKLDNN_ARG_DST, mem } });
    strm.wait();

    // Map the data to the host to validate the results
    float *mapped_data = mem.map_data<float>();
    for (size_t i = 0; i < N; i++) {
        float expected = (i % 2) ? 0.0f : (float)i;
        assert(mapped_data[i] == expected);
    }
    mem.unmap_data(mapped_data);

    OCL_CHECK(clReleaseKernel(ocl_init_kernel));

    std::cout << "PASSED" << std::endl;
    return 0;
}
~~~
