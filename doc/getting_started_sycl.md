Getting Started with Intel(R) MKL-DNN with SYCL\* support {#getting_started_sycl}
=============================================================================

This is an introduction to Intel MKL-DNN with SYCL support.
We are going to walk through a simple example to demonstrate SYCL extensions API in MKL-DNN.

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

## Initialize the data executing a custom SYCL kernel

We are going to create an SYCL kernel that should initialize our data. To execute SYCL kernel we need a SYCL queue.
The underlying SYCL buffer can be extracted from the memory object using
the interoperability interface: `memory::get_sycl_buffer<T>()`.
For simplicity we can construct a stream and extract the SYCL queue from it.
The kernel initializes the data by the `0, -1, 2, -3, ...` sequence: `data[i] = (-1)^i * i`.

~~~cpp
auto &sycl_buf = mem.get_sycl_buffer<float>();

mkldnn::stream strm(eng);

queue sycl_queue = strm.get_sycl_queue();
sycl_queue.submit([&](handler& cgh) {
    auto a = sycl_buf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<kernel_tag>(range<1>(N), [=](id<1> i) {
        int idx = i[0];
        a[idx] = (idx % 2) ? -idx : idx;
    });
});
~~~

## Create and execute a primitive

There are 3 steps to create an operation primitive in Intel MKL-DNN:

- Create an operation descriptor
- Create a primitive descriptor
- Create a primitive

Let's create the primitive to perform ReLU (recitified linear unit) operation: `x = max(0, x)`.

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

The simplest way to access the SYCL-backed memory on the host is to construct a host accessor.
Then we can directly read and write this data on the host.
However no any conflicting operations are allowed until the host accessor is destroyed.

~~~cpp
auto host_acc = sycl_buf.get_access<access::mode::read>();
for (size_t i = 0; i < N; i++) {
    float exp_value = (i % 2) ? 0.0f : i;
    assert(host_acc[i] == (float)exp_value);
}
~~~

---

The full code example is listed below:

~~~cpp
#include <CL/sycl.hpp>
#include <mkldnn.hpp>

#include <cassert>
#include <iostream>
#include <numeric>

using namespace mkldnn;
using namespace cl::sycl;

class kernel_tag;

int main() {
    memory::dims tz_dims = { 2, 3, 4, 5 };
    const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
            std::multiplies<size_t>());

    memory::desc mem_d(tz_dims, memory::data_type::f32,
            memory::format_tag::nchw);

    engine eng(engine::kind::gpu, 0);
    memory mem(mem_d, eng);

    // Extract SYCL buffer from memory object
    auto &sycl_buf = mem.get_sycl_buffer<float>();

    // Create stream
    mkldnn::stream strm(eng);

    // Initialize the buffer directly using SYCL
    queue q = strm.get_sycl_queue();
    q.submit([&](handler &cgh) {
        auto a = sycl_buf.get_access<access::mode::write>(cgh);
        cgh.parallel_for<kernel_tag>(range<1>(N), [=](id<1> i) {
            int idx = i[0];
            a[idx] = (idx % 2) ? -idx : idx;
        });
    });

    // Perform ReLU operation by executing the MKL-DNN primitive
    auto relu_d = eltwise_forward::desc(prop_kind::forward,
            algorithm::eltwise_relu, mem_d, 0.0f);
    auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
    auto relu = eltwise_forward(relu_pd);
    relu.execute(strm, { { MKLDNN_ARG_SRC, mem }, { MKLDNN_ARG_DST, mem } });
    strm.wait();

    // Create a host accessor to validate the results
    auto host_acc = sycl_buf.get_access<access::mode::read>();

    for (size_t i = 0; i < N; i++) {
        float exp_value = (i % 2) ? 0.0f : i;
        assert(host_acc[i] == (float)exp_value);
    }
    std::cout << "PASSED" << std::endl;
    return 0;
}
~~~

---

[Legal information](@ref legal_information)
