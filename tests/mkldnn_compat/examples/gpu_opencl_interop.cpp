/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/// @example  gpu_opencl_interop.cpp
/// Annotated version: @ref gpu_opencl_interop_cpp
///
/// @page  gpu_opencl_interop_cpp Getting started on GPU with OpenCL extensions API
/// Full example text: @ref gpu_opencl_interop.cpp
///
/// This C++ API example demonstrates programming for Intel(R) Processor
/// Graphics with OpenCL* extensions API in Intel MKL-DNN.
/// The workflow includes following steps:
///   - Create a GPU engine. It uses OpenCL as the runtime in this sample.
///   - Create a GPU memory descriptor/object.
///   - Create an OpenCL kernel for GPU data initialization
///   - Access a GPU memory via OpenCL interoperability interface
///   - Access a GPU command queue via OpenCL interoperability interface
///   - Execute a OpenCL kernel with related GPU command queue and GPU memory
///   - Create operation descriptor/operation primitives descriptor/primitive .
///   - Execute the primitive with the initialized GPU memory
///   - Validate the result by mapping the OpenCL memory via OpenCL interoperability
///     interface
///
/// @page gpu_opencl_interop_cpp

/// @section gpu_opencl_interop_cpp_headers Public headers
///
/// To start using Intel MKL-DNN, we must first include the @ref mkldnn.hpp
/// header file in the application. We also include CL/cl.h for using
/// OpenCL APIs and @ref mkldnn_debug.h, which  contains some debugging
/// facilities such as returning a string representation
/// for common Intel MKL-DNN C types.
/// All C++ API types and functions reside in the `mkldnn` namespace.
/// For simplicity of the example we import this namespace.
/// @page gpu_opencl_interop_cpp
/// @snippet  gpu_opencl_interop.cpp Prologue
// [Prologue]
#include <mkldnn.hpp>
#include <CL/cl.h>
// Optional header to access debug functions like `mkldnn_status2str()`
#include "mkldnn_debug.h"

#include <iostream>
#include <numeric>
#include <sstream>

using namespace mkldnn;
using namespace std;
// [Prologue]

#define OCL_CHECK(x) \
    do { \
        cl_int s = (x); \
        if (s != CL_SUCCESS) { \
            printf("OpenCL error: %d at %s:%d\n", s, __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

cl_kernel create_init_opencl_kernel(
        cl_context ocl_ctx, const char *kernel_name, const char *ocl_code) {
    cl_int err;
    const char *sources[] = {ocl_code};
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

/// @page gpu_opencl_interop_cpp
/// @section gpu_opencl_interop_cpp_tutorial gpu_opencl_interop_tutorial() function
/// @page gpu_opencl_interop_cpp
void gpu_opencl_interop_tutorial() {
    /// @page gpu_opencl_interop_cpp
    /// @subsection gpu_opencl_interop_cpp_sub1 Engine and stream
    ///
    /// All Intel MKL-DNN primitives and memory objects are attached to a
    /// particular @ref dnnl::engine, which is an abstraction of a
    /// computational device (see also @ref dev_guide_basic_concepts). The
    /// primitives are created and optimized for the device to which they are
    /// attached, and the memory objects refer to memory residing on the
    /// corresponding device. In particular, that means neither memory objects
    /// nor primitives that were created for one engine can be used on
    /// another.
    ///
    /// To create engines, we must specify the @ref dnnl::engine::kind
    /// and the index of the device of the given kind. In this example we use
    /// the first available GPU engine, so the index for the engine is 0.
    /// This example assumes OpenCL being a runtime for GPU. In such case,
    /// during engine creation, an OpenCL context is also created and attaches
    /// to the created engine.
    ///
    /// @snippet  gpu_opencl_interop.cpp Initialize engine
    // [Initialize engine]
    engine eng(engine::kind::gpu, 0);
    // [Initialize engine]

    /// In addition to an engine, all primitives require a @ref dnnl::stream
    /// for the execution. The stream encapsulates an execution context and is
    /// tied to a particular engine.
    ///
    /// In this example, a GPU stream is created.
    /// This example assumes OpenCL being a runtime for GPU. During stream creation,
    /// an OpenCL command queue is also created and attaches to this stream.
    ///
    /// @snippet  gpu_opencl_interop.cpp Initialize stream
    // [Initialize stream]
    mkldnn::stream strm(eng);
    // [Initialize stream]

    /// @subsection  gpu_opencl_interop_cpp_sub2 Wrapping data into Intel MKL-DNN memory object
    ///
    /// Next, we create a memory object. We need to specify dimensions of our
    /// memory by passing a memory::dims object. Then we create a memory
    /// descriptor with these dimensions, with the mkldnn::memory::data_type::f32
    /// data type, and with the mkldnn::memory::format_tag::nchw memory format.
    /// Finally, we construct a memory object and pass the memory descriptor.
    /// The library allocates memory internally.
    /// @snippet  gpu_opencl_interop.cpp memory alloc
    //  [memory alloc]
    memory::dims tz_dims = {2, 3, 4, 5};
    const size_t N = std::accumulate(tz_dims.begin(), tz_dims.end(), (size_t)1,
            std::multiplies<size_t>());

    memory::desc mem_d(
            tz_dims, memory::data_type::f32, memory::format_tag::nchw);

    memory mem(mem_d, eng);
    //  [memory alloc]

    /// @subsection  gpu_opencl_interop_cpp_sub3 Initialize the data by executing a custom OpenCL kernel
    /// We are going to create an OpenCL kernel that will initialize our data.
    /// It requires writing a bit of C code to create an OpenCL program from a
    /// string literal source. The kernel initializes the data by the
    /// 0, -1, 2, -3, ... sequence: `data[i] = (-1)^i * i`.
    /// @snippet  gpu_opencl_interop.cpp ocl kernel
    //  [ocl kernel]
    const char *ocl_code
            = "__kernel void init(__global float *data) {"
              "    int id = get_global_id(0);"
              "    data[id] = (id % 2) ? -id : id;"
              "}";
    //  [ocl kernel]

    /// Create/Build Opencl kernel by `create_init_opencl_kernel()` function.
    /// Refer to the full code example for the `create_init_opencl_kernel()`
    /// function.
    /// @snippet  gpu_opencl_interop.cpp oclkernel create
    // [oclkernel create]
    const char *kernel_name = "init";
    cl_kernel ocl_init_kernel = create_init_opencl_kernel(
            eng.get_ocl_context(), kernel_name, ocl_code);
    //  [oclkernel create]

    /// The next step is to execute our OpenCL kernel by setting its arguments
    /// and enqueueing to an OpenCL queue. You can extract the underlying OpenCL
    /// buffer from the memory object using  the interoperability interface:
    /// mkldnn::memory::get_ocl_mem_object() . For simplicity we can just construct a
    /// stream, extract the underlying OpenCL queue, and enqueue the kernel to
    /// this queue.
    /// @snippet  gpu_opencl_interop.cpp oclexecution
    // [oclexecution]
    cl_mem ocl_buf = mem.get_ocl_mem_object();
    OCL_CHECK(clSetKernelArg(ocl_init_kernel, 0, sizeof(ocl_buf), &ocl_buf));

    cl_command_queue ocl_queue = strm.get_ocl_command_queue();
    OCL_CHECK(clEnqueueNDRangeKernel(ocl_queue, ocl_init_kernel, 1, nullptr, &N,
            nullptr, 0, nullptr, nullptr));
    // [oclexecution]

    /// @subsection gpu_opencl_interop_cpp_sub4 Create and execute a primitive
    /// There are three steps to create an operation primitive in Intel MKL-DNN:
    /// 1. Create an operation descriptor.
    /// 2. Create a primitive descriptor.
    /// 3. Create a primitive.
    ///
    /// Let's create the primitive to perform the ReLU (rectified linear unit)
    /// operation: x = max(0, x). An operation descriptor has no dependency on a
    /// specific engine - it just describes some operation. On the contrary,
    /// primitive descriptors are attached to a specific engine and represent
    /// some implementation for this engine. A primitive object is a realization
    /// of a primitive descriptor, and its construction is usually much
    /// "heavier".
    /// @snippet gpu_opencl_interop.cpp relu creation
    //  [relu creation]
    auto relu_d = eltwise_forward::desc(
            prop_kind::forward, algorithm::eltwise_relu, mem_d, 0.0f);
    auto relu_pd = eltwise_forward::primitive_desc(relu_d, eng);
    auto relu = eltwise_forward(relu_pd);
    //  [relu creation]

    /// Next, execute the primitive.
    /// @snippet gpu_opencl_interop.cpp relu exec
    // [relu exec]
    relu.execute(strm, {{MKLDNN_ARG_SRC, mem}, {MKLDNN_ARG_DST, mem}});
    strm.wait();
    // [relu exec]
    ///
    ///@note
    ///    Our primitive mem serves as both input and output parameter.
    ///
    ///
    ///@note
    ///    Primitive submission on GPU is asynchronous; However, the user can
    ///    call mkldnn:stream::wait() to synchronize the stream and ensure that all
    ///    previously submitted primitives are completed.
    ///

    /// @page gpu_opencl_interop_cpp
    /// @subsection gpu_opencl_interop_cpp_sub5 Validate the results
    ///
    /// Before running validation codes, we need to access the OpenCL memory on
    /// the host. The simplest way to access the OpenCL memory is to map it to
    /// the host using the mkldnn::memory::map_data() and
    /// mkldnn::memory::unmap_data() APIs. After mapping, this data is directly
    /// accessible  for reading or writing on the host.
    /// We can run validation codes on the host accordingly. While
    /// the data is mapped, no GPU-side operations on this data are allowed.
    /// The data should be unmapped to release all resources associated with
    /// mapping.
    /// @snippet gpu_opencl_interop.cpp Check the results
    // [Check the results]
    float *mapped_data = mem.map_data<float>();
    for (size_t i = 0; i < N; i++) {
        float expected = (i % 2) ? 0.0f : (float)i;
        if (mapped_data[i] != expected)
            throw std::string(
                    "Unexpected output, find a negative value after the ReLU "
                    "execution");
    }
    mem.unmap_data(mapped_data);
    // [Check the results]

    OCL_CHECK(clReleaseKernel(ocl_init_kernel));
}
/// @section gpu_opencl_interop_cpp_main main() function
///
/// We now just call everything we prepared earlier.
///
/// Because we are using the Intel MKL-DNN C++ API, we use exceptions to handle
/// errors (see @ref dev_guide_c_and_cpp_apis). The Intel MKL-DNN C++ API throws
/// exceptions of type @ref dnnl::error, which contains the error status
/// (of type @ref mkldnn_status_t) and a human-readable error message accessible
/// through the regular `what()` method.
/// @page gpu_opencl_interop_cpp
/// @snippet gpu_opencl_interop.cpp Main
// [Main]
int main(int argc, char **argv) {
    try {
        gpu_opencl_interop_tutorial();
    } catch (mkldnn::error &e) {
        std::cerr << "Intel MKL-DNN error: " << e.what() << std::endl
                  << "Error status: " << mkldnn_status2str(e.status)
                  << std::endl;
        return 1;
    } catch (std::string &e) {
        std::cerr << "Error in the example: " << e << std::endl;
        return 2;
    }

    std::cout << "Example passes" << std::endl;
    return 0;
}
// [Main]
/// <b></b>
///
/// Upon compiling and running the example, the output should be just:
///
/// ~~~
/// Example passes
/// ~~~
///
/// @page gpu_opencl_interop_cpp
