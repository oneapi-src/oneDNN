/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include <memory>
#include "oneapi/dnnl/dnnl_ocl.h"
#include "oneapi/dnnl/dnnl_ocl.hpp"
#include <CL/cl.h>

namespace dnnl {
class ocl_stream_test_c_t : public ::testing::Test {
protected:
    void SetUp() override {
        if (!find_ocl_device(CL_DEVICE_TYPE_GPU)) { return; }

        DNNL_CHECK(dnnl_engine_create(&eng, dnnl_gpu, 0));

        DNNL_CHECK(dnnl_ocl_interop_engine_get_context(eng, &ocl_ctx));
        DNNL_CHECK(dnnl_ocl_interop_get_device(eng, &ocl_dev));
    }

    void TearDown() override {
        if (eng) { DNNL_CHECK(dnnl_engine_destroy(eng)); }
    }

    dnnl_engine_t eng = nullptr;
    cl_context ocl_ctx = nullptr;
    cl_device_id ocl_dev = nullptr;
};

class ocl_stream_test_cpp_t : public ::testing::Test {
protected:
    void SetUp() override {
        if (!find_ocl_device(CL_DEVICE_TYPE_GPU)) { return; }

        eng = engine(engine::kind::gpu, 0);

        ocl_ctx = ocl_interop::get_context(eng);
        ocl_dev = ocl_interop::get_device(eng);
    }

    engine eng;
    cl_context ocl_ctx = nullptr;
    cl_device_id ocl_dev = nullptr;
};

TEST_F(ocl_stream_test_c_t, CreateC) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    dnnl_stream_t stream;
    DNNL_CHECK(dnnl_stream_create(&stream, eng, dnnl_stream_default_flags));

    cl_command_queue ocl_queue;
    DNNL_CHECK(dnnl_ocl_interop_stream_get_command_queue(stream, &ocl_queue));

    cl_device_id ocl_queue_dev;
    cl_context ocl_queue_ctx;
    TEST_OCL_CHECK(clGetCommandQueueInfo(ocl_queue, CL_QUEUE_DEVICE,
            sizeof(ocl_queue_dev), &ocl_queue_dev, nullptr));
    TEST_OCL_CHECK(clGetCommandQueueInfo(ocl_queue, CL_QUEUE_CONTEXT,
            sizeof(ocl_queue_ctx), &ocl_queue_ctx, nullptr));

    ASSERT_EQ(ocl_dev, ocl_queue_dev);
    ASSERT_EQ(ocl_ctx, ocl_queue_ctx);

    DNNL_CHECK(dnnl_stream_destroy(stream));
}

TEST_F(ocl_stream_test_cpp_t, CreateCpp) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    stream s(eng);
    cl_command_queue ocl_queue = ocl_interop::get_command_queue(s);

    cl_device_id ocl_queue_dev;
    cl_context ocl_queue_ctx;
    TEST_OCL_CHECK(clGetCommandQueueInfo(ocl_queue, CL_QUEUE_DEVICE,
            sizeof(ocl_queue_dev), &ocl_queue_dev, nullptr));
    TEST_OCL_CHECK(clGetCommandQueueInfo(ocl_queue, CL_QUEUE_CONTEXT,
            sizeof(ocl_queue_ctx), &ocl_queue_ctx, nullptr));

    ASSERT_EQ(ocl_dev, ocl_queue_dev);
    ASSERT_EQ(ocl_ctx, ocl_queue_ctx);
}

TEST_F(ocl_stream_test_c_t, BasicInteropC) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    cl_int err;
#ifdef CL_VERSION_2_0
    cl_command_queue interop_ocl_queue = clCreateCommandQueueWithProperties(
            ocl_ctx, ocl_dev, nullptr, &err);
#else
    cl_command_queue interop_ocl_queue
            = clCreateCommandQueue(ocl_ctx, ocl_dev, 0, &err);
#endif
    TEST_OCL_CHECK(err);

    dnnl_stream_t stream;
    DNNL_CHECK(dnnl_ocl_interop_stream_create(&stream, eng, interop_ocl_queue));

    cl_command_queue ocl_queue;
    DNNL_CHECK(dnnl_ocl_interop_stream_get_command_queue(stream, &ocl_queue));
    ASSERT_EQ(ocl_queue, interop_ocl_queue);

    cl_uint ref_count;
    TEST_OCL_CHECK(clGetCommandQueueInfo(interop_ocl_queue,
            CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr));
    int i_ref_count = int(ref_count);
    ASSERT_EQ(i_ref_count, 2);

    DNNL_CHECK(dnnl_stream_destroy(stream));

    TEST_OCL_CHECK(clGetCommandQueueInfo(interop_ocl_queue,
            CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr));
    i_ref_count = int(ref_count);
    ASSERT_EQ(i_ref_count, 1);

    TEST_OCL_CHECK(clReleaseCommandQueue(interop_ocl_queue));
}

TEST_F(ocl_stream_test_cpp_t, BasicInteropC) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    cl_int err;
#ifdef CL_VERSION_2_0
    cl_command_queue interop_ocl_queue = clCreateCommandQueueWithProperties(
            ocl_ctx, ocl_dev, nullptr, &err);
#else
    cl_command_queue interop_ocl_queue
            = clCreateCommandQueue(ocl_ctx, ocl_dev, 0, &err);
#endif
    TEST_OCL_CHECK(err);

    {
        auto s = ocl_interop::make_stream(eng, interop_ocl_queue);

        cl_uint ref_count;
        TEST_OCL_CHECK(clGetCommandQueueInfo(interop_ocl_queue,
                CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count,
                nullptr));
        int i_ref_count = int(ref_count);
        ASSERT_EQ(i_ref_count, 2);

        cl_command_queue ocl_queue = ocl_interop::get_command_queue(s);
        ASSERT_EQ(ocl_queue, interop_ocl_queue);
    }

    cl_uint ref_count;
    TEST_OCL_CHECK(clGetCommandQueueInfo(interop_ocl_queue,
            CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, nullptr));
    int i_ref_count = int(ref_count);
    ASSERT_EQ(i_ref_count, 1);

    TEST_OCL_CHECK(clReleaseCommandQueue(interop_ocl_queue));
}

TEST_F(ocl_stream_test_c_t, InteropIncompatibleQueueC) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    cl_device_id cpu_ocl_dev = find_ocl_device(CL_DEVICE_TYPE_CPU);
    SKIP_IF(!cpu_ocl_dev, "OpenCL CPU devices not found.");

    cl_int err;
    cl_context cpu_ocl_ctx
            = clCreateContext(nullptr, 1, &cpu_ocl_dev, nullptr, nullptr, &err);
    TEST_OCL_CHECK(err);

#ifdef CL_VERSION_2_0
    cl_command_queue cpu_ocl_queue = clCreateCommandQueueWithProperties(
            cpu_ocl_ctx, cpu_ocl_dev, nullptr, &err);
#else
    cl_command_queue cpu_ocl_queue
            = clCreateCommandQueue(cpu_ocl_ctx, cpu_ocl_dev, 0, &err);
#endif
    TEST_OCL_CHECK(err);

    dnnl_stream_t stream;
    dnnl_status_t status
            = dnnl_ocl_interop_stream_create(&stream, eng, cpu_ocl_queue);
    ASSERT_EQ(status, dnnl_invalid_arguments);

    TEST_OCL_CHECK(clReleaseCommandQueue(cpu_ocl_queue));
}

TEST_F(ocl_stream_test_cpp_t, InteropIncompatibleQueueCpp) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    cl_device_id cpu_ocl_dev = find_ocl_device(CL_DEVICE_TYPE_CPU);
    SKIP_IF(!cpu_ocl_dev, "OpenCL CPU devices not found.");

    cl_int err;
    cl_context cpu_ocl_ctx
            = clCreateContext(nullptr, 1, &cpu_ocl_dev, nullptr, nullptr, &err);
    TEST_OCL_CHECK(err);

#ifdef CL_VERSION_2_0
    cl_command_queue cpu_ocl_queue = clCreateCommandQueueWithProperties(
            cpu_ocl_ctx, cpu_ocl_dev, nullptr, &err);
#else
    cl_command_queue cpu_ocl_queue
            = clCreateCommandQueue(cpu_ocl_ctx, cpu_ocl_dev, 0, &err);
#endif
    TEST_OCL_CHECK(err);

    catch_expected_failures(
            [&] { ocl_interop::make_stream(eng, cpu_ocl_queue); }, true,
            dnnl_invalid_arguments);

    TEST_OCL_CHECK(clReleaseCommandQueue(cpu_ocl_queue));
}

TEST_F(ocl_stream_test_cpp_t, out_of_order_queue) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    cl_int err;

#ifdef CL_VERSION_2_0
    cl_queue_properties properties[]
            = {CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
    cl_command_queue ocl_queue = clCreateCommandQueueWithProperties(
            ocl_ctx, ocl_dev, properties, &err);
#else
    cl_command_queue_properties properties
            = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cl_command_queue ocl_queue
            = clCreateCommandQueue(ocl_ctx, ocl_dev, properties, &err);
#endif
    TEST_OCL_CHECK(err);

    memory::dims dims = {2, 3, 4, 5};
    memory::desc mem_d(dims, memory::data_type::f32, memory::format_tag::nchw);

    auto eltwise_pd = eltwise_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::eltwise_relu, mem_d, mem_d, 0.0f);
    auto eltwise = eltwise_forward(eltwise_pd);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::buffer);

    auto stream = ocl_interop::make_stream(eng, ocl_queue);

    const int size = std::accumulate(dims.begin(), dims.end(),
            (dnnl::memory::dim)1, std::multiplies<dnnl::memory::dim>());

    std::vector<float> host_data_src(size);
    for (int i = 0; i < size; i++)
        host_data_src[i] = static_cast<float>(i - size / 2);

    cl_event write_buffer_event;
    TEST_OCL_CHECK(
            clEnqueueWriteBuffer(ocl_queue, ocl_interop::get_mem_object(mem),
                    /* blocking */ CL_FALSE, 0, size * sizeof(float),
                    host_data_src.data(), 0, nullptr, &write_buffer_event));

    cl_event eltwise_event = ocl_interop::execute(eltwise, stream,
            {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}}, {write_buffer_event});

    // Check results.
    std::vector<float> host_data_dst(size, -1);
    cl_event read_buffer_event;
    TEST_OCL_CHECK(clEnqueueReadBuffer(ocl_queue,
            ocl_interop::get_mem_object(mem),
            /* blocking */ CL_FALSE, 0, size * sizeof(float),
            host_data_dst.data(), 1, &eltwise_event, &read_buffer_event));
    TEST_OCL_CHECK(clWaitForEvents(1, &read_buffer_event));

    for (int i = 0; i < size; i++) {
        float exp_value
                = static_cast<float>((i - size / 2) <= 0 ? 0 : (i - size / 2));
        EXPECT_EQ(host_data_dst[i], exp_value);
    }

    TEST_OCL_CHECK(clReleaseEvent(read_buffer_event));
    TEST_OCL_CHECK(clReleaseEvent(write_buffer_event));
    TEST_OCL_CHECK(clReleaseEvent(eltwise_event));
    TEST_OCL_CHECK(clReleaseCommandQueue(ocl_queue));
}

#ifdef DNNL_EXPERIMENTAL_PROFILING
TEST_F(ocl_stream_test_cpp_t, TestProfilingAPIUserQueue) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    memory::dims dims = {2, 3, 4, 5};
    memory::desc md(dims, memory::data_type::f32, memory::format_tag::nchw);

    auto eltwise_pd = eltwise_forward::primitive_desc(
            eng, prop_kind::forward, algorithm::eltwise_relu, md, md, 0.0f);
    auto eltwise = eltwise_forward(eltwise_pd);
    auto mem = memory(md, eng);

    cl_int err;
#ifdef CL_VERSION_2_0
    cl_queue_properties properties[]
            = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue ocl_queue = clCreateCommandQueueWithProperties(
            ocl_ctx, ocl_dev, properties, &err);
#else
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue ocl_queue
            = clCreateCommandQueue(ocl_ctx, ocl_dev, properties, &err);
#endif

    TEST_OCL_CHECK(err);

    auto stream = ocl_interop::make_stream(eng, ocl_queue);
    TEST_OCL_CHECK(clReleaseCommandQueue(ocl_queue));

    // Reset profiler's state.
    ASSERT_NO_THROW(reset_profiling(stream));

    eltwise.execute(stream, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
    stream.wait();

    // Query profiling data.
    std::vector<uint64_t> nsec;
    ASSERT_NO_THROW(
            nsec = get_profiling_data(stream, profiling_data_kind::time));
    ASSERT_FALSE(nsec.empty());

    // Reset profiler's state.
    ASSERT_NO_THROW(reset_profiling(stream));
    // Test that the profiler's state was reset.
    ASSERT_NO_THROW(
            nsec = get_profiling_data(stream, profiling_data_kind::time));
    ASSERT_TRUE(nsec.empty());
}

TEST_F(ocl_stream_test_cpp_t, TestProfilingAPILibraryQueue) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    memory::dims dims = {2, 3, 4, 5};
    memory::desc md(dims, memory::data_type::f32, memory::format_tag::nchw);

    auto eltwise_pd = eltwise_forward::primitive_desc(
            eng, prop_kind::forward, algorithm::eltwise_relu, md, md, 0.0f);
    auto eltwise = eltwise_forward(eltwise_pd);
    auto mem = memory(md, eng);

    auto stream = dnnl::stream(eng, stream::flags::profiling);

    // Reset profiler's state.
    ASSERT_NO_THROW(reset_profiling(stream));

    eltwise.execute(stream, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
    stream.wait();

    // Query profiling data.
    std::vector<uint64_t> nsec;
    ASSERT_NO_THROW(
            nsec = get_profiling_data(stream, profiling_data_kind::time));
    ASSERT_FALSE(nsec.empty());

    // Reset profiler's state.
    ASSERT_NO_THROW(reset_profiling(stream));
    // Test that the profiler's state was reset.
    ASSERT_NO_THROW(
            nsec = get_profiling_data(stream, profiling_data_kind::time));
    ASSERT_TRUE(nsec.empty());
}

TEST_F(ocl_stream_test_cpp_t, TestProfilingAPIOutOfOrderQueue) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");
    cl_int err;
#ifdef CL_VERSION_2_0
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES,
            CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            0};
    cl_command_queue ocl_queue = clCreateCommandQueueWithProperties(
            ocl_ctx, ocl_dev, properties, &err);
#else
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE
            | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cl_command_queue ocl_queue
            = clCreateCommandQueue(ocl_ctx, ocl_dev, properties, &err);
#endif
    TEST_OCL_CHECK(err);

    // Create stream with a user provided queue.
    ASSERT_ANY_THROW(auto stream = ocl_interop::make_stream(eng, ocl_queue));
    TEST_OCL_CHECK(clReleaseCommandQueue(ocl_queue));
    // Create a stream with a library provided queue.
    ASSERT_ANY_THROW(
            auto stream = dnnl::stream(eng,
                    stream::flags::out_of_order | stream ::flags::profiling));
}

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
TEST_F(ocl_stream_test_cpp_t, TestProfilingAPICPU) {
    auto eng = engine(engine::kind::cpu, 0);
    ASSERT_ANY_THROW(auto stream = dnnl::stream(eng, stream::flags::profiling));
}
#endif

#endif

#ifndef DNNL_EXPERIMENTAL_PROFILING
extern "C" dnnl_status_t dnnl_reset_profiling(dnnl_stream_t stream);
#endif

TEST_F(ocl_stream_test_cpp_t, TestProfilingAPIDisabledAndEnabled) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    memory::dims dims = {2, 3, 4, 5};
    memory::desc md(dims, memory::data_type::f32, memory::format_tag::nchw);

    auto eltwise_pd = eltwise_forward::primitive_desc(
            eng, prop_kind::forward, algorithm::eltwise_relu, md, md, 0.0f);
    auto eltwise = eltwise_forward(eltwise_pd);
    auto mem = memory(md, eng);

    cl_int err;
#ifdef CL_VERSION_2_0
    cl_queue_properties properties[]
            = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue ocl_queue = clCreateCommandQueueWithProperties(
            ocl_ctx, ocl_dev, properties, &err);
#else
    cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue ocl_queue
            = clCreateCommandQueue(ocl_ctx, ocl_dev, properties, &err);
#endif

    TEST_OCL_CHECK(err);

    auto stream = ocl_interop::make_stream(eng, ocl_queue);
    TEST_OCL_CHECK(clReleaseCommandQueue(ocl_queue));

    eltwise.execute(stream, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
    stream.wait();

    auto st = dnnl_reset_profiling(stream.get());

// If the experimental profiling API is not enabled then the library should not
// enable profiling regardless of the queue's properties.
#ifdef DNNL_EXPERIMENTAL_PROFILING
    EXPECT_EQ(st, dnnl_success);
#else
    EXPECT_EQ(st, dnnl_invalid_arguments);
#endif
}

} // namespace dnnl
