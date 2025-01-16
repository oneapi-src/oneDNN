/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "oneapi/dnnl/dnnl_ocl.h"
#include "oneapi/dnnl/dnnl_ocl.hpp"

#include <algorithm>
#include <memory>
#include <vector>
#include <CL/cl.h>

namespace dnnl {

class ocl_memory_buffer_test_c_t : public ::testing::Test {
protected:
    HANDLE_EXCEPTIONS_FOR_TEST_SETUP() {
        if (!find_ocl_device(CL_DEVICE_TYPE_GPU)) { return; }

        DNNL_CHECK(dnnl_engine_create(&engine, dnnl_gpu, 0));
        DNNL_CHECK(dnnl_ocl_interop_engine_get_context(engine, &ocl_ctx));

        DNNL_CHECK(dnnl_memory_desc_create_with_tag(
                &memory_d, dim, dims, dnnl_f32, dnnl_nchw));
        DNNL_CHECK(dnnl_memory_create(
                &memory, memory_d, engine, DNNL_MEMORY_NONE));
        DNNL_CHECK(dnnl_memory_desc_destroy(memory_d));
    }

    void TearDown() override {
        if (memory) { DNNL_CHECK(dnnl_memory_destroy(memory)); }
        if (engine) { DNNL_CHECK(dnnl_engine_destroy(engine)); }
    }

    dnnl_engine_t engine = nullptr;
    cl_context ocl_ctx = nullptr;

    static const int dim = 4;
    static const dnnl_dim_t N = 2;
    static const dnnl_dim_t C = 3;
    static const dnnl_dim_t H = 4;
    static const dnnl_dim_t W = 5;
    dnnl_dims_t dims = {N, C, H, W};

    dnnl_memory_desc_t memory_d;
    dnnl_memory_t memory = nullptr;
};

class ocl_memory_buffer_test_cpp_t : public ::testing::Test {};

// Use runtime agnostic API to create memory. The created memory expected to
// contain an OpenCL buffer.
HANDLE_EXCEPTIONS_FOR_TEST_F(ocl_memory_buffer_test_c_t, BasicInteropC) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    dnnl_ocl_interop_memory_kind_t memory_kind;
    DNNL_CHECK(dnnl_ocl_interop_memory_get_memory_kind(memory, &memory_kind));
    ASSERT_EQ(dnnl_ocl_interop_buffer, memory_kind);

    cl_mem ocl_mem;
    DNNL_CHECK(dnnl_ocl_interop_memory_get_mem_object(memory, &ocl_mem));
    ASSERT_EQ(ocl_mem, nullptr);

    cl_int err;
    cl_mem interop_ocl_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE,
            sizeof(float) * N * C * H * W, nullptr, &err);
    TEST_OCL_CHECK(err);

    DNNL_CHECK(dnnl_ocl_interop_memory_set_mem_object(memory, interop_ocl_mem));

    DNNL_CHECK(dnnl_ocl_interop_memory_get_mem_object(memory, &ocl_mem));
    ASSERT_EQ(ocl_mem, interop_ocl_mem);

    DNNL_CHECK(dnnl_memory_destroy(memory));
    memory = nullptr;

    cl_uint ref_count;
    TEST_OCL_CHECK(clGetMemObjectInfo(interop_ocl_mem, CL_MEM_REFERENCE_COUNT,
            sizeof(cl_uint), &ref_count, nullptr));
    int i_ref_count = int(ref_count);
    ASSERT_EQ(i_ref_count, 1);

    TEST_OCL_CHECK(clReleaseMemObject(interop_ocl_mem));
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_buffer_test_cpp_t, BasicInteropCpp) {
    SKIP_IF(!find_ocl_device(CL_DEVICE_TYPE_GPU),
            "OpenCL GPU devices not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dims tz = {4, 4, 4, 4};

    cl_context ocl_ctx = ocl_interop::get_context(eng);

    cl_int err;
    cl_mem interop_ocl_mem = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE,
            sizeof(float) * tz[0] * tz[1] * tz[2] * tz[3], nullptr, &err);
    TEST_OCL_CHECK(err);

    {
        memory::desc mem_d(
                tz, memory::data_type::f32, memory::format_tag::nchw);
        auto mem = test::make_memory(mem_d, eng);

        ASSERT_EQ(ocl_interop::memory_kind::buffer,
                ocl_interop::get_memory_kind(mem));

        cl_mem ocl_mem = ocl_interop::get_mem_object(mem);
        ASSERT_NE(ocl_mem, nullptr);

        ocl_interop::set_mem_object(mem, interop_ocl_mem);

        ocl_mem = ocl_interop::get_mem_object(mem);
        ASSERT_EQ(ocl_mem, interop_ocl_mem);
    }

    cl_uint ref_count;
    TEST_OCL_CHECK(clGetMemObjectInfo(interop_ocl_mem, CL_MEM_REFERENCE_COUNT,
            sizeof(cl_uint), &ref_count, nullptr));
    int i_ref_count = int(ref_count);
    ASSERT_EQ(i_ref_count, 1);

    TEST_OCL_CHECK(clReleaseMemObject(interop_ocl_mem));
}

// Use interop API to create memory.
HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_buffer_test_cpp_t, BasicInteropCtor) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dims tz = {4, 4, 4, 4};

    size_t sz = size_t(tz[0]) * tz[1] * tz[2] * tz[3];

    cl_context ocl_ctx = ocl_interop::get_context(eng);

    cl_int err;
    cl_mem ocl_mem = clCreateBuffer(
            ocl_ctx, CL_MEM_READ_WRITE, sizeof(float) * sz, nullptr, &err);

    memory::desc mem_d(tz, memory::data_type::f32, memory::format_tag::nchw);

    cl_mem ocl_mem_from_mem;
    {
        auto mem = ocl_interop::make_memory(mem_d, eng, ocl_mem);
        ocl_mem_from_mem = ocl_interop::get_mem_object(mem);
    }

    ASSERT_EQ(ocl_mem, ocl_mem_from_mem);
    TEST_OCL_CHECK(err);
    cl_uint ref_count;
    TEST_OCL_CHECK(clGetMemObjectInfo(ocl_mem, CL_MEM_REFERENCE_COUNT,
            sizeof(cl_uint), &ref_count, nullptr));
    int i_ref_count = int(ref_count);
    ASSERT_EQ(i_ref_count, 1);

    TEST_OCL_CHECK(clReleaseMemObject(ocl_mem));
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_buffer_test_cpp_t, BufferMapUnmap) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    cl_context ocl_ctx = ocl_interop::get_context(eng);
    cl_int err;
    cl_mem ocl_mem = clCreateBuffer(
            ocl_ctx, CL_MEM_READ_WRITE, mem_d.get_size(), nullptr, &err);
    TEST_OCL_CHECK(err);
    ASSERT_NE(ocl_mem, nullptr);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::buffer, ocl_mem);

    ASSERT_EQ(ocl_interop::memory_kind::buffer,
            ocl_interop::get_memory_kind(mem));

    {
        float *mapped_ptr = mem.template map_data<float>();
        GTEST_EXPECT_NE(mapped_ptr, nullptr);
        for (int i = 0; i < n; i++) {
            mapped_ptr[i] = (float)i;
        }
        mem.unmap_data(mapped_ptr);
    }

    {
        float *mapped_ptr = mem.template map_data<float>();
        GTEST_EXPECT_NE(mapped_ptr, nullptr);
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(mapped_ptr[i], (float)i);
        }
        mem.unmap_data(mapped_ptr);
    }
    TEST_OCL_CHECK(clReleaseMemObject(ocl_mem));
}

HANDLE_EXCEPTIONS_FOR_TEST(
        ocl_memory_buffer_test_cpp_t, TestSparseMemoryCreation) {
    engine eng(engine::kind::gpu, 0);
    const int nnz = 12;
    memory::desc md;

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({64, 128}, memory::data_type::f32,
                            nnz, memory::data_type::s32));

    memory mem;
    // Default memory constructor.
    EXPECT_NO_THROW(mem = memory(md, eng));
    // Memory object is expected to have 3 handles.
    EXPECT_NO_THROW(mem.get_data_handle(0));
    EXPECT_NO_THROW(mem.get_data_handle(1));
    EXPECT_NO_THROW(mem.get_data_handle(2));

    // Default interop AI to create a memory object.
    EXPECT_NO_THROW(mem = ocl_interop::make_memory(
                            md, eng, ocl_interop::memory_kind::buffer));
    // Memory object is expected to have 3 handles.
    EXPECT_NO_THROW(mem.get_data_handle(0));
    EXPECT_NO_THROW(mem.get_data_handle(1));
    EXPECT_NO_THROW(mem.get_data_handle(2));

    // User provided buffers.
    cl_context ocl_ctx = ocl_interop::get_context(eng);
    cl_int err;
    cl_mem ocl_values = clCreateBuffer(
            ocl_ctx, CL_MEM_READ_WRITE, md.get_size(0), nullptr, &err);
    TEST_OCL_CHECK(err);
    ASSERT_NE(ocl_values, nullptr);

    cl_mem ocl_row_indices = clCreateBuffer(
            ocl_ctx, CL_MEM_READ_WRITE, md.get_size(1), nullptr, &err);
    TEST_OCL_CHECK(err);
    ASSERT_NE(ocl_row_indices, nullptr);

    cl_mem ocl_col_indices = clCreateBuffer(
            ocl_ctx, CL_MEM_READ_WRITE, md.get_size(2), nullptr, &err);
    TEST_OCL_CHECK(err);
    ASSERT_NE(ocl_col_indices, nullptr);

    EXPECT_NO_THROW(mem = ocl_interop::make_memory(md, eng,
                            {ocl_values, ocl_row_indices, ocl_col_indices}));

    ASSERT_NO_THROW(mem.set_data_handle(nullptr, 0));
    ASSERT_NO_THROW(mem.set_data_handle(nullptr, 1));
    ASSERT_NO_THROW(mem.set_data_handle(nullptr, 2));

    ASSERT_EQ(mem.get_data_handle(0), nullptr);
    ASSERT_EQ(mem.get_data_handle(1), nullptr);
    ASSERT_EQ(mem.get_data_handle(2), nullptr);
}

HANDLE_EXCEPTIONS_FOR_TEST(
        ocl_memory_buffer_test_cpp_t, TestSparseMemoryMapUnmap) {
    engine eng(engine::kind::gpu, 0);

    const int nnz = 2;
    memory::desc md;

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({2, 2}, memory::data_type::f32, nnz,
                            memory::data_type::s32));

    // User provided buffers.
    std::vector<float> coo_values = {1.5, 2.5};
    std::vector<int> row_indices = {0, 1};
    std::vector<int> col_indices = {0, 1};

    cl_context ocl_ctx = ocl_interop::get_context(eng);
    cl_int err;
    cl_mem ocl_values
            = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    md.get_size(0), coo_values.data(), &err);
    TEST_OCL_CHECK(err);
    ASSERT_NE(ocl_values, nullptr);

    cl_mem ocl_row_indices
            = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    md.get_size(1), row_indices.data(), &err);
    TEST_OCL_CHECK(err);
    ASSERT_NE(ocl_row_indices, nullptr);

    cl_mem ocl_col_indices
            = clCreateBuffer(ocl_ctx, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    md.get_size(2), col_indices.data(), &err);
    TEST_OCL_CHECK(err);
    ASSERT_NE(ocl_col_indices, nullptr);

    memory coo_mem;
    EXPECT_NO_THROW(coo_mem = ocl_interop::make_memory(md, eng,
                            {ocl_values, ocl_row_indices, ocl_col_indices}));

    float *mapped_coo_values = nullptr;
    int *mapped_row_indices = nullptr;
    int *mapped_col_indices = nullptr;

    ASSERT_NO_THROW(mapped_coo_values = coo_mem.map_data<float>(0));
    ASSERT_NO_THROW(mapped_row_indices = coo_mem.map_data<int>(1));
    ASSERT_NO_THROW(mapped_col_indices = coo_mem.map_data<int>(2));

    for (size_t i = 0; i < coo_values.size(); i++)
        ASSERT_EQ(coo_values[i], mapped_coo_values[i]);

    for (size_t i = 0; i < row_indices.size(); i++)
        ASSERT_EQ(row_indices[i], mapped_row_indices[i]);

    for (size_t i = 0; i < col_indices.size(); i++)
        ASSERT_EQ(col_indices[i], mapped_col_indices[i]);

    ASSERT_NO_THROW(coo_mem.unmap_data(mapped_coo_values, 0));
    ASSERT_NO_THROW(coo_mem.unmap_data(mapped_row_indices, 1));
    ASSERT_NO_THROW(coo_mem.unmap_data(mapped_col_indices, 2));
}

} // namespace dnnl
