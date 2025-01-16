/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#include "src/xpu/ocl/usm_utils.hpp"

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

#include <cstdint>
#include <vector>

using namespace dnnl::impl::xpu::ocl;

namespace dnnl {

namespace {
void fill_data(void *usm_ptr, memory::dim n, const engine &eng) {
    auto alloc_kind = usm::get_pointer_type(eng.get(), usm_ptr);
    if (alloc_kind == usm::kind_t::host || alloc_kind == usm::kind_t::shared) {
        for (int i = 0; i < n; i++)
            ((float *)usm_ptr)[i] = float(i);
    } else {
        std::vector<float> host_ptr(n);
        for (int i = 0; i < n; i++)
            host_ptr[i] = float(i);

        auto s = stream(eng);
        usm::memcpy(s.get(), usm_ptr, host_ptr.data(), n * sizeof(float));
        s.wait();
    }
}

using usm_unique_ptr_t = std::unique_ptr<void, std::function<void(void *)>>;
usm_unique_ptr_t allocate_usm(size_t size, const engine &eng) {
    return usm_unique_ptr_t(usm::malloc_shared(eng.get(), size),
            [&](void *ptr) { usm::free(eng.get(), ptr); });
}

} // namespace

class ocl_memory_usm_test_t : public ::testing::Test {};

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, Constructor) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto ptr = allocate_usm(sizeof(float) * n, eng);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::usm, ptr.get());

    ASSERT_EQ(ptr.get(), mem.get_data_handle());
    ASSERT_EQ(ocl_interop::memory_kind::usm, ocl_interop::get_memory_kind(mem));

    {
        for (int i = 0; i < n; i++) {
            ((float *)ptr.get())[i] = float(i);
        }
    }

    {
        float *ptr_f32 = (float *)mem.get_data_handle();
        GTEST_EXPECT_NE(ptr_f32, nullptr);
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(ptr_f32[i], float(i));
        }
    }
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, ConstructorNone) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::desc mem_d({0}, memory::data_type::f32, memory::format_tag::x);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::usm, DNNL_MEMORY_NONE);

    ASSERT_EQ(nullptr, mem.get_data_handle());
    ASSERT_EQ(ocl_interop::memory_kind::usm, ocl_interop::get_memory_kind(mem));
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, ConstructorAllocate) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::usm, DNNL_MEMORY_ALLOCATE);

    ASSERT_EQ(ocl_interop::memory_kind::usm, ocl_interop::get_memory_kind(mem));

    void *ptr = mem.get_data_handle();
    GTEST_EXPECT_NE(ptr, nullptr);
    fill_data(ptr, n, eng);

    float *mapped_ptr = mem.map_data<float>();
    GTEST_EXPECT_NE(mapped_ptr, nullptr);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, DefaultConstructor) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::usm);

    ASSERT_EQ(ocl_interop::memory_kind::usm, ocl_interop::get_memory_kind(mem));

    void *ptr = mem.get_data_handle();
    GTEST_EXPECT_NE(ptr, nullptr);
    fill_data(ptr, n, eng);

    float *mapped_ptr = mem.map_data<float>();
    GTEST_EXPECT_NE(mapped_ptr, nullptr);
    for (int i = 0; i < n; i++) {
        ASSERT_EQ(mapped_ptr[i], float(i));
    }
    mem.unmap_data(mapped_ptr);
}

template <typename AllocFuncT, typename FreeFuncT>
void test_usm_map_unmap(
        const AllocFuncT &alloc_func, const FreeFuncT &free_func) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    auto *ptr = alloc_func(eng.get(), mem_d.get_size());
    ASSERT_NE(ptr, nullptr);

    auto mem = ocl_interop::make_memory(
            mem_d, eng, ocl_interop::memory_kind::usm, ptr);

    ASSERT_EQ(ocl_interop::memory_kind::usm, ocl_interop::get_memory_kind(mem));

    {
        float *mapped_ptr = mem.template map_data<float>();
        GTEST_EXPECT_NE(mapped_ptr, nullptr);
        fill_data(mapped_ptr, n, eng);
        mem.unmap_data(mapped_ptr);
    }

    {
        float *mapped_ptr = mem.template map_data<float>();
        GTEST_EXPECT_NE(mapped_ptr, nullptr);
        for (int i = 0; i < n; i++) {
            ASSERT_EQ(mapped_ptr[i], float(i));
        }
        mem.unmap_data(mapped_ptr);
    }
    free_func(eng.get(), ptr);
}

/// This test checks if passing system allocated memory(e.g. using malloc)
/// will throw if passed into the make_memory
TEST(ocl_memory_usm_test, ErrorMakeMemoryUsingSystemMemory) {
    SKIP_IF(engine::get_count(engine::kind::gpu) == 0, "Engine not found.");

    engine eng(engine::kind::gpu, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    std::vector<float> system_buf(n);
    EXPECT_THROW(memory mem = ocl_interop::make_memory(mem_d, eng,
                         ocl_interop::memory_kind::usm, system_buf.data()),
            dnnl::error);
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, DeviceMapUnmap) {
    test_usm_map_unmap(dnnl::impl::xpu::ocl::usm::malloc_device,
            dnnl::impl::xpu::ocl::usm::free);
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, SharedMapUnmap) {
    test_usm_map_unmap(dnnl::impl::xpu::ocl::usm::malloc_shared,
            dnnl::impl::xpu::ocl::usm::free);
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, TestSparseMemoryCreation) {
    engine eng(engine::kind::gpu, 0);
    const int nnz = 12;
    memory::desc md;

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({64, 128}, memory::data_type::f32,
                            nnz, memory::data_type::s32));

    memory mem;
    // Default memory constructor.
    EXPECT_NO_THROW(mem = memory(md, eng));
    // Default interop API to create a memory object.
    EXPECT_NO_THROW(mem
            = ocl_interop::make_memory(md, eng, ocl_interop::memory_kind::usm));
    // User provided buffers.
    auto ocl_values = allocate_usm(md.get_size(0), eng);
    ASSERT_NE(ocl_values, nullptr);

    auto ocl_row_indices = allocate_usm(md.get_size(1), eng);
    ASSERT_NE(ocl_row_indices, nullptr);

    auto ocl_col_indices = allocate_usm(md.get_size(2), eng);
    ASSERT_NE(ocl_col_indices, nullptr);

    EXPECT_NO_THROW(mem
            = ocl_interop::make_memory(md, eng, ocl_interop::memory_kind::usm,
                    {ocl_values.get(), ocl_row_indices.get(),
                            ocl_col_indices.get()}));

    ASSERT_NO_THROW(mem.set_data_handle(nullptr, 0));
    ASSERT_NO_THROW(mem.set_data_handle(nullptr, 1));
    ASSERT_NO_THROW(mem.set_data_handle(nullptr, 2));

    ASSERT_EQ(mem.get_data_handle(0), nullptr);
    ASSERT_EQ(mem.get_data_handle(1), nullptr);
    ASSERT_EQ(mem.get_data_handle(2), nullptr);
}

HANDLE_EXCEPTIONS_FOR_TEST(ocl_memory_usm_test_t, TestSparseMemoryMapUnmap) {
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

    // User provided buffers.
    auto ocl_values = allocate_usm(md.get_size(0), eng);
    ASSERT_NE(ocl_values, nullptr);

    auto ocl_row_indices = allocate_usm(md.get_size(1), eng);
    ASSERT_NE(ocl_row_indices, nullptr);

    auto ocl_col_indices = allocate_usm(md.get_size(2), eng);
    ASSERT_NE(ocl_col_indices, nullptr);

    auto s = stream(eng);
    usm::memcpy(s.get(), ocl_values.get(), coo_values.data(), md.get_size(0));
    usm::memcpy(
            s.get(), ocl_row_indices.get(), row_indices.data(), md.get_size(1));
    usm::memcpy(
            s.get(), ocl_col_indices.get(), col_indices.data(), md.get_size(2));
    s.wait();

    memory coo_mem;
    EXPECT_NO_THROW(coo_mem
            = ocl_interop::make_memory(md, eng, ocl_interop::memory_kind::usm,
                    {ocl_values.get(), ocl_row_indices.get(),
                            ocl_col_indices.get()}));

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
