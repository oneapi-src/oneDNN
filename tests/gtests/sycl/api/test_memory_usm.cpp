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

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"

#include <cstdint>
#include <vector>

using namespace sycl;

namespace dnnl {

class fill_kernel;

class sycl_memory_usm_test : public ::testing::TestWithParam<engine::kind> {
    using usm_unique_ptr_t = std::unique_ptr<void, std::function<void(void *)>>;

protected:
    static void fill_data(void *usm_ptr, memory::dim n, const engine &eng) {
        auto alloc_kind = ::sycl::get_pointer_type(
                usm_ptr, sycl_interop::get_context(eng));
        if (alloc_kind == ::sycl::usm::alloc::host
                || alloc_kind == ::sycl::usm::alloc::shared) {
            for (int i = 0; i < n; i++)
                ((float *)usm_ptr)[i] = float(i);
        } else {
            std::vector<float> host_ptr(n);
            for (int i = 0; i < n; i++)
                host_ptr[i] = float(i);

            auto q = sycl_interop::get_queue(stream(eng));
            q.memcpy(usm_ptr, host_ptr.data(), n * sizeof(float)).wait();
        }
    }

    usm_unique_ptr_t allocate_usm(size_t size, const ::sycl::device &dev,
            const ::sycl::context &ctx) {
        return usm_unique_ptr_t(::sycl::malloc_shared(size, dev, ctx),
                [&](void *ptr) { ::sycl::free(ptr, ctx); });
    }
};

TEST_P(sycl_memory_usm_test, Constructor) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng_kind == engine::kind::cpu) {
        int dummy_ptr;
        EXPECT_ANY_THROW(sycl_interop::make_memory(
                mem_d, eng, sycl_interop::memory_kind::usm, &dummy_ptr));
        return;
    }
#endif
    auto dev = sycl_interop::get_device(eng);
    auto ctx = sycl_interop::get_context(eng);
    auto ptr = allocate_usm(sizeof(float) * n, dev, ctx);

    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::usm, ptr.get());

    ASSERT_EQ(ptr.get(), mem.get_data_handle());

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

TEST_P(sycl_memory_usm_test, ConstructorNone) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::desc mem_d({0}, memory::data_type::f32, memory::format_tag::x);

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng_kind == engine::kind::cpu) {
        EXPECT_ANY_THROW(sycl_interop::make_memory(
                mem_d, eng, sycl_interop::memory_kind::usm, DNNL_MEMORY_NONE));
        return;
    }
#endif

    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::usm, DNNL_MEMORY_NONE);

    ASSERT_EQ(nullptr, mem.get_data_handle());
}

TEST_P(sycl_memory_usm_test, ConstructorAllocate) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng_kind == engine::kind::cpu) {
        EXPECT_ANY_THROW(sycl_interop::make_memory(mem_d, eng,
                sycl_interop::memory_kind::usm, DNNL_MEMORY_ALLOCATE));
        return;
    }
#endif

    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::usm, DNNL_MEMORY_ALLOCATE);

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

TEST_P(sycl_memory_usm_test, DefaultConstructor) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    if (eng_kind == engine::kind::cpu) {
        EXPECT_ANY_THROW(sycl_interop::make_memory(
                mem_d, eng, sycl_interop::memory_kind::usm));
        return;
    }
#endif
    auto mem = sycl_interop::make_memory(
            mem_d, eng, sycl_interop::memory_kind::usm);

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

/// This test checks if passing system allocated memory(e.g. using malloc)
/// will throw if passed into the make_memory
TEST_P(sycl_memory_usm_test, ErrorMakeMemoryUsingSystemMemory) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    std::vector<float> system_buf(n);
    sycl::device device;
    try {
        device = sycl_interop::get_device(eng);
    } catch (const error &err) {
        if (err.status == dnnl_status_t::dnnl_invalid_arguments)
            GTEST_SKIP() << "The selected device is not using a sycl runtime";
        else
            GTEST_FAIL() << "Failed to create a device from the engine.";
    }
    if (device.has(::sycl::aspect::usm_system_allocations)) {
        memory mem = sycl_interop::make_memory(
                mem_d, eng, sycl_interop::memory_kind::usm, system_buf.data());
    } else {
        EXPECT_THROW(memory mem = sycl_interop::make_memory(mem_d, eng,
                             sycl_interop::memory_kind::usm, system_buf.data()),
                dnnl::error);
    }
}

/// This test checks if passing system allocated memory(e.g. using malloc)
/// will throw if passed into the make_memory
TEST_P(sycl_memory_usm_test, ErrorMemoryConstructorUsingSystemMemory) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
    memory::dim n = 100;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::x);

    std::vector<float> system_buf(n);
    sycl::device device;
    try {
        device = sycl_interop::get_device(eng);
    } catch (const error &err) {
        if (err.status == dnnl_status_t::dnnl_invalid_arguments)
            GTEST_SKIP() << "The selected device is not using a sycl runtime";
        else
            GTEST_FAIL() << "Failed to create a device from the engine.";
    }
    if (device.has(::sycl::aspect::usm_system_allocations)) {
        memory mem(mem_d, eng, system_buf.data());
    } else {
        EXPECT_THROW(memory mem(mem_d, eng, system_buf.data()), dnnl::error);
    }
}

TEST_P(sycl_memory_usm_test, MemoryOutOfScope) {
    engine::kind eng_kind = GetParam();
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(eng_kind == engine::kind::cpu,
            "Skip this test for classic CPU runtime");
#endif
    engine eng(eng_kind, 0);

    memory::dim n = 2048;
    memory::desc mem_d({n}, memory::data_type::f32, memory::format_tag::a);

    auto dev = sycl_interop::get_device(eng);
    auto ctx = sycl_interop::get_context(eng);
    auto ptr = allocate_usm(sizeof(float) * n, dev, ctx);

    auto eltwise_pd = eltwise_forward::primitive_desc(eng, prop_kind::forward,
            algorithm::eltwise_relu, mem_d, mem_d, 0.0f);
    auto eltwise = eltwise_forward(eltwise_pd);

    stream s(eng);
    {
        memory mem = sycl_interop::make_memory(
                mem_d, eng, sycl_interop::memory_kind::usm, ptr.get());
        eltwise.execute(s, {{DNNL_ARG_SRC, mem}, {DNNL_ARG_DST, mem}});
    }
    s.wait();
}

TEST_P(sycl_memory_usm_test, TestSparseMemoryCreation) {
    engine::kind eng_kind = GetParam();

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(eng_kind == engine::kind::cpu,
            "Skip this test for classic CPU runtime");
#endif
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);
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

    // Default interop API to create a memory object.
    EXPECT_NO_THROW(mem = sycl_interop::make_memory(
                            md, eng, sycl_interop::memory_kind::usm));
    // Memory object is expected to have 3 handles.
    EXPECT_NO_THROW(mem.get_data_handle(0));
    EXPECT_NO_THROW(mem.get_data_handle(1));
    EXPECT_NO_THROW(mem.get_data_handle(2));

    // User provided buffers.
    auto dev = sycl_interop::get_device(eng);
    auto ctx = sycl_interop::get_context(eng);

    auto usm_values = allocate_usm(md.get_size(0), dev, ctx);
    auto usm_row_indices = allocate_usm(md.get_size(1), dev, ctx);
    auto usm_col_indices = allocate_usm(md.get_size(2), dev, ctx);

    EXPECT_NO_THROW(mem
            = sycl_interop::make_memory(md, eng, sycl_interop::memory_kind::usm,
                    {usm_values.get(), usm_row_indices.get(),
                            usm_col_indices.get()}));

    ASSERT_EQ(mem.get_data_handle(0), usm_values.get());
    ASSERT_EQ(mem.get_data_handle(1), usm_row_indices.get());
    ASSERT_EQ(mem.get_data_handle(2), usm_col_indices.get());
}

TEST_P(sycl_memory_usm_test, TestSparseMemoryMapUnmap) {
    engine::kind eng_kind = GetParam();

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_SYCL
    SKIP_IF(eng_kind == engine::kind::cpu,
            "Skip this test for classic CPU runtime");
#endif
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine not found.");

    engine eng(eng_kind, 0);

    const int nnz = 2;
    memory::desc md;

    // COO.
    ASSERT_NO_THROW(md = memory::desc::coo({2, 2}, memory::data_type::f32, nnz,
                            memory::data_type::s32));

    // User provided buffers.
    std::vector<float> coo_values = {1.5, 2.5};
    std::vector<int> row_indices = {0, 1};
    std::vector<int> col_indices = {0, 1};

    auto dev = sycl_interop::get_device(eng);
    auto ctx = sycl_interop::get_context(eng);
    auto usm_values = allocate_usm(md.get_size(0), dev, ctx);
    auto usm_row_indices = allocate_usm(md.get_size(1), dev, ctx);
    auto usm_col_indices = allocate_usm(md.get_size(2), dev, ctx);

    for (size_t i = 0; i < coo_values.size(); i++)
        static_cast<float *>(usm_values.get())[i] = coo_values[i];
    for (size_t i = 0; i < row_indices.size(); i++)
        static_cast<int *>(usm_row_indices.get())[i] = row_indices[i];
    for (size_t i = 0; i < col_indices.size(); i++)
        static_cast<int *>(usm_col_indices.get())[i] = col_indices[i];

    memory coo_mem;
    EXPECT_NO_THROW(coo_mem
            = sycl_interop::make_memory(md, eng, sycl_interop::memory_kind::usm,
                    {usm_values.get(), usm_row_indices.get(),
                            usm_col_indices.get()}));

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

namespace {
struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        return to_string(static_cast<dnnl_engine_kind_t>(info.param));
    }
};
} // namespace

INSTANTIATE_TEST_SUITE_P(Simple, sycl_memory_usm_test,
        ::testing::Values(engine::kind::cpu, engine::kind::gpu),
        PrintToStringParamName());

} // namespace dnnl
