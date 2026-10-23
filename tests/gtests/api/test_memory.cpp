/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl.hpp"

#include <limits>
#include <new>

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

namespace dnnl {

class memory_test_c_t : public ::testing::TestWithParam<dnnl_engine_kind_t> {
protected:
    void SetUp() override {
        eng_kind = GetParam();

        if (dnnl_engine_get_count(eng_kind) == 0) return;

        DNNL_CHECK(dnnl_engine_create(&engine, eng_kind, 0));
    }

    void TearDown() override {
        if (engine) { DNNL_CHECK(dnnl_engine_destroy(engine)); }
    }

    dnnl_engine_kind_t eng_kind;
    dnnl_engine_t engine = nullptr;
};

class memory_test_cpp_t : public ::testing::TestWithParam<dnnl_engine_kind_t> {
};

TEST_P(memory_test_c_t, OutOfMemory) {
    SKIP_IF(!engine, "Engine is not found.");
    SKIP_IF(is_sycl_engine(static_cast<engine::kind>(eng_kind)),
            "Do not test C API with SYCL.");

    dnnl_dim_t sz = std::numeric_limits<memory::dim>::max();
    dnnl_dims_t dims = {sz};
    dnnl_memory_desc_t md;
    DNNL_CHECK(dnnl_memory_desc_create_with_tag(&md, 1, dims, dnnl_u8, dnnl_x));

    dnnl_data_type_t data_type;
    DNNL_CHECK(dnnl_memory_desc_query(md, dnnl_query_data_type, &data_type));
    ASSERT_EQ(dnnl_data_type_size(data_type), sizeof(uint8_t));

    dnnl_memory_t mem;
    dnnl_status_t s
            = dnnl_memory_create(&mem, md, engine, DNNL_MEMORY_ALLOCATE);
    ASSERT_EQ(s, dnnl_out_of_memory);

    DNNL_CHECK(dnnl_memory_desc_destroy(md));
}

TEST_P(memory_test_cpp_t, OutOfMemory) {
    dnnl_engine_kind_t eng_kind_c = GetParam();
    engine::kind eng_kind = static_cast<engine::kind>(eng_kind_c);
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine is not found.");

    engine eng(eng_kind, 0);

    bool is_sycl = is_sycl_engine(eng_kind);

    auto sz = std::numeric_limits<memory::dim>::max();
#ifdef DNNL_WITH_SYCL
    if (is_sycl) {
        auto dev = sycl_interop::get_device(eng);
        const memory::dim max_alloc_size
                = dev.get_info<::sycl::info::device::max_mem_alloc_size>();
        sz = (max_alloc_size < sz) ? max_alloc_size + 1 : sz;
    }
#endif

    auto dt = memory::data_type::u8;
    auto tag = memory::format_tag::x;
    memory::desc md({sz}, dt, tag);
    ASSERT_EQ(memory::data_type_size(dt), sizeof(uint8_t));
    try {
        auto mem = test::make_memory(md, eng);
        ASSERT_NE(mem.get_data_handle(), nullptr);
    } catch (const dnnl::error &e) {
        ASSERT_EQ(e.status, dnnl_out_of_memory);
        return;
    } catch (const std::bad_alloc &) {
        // Expect bad_alloc only with SYCL.
        if (is_sycl) return;
        throw;
    }

    // XXX: SYCL does not always throw, even when allocating
    //  > max_mem_alloc_size bytes.
    if (!is_sycl) FAIL() << "Expected exception.";
}

namespace {
struct print_to_string_param_name_t {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        return to_string(info.param);
    }
};

auto all_engine_kinds = ::testing::Values(dnnl_cpu, dnnl_gpu);

} // namespace

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, memory_test_c_t, all_engine_kinds,
        print_to_string_param_name_t());
INSTANTIATE_TEST_SUITE_P(AllEngineKinds, memory_test_cpp_t, all_engine_kinds,
        print_to_string_param_name_t());

/**
 * Test creation and manipulation of host-side scalar memory objects.
 *
 * Validates core supported operations: creation, value access, modification,
 * memory mapping, and engine query.
 */
TEST(c_api_host_scalar_mem, TestSupportedFunctions) {
    dnnl_memory_desc_t scalar_md = nullptr;
    DNNL_CHECK(dnnl_memory_desc_create_host_scalar(&scalar_md, dnnl_f32));

    float scalar_value = 42.0f;
    dnnl_memory_t scalar_mem = nullptr;
    DNNL_CHECK(dnnl_memory_create_host_scalar(
            &scalar_mem, scalar_md, &scalar_value));

    float retrieved_value = 0.0f;
    DNNL_CHECK(dnnl_memory_get_host_scalar_value(scalar_mem, &retrieved_value));
    EXPECT_EQ(retrieved_value, 42.0f);

    float new_value = 84.0f;
    DNNL_CHECK(dnnl_memory_set_host_scalar_value(scalar_mem, &new_value));
    DNNL_CHECK(dnnl_memory_get_host_scalar_value(scalar_mem, &retrieved_value));
    EXPECT_EQ(retrieved_value, 84.0f);

    dnnl_engine_t engine;
    DNNL_CHECK(dnnl_memory_get_engine(scalar_mem, &engine));
    ASSERT_EQ(engine, nullptr);

    void *mapped_value_ptr;
    DNNL_CHECK(dnnl_memory_map_data_v2(scalar_mem, &mapped_value_ptr, 0));
    ASSERT_NE(mapped_value_ptr, nullptr);
    EXPECT_EQ(*(float *)(mapped_value_ptr), 84.0f);

    DNNL_CHECK(dnnl_memory_unmap_data_v2(scalar_mem, mapped_value_ptr, 0));

    DNNL_CHECK(dnnl_memory_desc_destroy(scalar_md));
    DNNL_CHECK(dnnl_memory_destroy(scalar_mem));
}

/**
 * Test host scalar memory with unsupported functions.
 */
TEST(c_api_host_scalar_mem, TestUnsupportedFunctions) {
    dnnl_memory_desc_t scalar_md = nullptr;
    DNNL_CHECK(dnnl_memory_desc_create_host_scalar(&scalar_md, dnnl_f32));

    float scalar_value = 42.0f;
    dnnl_memory_t scalar_mem = nullptr;

    // Ensure that dnnl_memory_create is not allowed with host scalar memory
    dnnl_engine_t engine = nullptr;
    dnnl_engine_create(&engine, dnnl_cpu, 0);
    SKIP_IF(!engine,
            "Engine is not found."); // skip for testing with ONEDNN_CPU_RUNTIME=NONE

    EXPECT_EQ(dnnl_memory_create(&scalar_mem, scalar_md, engine, &scalar_value),
            dnnl_invalid_arguments);

    std::vector<void *> handles(1, &scalar_value);
    EXPECT_EQ(dnnl_memory_create_v2(
                      &scalar_mem, scalar_md, engine, 1, handles.data()),
            dnnl_invalid_arguments);

    dnnl_engine_destroy(engine);

    // Ensure that dnnl_memory_{set,get}_data_handle are not allowed with host scalar memory
    DNNL_CHECK(dnnl_memory_create_host_scalar(
            &scalar_mem, scalar_md, &scalar_value));

    void *handle = nullptr;
    EXPECT_EQ(dnnl_memory_get_data_handle(scalar_mem, &handle),
            dnnl_invalid_arguments);
    EXPECT_EQ(dnnl_memory_get_data_handle_v2(scalar_mem, handles.data(), 1),
            dnnl_invalid_arguments);

    float new_value = 84.0f;
    EXPECT_EQ(dnnl_memory_set_data_handle(scalar_mem, &new_value),
            dnnl_invalid_arguments);
    EXPECT_EQ(dnnl_memory_set_data_handle_v2(scalar_mem, &new_value, 1),
            dnnl_invalid_arguments);
}

TEST(c_api_host_scalar_mem, TestNullPtr) {
    dnnl_memory_t scalar_mem;
    float scalar_value = 42.0f;

    EXPECT_EQ(
            dnnl_memory_create_host_scalar(&scalar_mem, nullptr, &scalar_value),
            dnnl_invalid_arguments);

    dnnl_memory_desc_t scalar_md;
    DNNL_CHECK(dnnl_memory_desc_create_host_scalar(&scalar_md, dnnl_f32));

    EXPECT_EQ(dnnl_memory_create_host_scalar(&scalar_mem, scalar_md, nullptr),
            dnnl_invalid_arguments);

    DNNL_CHECK(dnnl_memory_desc_destroy(scalar_md));
}

class c_api_mem_handle_idx_t
    : public ::testing::TestWithParam<dnnl_engine_kind_t> {};

class cpp_api_mem_handle_idx_t
    : public ::testing::TestWithParam<dnnl_engine_kind_t> {};

TEST_P(c_api_mem_handle_idx_t, TestOutOfRangeIndex) {
    dnnl_engine_kind_t eng_kind_c = GetParam();
    engine::kind eng_kind = static_cast<engine::kind>(eng_kind_c);
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine is not found.");

    dnnl_engine_t engine = nullptr;
    DNNL_CHECK(dnnl_engine_create(&engine, eng_kind_c, 0));

    dnnl_dims_t dims = {4};
    dnnl_memory_desc_t md = nullptr;
    DNNL_CHECK(
            dnnl_memory_desc_create_with_tag(&md, 1, dims, dnnl_f32, dnnl_x));

    dnnl_memory_t mem = nullptr;
    DNNL_CHECK(dnnl_memory_create(&mem, md, engine, DNNL_MEMORY_ALLOCATE));

    void *handle = nullptr;
    EXPECT_EQ(dnnl_memory_get_data_handle_v2(mem, &handle, -1),
            dnnl_invalid_arguments);
    EXPECT_EQ(dnnl_memory_get_data_handle_v2(mem, &handle, 1),
            dnnl_invalid_arguments);

    float value = 1.0f;
    EXPECT_EQ(dnnl_memory_set_data_handle_v2(mem, &value, -1),
            dnnl_invalid_arguments);
    EXPECT_EQ(dnnl_memory_set_data_handle_v2(mem, &value, 1),
            dnnl_invalid_arguments);

    DNNL_CHECK(dnnl_memory_destroy(mem));
    DNNL_CHECK(dnnl_memory_desc_destroy(md));
    DNNL_CHECK(dnnl_engine_destroy(engine));
}

TEST_P(cpp_api_mem_handle_idx_t, TestOutOfRangeIndex) {
    dnnl_engine_kind_t eng_kind_c = GetParam();
    engine::kind eng_kind = static_cast<engine::kind>(eng_kind_c);
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine is not found.");

    engine eng(eng_kind, 0);

    memory::desc md({4}, memory::data_type::f32, memory::format_tag::x);
    auto mem = test::make_memory(md, eng);

    EXPECT_THROW(mem.get_data_handle(-1), dnnl::error);
    EXPECT_THROW(mem.get_data_handle(1), dnnl::error);

    float value = 1.0f;
    EXPECT_THROW(mem.set_data_handle(&value, -1), dnnl::error);
    EXPECT_THROW(mem.set_data_handle(&value, 1), dnnl::error);
}

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, c_api_mem_handle_idx_t,
        all_engine_kinds, print_to_string_param_name_t());
INSTANTIATE_TEST_SUITE_P(AllEngineKinds, cpp_api_mem_handle_idx_t,
        all_engine_kinds, print_to_string_param_name_t());

/**
 * Test host scalar memory with supported functions.
 */
TEST(cpp_api_host_scalar_mem, TestSupportedFunctions) {
    using namespace dnnl;

    float scalar_value = 42.0f;
    memory scalar_mem(
            memory::desc::host_scalar(memory::data_type::f32), scalar_value);

    EXPECT_EQ(scalar_mem.get_host_scalar_value<float>(), 42.0f);

    float new_value = 84.0f;
    scalar_mem.set_host_scalar_value(new_value);
    EXPECT_EQ(scalar_mem.get_host_scalar_value<float>(), 84.0f);

    dnnl::engine eng = {};
    EXPECT_EQ(scalar_mem.get_engine(), eng);

    void *mapped_value_ptr = scalar_mem.map_data(0);
    ASSERT_NE(mapped_value_ptr, nullptr);
    EXPECT_EQ(*(float *)mapped_value_ptr, 84.0f);

    scalar_mem.unmap_data(mapped_value_ptr, 0);
}

TEST(cpp_api_host_scalar_mem, TestUnsupportedFunctions) {
    using namespace dnnl;

    auto scalar_md = memory::desc::host_scalar(memory::data_type::f32);
    float scalar_value = 42.0f;

    engine::kind eng_kind = engine::kind::cpu;
    SKIP_IF(engine::get_count(eng_kind) == 0, "Engine is not found.");
    engine engine(eng_kind, 0);

    // Ensure that memory creation with host scalar descriptor is not allowed
    EXPECT_THROW(memory(scalar_md, engine), dnnl::error);

    // Ensure that set/get data handle is not allowed with host scalar memory
    memory scalar_mem(scalar_md, scalar_value);
    EXPECT_THROW(scalar_mem.get_data_handle(), dnnl::error);
    EXPECT_THROW(scalar_mem.set_data_handle(&scalar_value), dnnl::error);
}

TEST(cpp_api_host_scalar_mem, TestDataTypeMismatch) {
    using namespace dnnl;

    auto scalar_md = memory::desc::host_scalar(memory::data_type::f16);

    // Attempt to create a memory object with a data type that does not match
    try {
        float scalar_value = 42.0f;
        memory scalar_mem(scalar_md, scalar_value);
    } catch (const dnnl::error &e) {
        EXPECT_EQ(e.status, dnnl_invalid_arguments);
        EXPECT_EQ(std::string(e.message),
                "scalar type size does not match memory descriptor data type "
                "size");
    }

    auto scalar_md_f32 = memory::desc::host_scalar(memory::data_type::f32);
    float scalar_value = 42.0f;
    memory scalar_mem(scalar_md_f32, scalar_value);

    try {
        scalar_mem.set_host_scalar_value(1);
    } catch (const dnnl::error &e) {
        EXPECT_EQ(e.status, dnnl_invalid_arguments);
        EXPECT_EQ(std::string(e.message),
                "scalar type size does not match memory descriptor data type "
                "size");
    }

    try {
        (void)scalar_mem.get_host_scalar_value<int>();
    } catch (const dnnl::error &e) {
        EXPECT_EQ(e.status, dnnl_invalid_arguments);
        EXPECT_EQ(std::string(e.message),
                "scalar type size does not match memory descriptor data type "
                "size");
    }
}

} // namespace dnnl
