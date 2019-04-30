/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "mkldnn_test_common.hpp"
#include "gtest/gtest.h"

#include "mkldnn.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <vector>

namespace mkldnn {

class memory_map_test_c : public ::testing::TestWithParam<mkldnn_engine_kind_t>
{
protected:
    virtual void SetUp() {
        auto engine_kind = GetParam();
        if (mkldnn_engine_get_count(engine_kind) == 0)
            return;

        MKLDNN_CHECK(mkldnn_engine_create(&engine, engine_kind, 0));
        MKLDNN_CHECK(mkldnn_stream_create(
                &stream, engine, mkldnn_stream_default_flags));
    }

    virtual void TearDown() {
        if (engine) {
            MKLDNN_CHECK(mkldnn_engine_destroy(engine));
        }
        if (stream) {
            MKLDNN_CHECK(mkldnn_stream_destroy(stream));
        }
    }

    mkldnn_engine_t engine = nullptr;
    mkldnn_stream_t stream = nullptr;
};

class memory_map_test_cpp
    : public ::testing::TestWithParam<mkldnn_engine_kind_t>
{
};

TEST_P(memory_map_test_c, MapNullMemory) {
    SKIP_IF(!engine, "Engine kind is not supported.");

    int ndims = 4;
    mkldnn_dims_t dims = { 2, 3, 4, 5 };
    mkldnn_memory_desc_t mem_d;
    mkldnn_memory_t mem;

    MKLDNN_CHECK(mkldnn_memory_desc_init_by_tag(
            &mem_d, ndims, dims, mkldnn_f32, mkldnn_nchw));
    MKLDNN_CHECK(mkldnn_memory_create(&mem, &mem_d, engine, nullptr));

    void *mapped_ptr;
    MKLDNN_CHECK(mkldnn_memory_map_data(mem, &mapped_ptr));
    ASSERT_EQ(mapped_ptr, nullptr);

    MKLDNN_CHECK(mkldnn_memory_unmap_data(mem, mapped_ptr));
    MKLDNN_CHECK(mkldnn_memory_destroy(mem));
}

TEST_P(memory_map_test_c, Map) {
    SKIP_IF(!engine, "Engine kind is not supported.");

    const int ndims = 1;
    const mkldnn_dim_t N = 15;
    const mkldnn_dims_t dims = { N };

    mkldnn_memory_desc_t mem_d;
    MKLDNN_CHECK(mkldnn_memory_desc_init_by_tag(
            &mem_d, ndims, dims, mkldnn_f32, mkldnn_x));

    // Create and fill mem_ref to use as a reference
    mkldnn_memory_t mem_ref;
    MKLDNN_CHECK(mkldnn_memory_create(
            &mem_ref, &mem_d, engine, MKLDNN_MEMORY_ALLOCATE));

    float buffer_ref[N];
    std::iota(buffer_ref, buffer_ref + N, 1);

    void *mapped_ptr_ref;
    MKLDNN_CHECK(mkldnn_memory_map_data(mem_ref, &mapped_ptr_ref));
    float *mapped_ptr_ref_f32 = static_cast<float *>(mapped_ptr_ref);
    std::copy(buffer_ref, buffer_ref + N, mapped_ptr_ref_f32);
    MKLDNN_CHECK(mkldnn_memory_unmap_data(mem_ref, mapped_ptr_ref));

    // Create memory for the tested engine
    mkldnn_memory_t mem;
    MKLDNN_CHECK(mkldnn_memory_create(
            &mem, &mem_d, engine, MKLDNN_MEMORY_ALLOCATE));

    // Reorder mem_ref to memory
    mkldnn_primitive_desc_t reorder_pd;
    MKLDNN_CHECK(mkldnn_reorder_primitive_desc_create(
            &reorder_pd, &mem_d, engine, &mem_d, engine, nullptr));

    mkldnn_primitive_t reorder;
    MKLDNN_CHECK(mkldnn_primitive_create(&reorder, reorder_pd));

    mkldnn_exec_arg_t reorder_args[2]
            = { { MKLDNN_ARG_SRC, mem_ref }, { MKLDNN_ARG_DST, mem } };
    MKLDNN_CHECK(mkldnn_primitive_execute(reorder, stream, 2, reorder_args));
    MKLDNN_CHECK(mkldnn_stream_wait(stream));

    // Validate the results
    void *mapped_ptr;
    MKLDNN_CHECK(mkldnn_memory_map_data(mem, &mapped_ptr));
    float *mapped_ptr_f32 = static_cast<float *>(mapped_ptr);
    for (size_t i = 0; i < N; i++) {
        ASSERT_EQ(mapped_ptr_f32[i], buffer_ref[i]);
    }
    MKLDNN_CHECK(mkldnn_memory_unmap_data(mem, mapped_ptr));

    // Clean up
    MKLDNN_CHECK(mkldnn_primitive_destroy(reorder));
    MKLDNN_CHECK(mkldnn_primitive_desc_destroy(reorder_pd));

    MKLDNN_CHECK(mkldnn_memory_destroy(mem));
    MKLDNN_CHECK(mkldnn_memory_destroy(mem_ref));
}

TEST_P(memory_map_test_cpp, Map) {
    auto engine_kind = static_cast<engine::kind>(GetParam());

    SKIP_IF(engine::get_count(engine_kind) == 0,
            "Engine kind is not supported");

    engine eng(engine_kind, 0);

    const mkldnn::memory::dim N = 7;
    memory::desc mem_d({ N }, memory::data_type::f32, memory::format_tag::x);

    memory mem_ref(mem_d, eng);

    float buffer_ref[N];
    std::iota(buffer_ref, buffer_ref + N, 1);

    float *mapped_ptr_ref = mem_ref.map_data<float>();
    std::copy(buffer_ref, buffer_ref + N, mapped_ptr_ref);
    mem_ref.unmap_data(mapped_ptr_ref);

    memory mem(mem_d, eng);

    reorder::primitive_desc reorder_pd(
            eng, mem_d, eng, mem_d, primitive_attr());
    reorder reorder_prim(reorder_pd);

    stream strm(eng);
    reorder_prim.execute(strm, mem_ref, mem);
    strm.wait();

    float *mapped_ptr = mem.map_data<float>();
    for (size_t i = 0; i < N; i++) {
        ASSERT_EQ(mapped_ptr[i], buffer_ref[i]);
    }
    mem.unmap_data(mapped_ptr);
}

namespace {
struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
            const ::testing::TestParamInfo<ParamType> &info) const {
        return to_string(info.param);
    }
};

auto all_engine_kinds = ::testing::Values(mkldnn_cpu, mkldnn_gpu);

} // namespace

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, memory_map_test_c, all_engine_kinds,
        PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(AllEngineKinds, memory_map_test_cpp, all_engine_kinds,
        PrintToStringParamName());

} // namespace mkldnn
