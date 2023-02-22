/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#include "gtest/gtest.h"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

#include "interface/c_types_map.hpp"
#include "interface/tensor.hpp"

#include "backend/dnnl/common.hpp"
#include "backend/dnnl/dnnl_backend.hpp"

#include "dnnl.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
namespace dnnl_impl = graph::dnnl_impl;

TEST(Common, GetNxcStride) {
    graph::dims shape {1, 2, 3, 4, 5, 6};
    graph::dims shape_def {720, 1, 240, 60, 12, 2};
    const auto &result = dnnl_impl::get_nxc_strides(shape);
    ASSERT_TRUE(result.size() == shape_def.size()
            && std::equal(result.begin(), result.end(), shape_def.begin()));
}

TEST(Common, GetNcxFormat) {
    ASSERT_EQ(dnnl_impl::get_ncx_format(1), dnnl_impl::format_tag::a);
    ASSERT_EQ(dnnl_impl::get_ncx_format(2), dnnl_impl::format_tag::ab);
    ASSERT_EQ(dnnl_impl::get_ncx_format(3), dnnl_impl::format_tag::abc);
    ASSERT_EQ(dnnl_impl::get_ncx_format(4), dnnl_impl::format_tag::abcd);
    ASSERT_EQ(dnnl_impl::get_ncx_format(5), dnnl_impl::format_tag::abcde);
    ASSERT_EQ(dnnl_impl::get_ncx_format(6), dnnl_impl::format_tag::abcdef);
    ASSERT_EQ(dnnl_impl::get_ncx_format(7), dnnl_impl::format_tag::undef);
}

TEST(Common, MakeDnnlMemory) {
    graph::engine_t &eng = *get_engine();

    graph::logical_tensor_t lt
            = utils::logical_tensor_init(0, {1, 2}, graph::data_type::f32);
    graph::tensor_t t1 {lt, &eng, nullptr};
    if (eng.kind() == graph::engine_kind::cpu) {
        ASSERT_NO_THROW(graph::dnnl_impl::make_dnnl_memory(
                t1, dnnl::engine(dnnl::engine::kind::cpu, 0)));
    } else if (eng.kind() == graph::engine_kind::gpu) {
        ASSERT_NO_THROW(graph::dnnl_impl::make_dnnl_memory(
                t1, dnnl::engine(dnnl::engine::kind::gpu, 0)));
    }
}

TEST(Common, Is4cBlocked) {
    {
        graph::logical_tensor_t lt = utils::logical_tensor_init(
                0, {1, 2}, graph::data_type::f32, graph::layout_type::any);
        auto desc = dnnl_impl::make_dnnl_memory_desc(lt);
        ASSERT_FALSE(dnnl_impl::is_4c_blocked(desc));
    }
    {
        dnnl::memory::desc md3 {{1, 4, 3, 4}, dnnl::memory::data_type::f32,
                dnnl::memory::format_tag::nChw4c, true};
        ASSERT_TRUE(dnnl_impl::is_4c_blocked(md3));
    }
}

TEST(CommonDeathTest, FillLayoutInfo) {
    {
        graph::logical_tensor_t lt = utils::logical_tensor_init(
                0, {1, 2}, graph::data_type::f32, graph::layout_type::any);
        lt.ndims = -1;
        dnnl::memory::desc md;
        ASSERT_EQ(dnnl_impl::fill_layout_info(&lt, md), graph::status::success);
        ASSERT_EQ(lt.layout_type, graph::layout_type::undef);
    }
    {
        graph::logical_tensor_t lt = utils::logical_tensor_init(
                0, {1, 2}, graph::data_type::f32, graph::layout_type::any);
        lt.ndims = -1;
        dnnl::memory::desc md {{1, 2}, dnnl::memory::data_type::f32, {2, 1}};
        ASSERT_EQ(dnnl_impl::fill_layout_info(&lt, md), graph::status::success);
        ASSERT_EQ(lt.data_type, graph::data_type::f32);
        ASSERT_EQ(lt.ndims, static_cast<int32_t>(2));
    }
    {
        graph::logical_tensor_t lt = utils::logical_tensor_init(
                0, {1, 2}, graph::data_type::f32, graph::layout_type::any);
        dnnl::memory::desc md;
#ifndef NDEBUG
        EXPECT_DEATH(dnnl_impl::fill_layout_info(&lt, md), "");
#else
        ASSERT_EQ(dnnl_impl::fill_layout_info(&lt, md),
                graph::status::invalid_arguments);
#endif
    }
    {
        using ltw = graph::logical_tensor_wrapper_t;
        graph::logical_tensor_t lt = utils::logical_tensor_init(
                0, {1, 2}, graph::data_type::f32, graph::layout_type::any);
        dnnl::memory::desc md {{1, 2}, dnnl::memory::data_type::f32, {2, 1}};
        ASSERT_EQ(dnnl_impl::fill_layout_info(&lt, md), graph::status::success);
        ASSERT_TRUE(lt.data_type == graph::data_type::f32 && lt.ndims == 2);
        std::vector<graph::dim_t> dims {1, 2};
        ASSERT_EQ(ltw(lt).vdims(), dims);
        std::vector<graph::dim_t> strides {2, 1};
        ASSERT_EQ(ltw(lt).vstrides(), strides);
    }
}
