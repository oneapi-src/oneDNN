/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#include <iostream>
#include <memory>

#include "gtest/gtest.h"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

#include "interface/backend.hpp"

#include "backend/dnnl/dnnl_backend.hpp"

namespace graph = dnnl::impl::graph;
namespace dnnl_impl = dnnl::impl::graph::dnnl_impl;

TEST(LayoutId, OpaqueMdLayoutIdMapping) {
    using memory = dnnl_impl::memory;
    using data_type = dnnl_impl::data_type;
    using format_tag = dnnl_impl::format_tag;

    dnnl_impl::dnnl_layout_id_manager_t &mgr
            = graph::dnnl_impl::dnnl_backend::get_singleton()
                      .get_layout_id_manager();

    // opaque md should be cached and generate a layout id, and the later
    // layout id should be greater than the former one
    memory::desc md1({8, 3, 224, 224}, data_type::f32, format_tag::nChw16c);
    auto id1 = mgr.set_mem_desc(md1);
    ASSERT_TRUE(id1.has_value());

    memory::desc md2({8, 16, 96, 96}, data_type::f32, format_tag::nChw8c);
    auto id2 = mgr.set_mem_desc(md2);
    ASSERT_TRUE(id2.has_value());

#ifdef DNNL_GRAPH_LAYOUT_DEBUG
    ASSERT_EQ(id1.value(), static_cast<size_t>(format_tag::nChw16c));
    ASSERT_EQ(id2.value(), static_cast<size_t>(format_tag::nChw8c));

    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::memory::desc conv_src(
            {1, 64, 224, 224}, data_type::u8, format_tag::any);
    dnnl::memory::desc conv_wei({64, 64, 1, 1}, data_type::s8, format_tag::any);
    dnnl::memory::desc conv_dst(
            {1, 64, 224, 224}, data_type::u8, format_tag::any);
    dnnl::primitive_attr conv_attr;
    conv_attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
    conv_attr.set_zero_points_mask(DNNL_ARG_DST, 0);
    conv_attr.set_scales_mask(DNNL_ARG_DST, 0);
    conv_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    dnnl::convolution_forward::primitive_desc conv_pd(eng,
            dnnl::prop_kind::forward, dnnl::algorithm::convolution_direct,
            conv_src, conv_wei, conv_dst, dnnl::memory::dims {1, 1},
            dnnl::memory::dims {0, 0}, dnnl::memory::dims {0, 0}, conv_attr);
    // the weight desc for asymc conv will have extra flags
    memory::desc md3 = conv_pd.weights_desc();

    auto id3_asym = mgr.set_mem_desc(md3);
    auto recovered_md3_asym = mgr.get_mem_desc(id3_asym.value());
    ASSERT_TRUE(recovered_md3_asym.has_value());
    ASSERT_EQ(graph::utils::any_cast<memory::desc>(recovered_md3_asym.value()),
            md3);
#else
    ASSERT_GT(id2.value(), id1.value());

    // we should be able to get cached opaque md out according to the
    // layout id
    auto recovered_md1 = mgr.get_mem_desc(id1.value());
    ASSERT_TRUE(recovered_md1.has_value());
    ASSERT_EQ(graph::utils::any_cast<memory::desc>(recovered_md1.value()), md1);

    auto recovered_md2 = mgr.get_mem_desc(id2.value());
    ASSERT_TRUE(recovered_md2.has_value());
    ASSERT_EQ(graph::utils::any_cast<memory::desc>(recovered_md2.value()), md2);
#endif // DNNL_GRAPH_LAYOUT_DEBUG
}
