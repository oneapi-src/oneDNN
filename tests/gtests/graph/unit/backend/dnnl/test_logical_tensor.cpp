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

#include <vector>
#include <gtest/gtest.h>

#include "graph/unit/utils.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "interface/backend.hpp"
#include "interface/logical_tensor.hpp"

namespace graph = dnnl::impl::graph;
namespace dnnl_impl = dnnl::impl::graph::dnnl_impl;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(LogicalTensor, ImplicitEqualLayout) {
    using ltw = graph::logical_tensor_wrapper_t;

    dnnl::memory::desc md({1, 2, 3, 4}, dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::nchw);
    auto layout_idx = dnnl_impl::dnnl_backend::get_singleton().set_mem_desc(md);
    ASSERT_TRUE(layout_idx.has_value());
    auto backend_idx = dnnl_impl::dnnl_backend::get_singleton().get_id();
    auto id = graph::backend_registry_t::encode_layout_id(
            layout_idx.value(), backend_idx);

    graph::logical_tensor_t lt1 = utils::logical_tensor_init(
            0, {1, 2, 3, 4}, graph::data_type::f32, graph::layout_type::any);
    // set opaque layout id
    lt1.layout_type = graph::layout_type::opaque;
    lt1.layout.layout_id = id;

    // public layout
    graph::logical_tensor_t lt2 = utils::logical_tensor_init(0, {1, 2, 3, 4},
            graph::data_type::f32, graph::layout_type::strided);

    ASSERT_TRUE(ltw(lt1).has_same_layout_as(ltw(lt2)));
    ASSERT_TRUE(ltw(lt2).has_same_layout_as(ltw(lt1)));
}
