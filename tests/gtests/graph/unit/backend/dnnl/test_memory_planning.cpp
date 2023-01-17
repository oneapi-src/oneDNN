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
#include <memory>

#include "interface/c_types_map.hpp"

#include "backend/dnnl/passes/memory_planning.hpp"

#include "gtest/gtest.h"

#include "graph/unit/backend/dnnl/dnnl_test_common.hpp"
#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;
namespace dnnl_impl = graph::dnnl_impl;

TEST(MemoryPlanning, GetMemoryInfo) {
    dnnl_impl::memory_planner_t mp;
    graph::op_t op {0, graph::op_kind::Abs, "abs"};
    auto lt = utils::logical_tensor_init(
            1, {1, 2}, graph::data_type::f32, graph::layout_type::strided);
    graph::value_t val {op, 0, lt};
    ASSERT_NO_THROW(mp.get_memory_info(&val));
}
