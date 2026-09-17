/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gtest/gtest.h"

#include "backend/dnnl/fusion_info.hpp"

#include "interface/c_types_map.hpp"

#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;

using namespace dnnl::graph::tests::unit::utils;

TEST(test_fusion_info, GetMutableZeroPoints) {
    auto zp_op
            = std::make_shared<graph::op_t>(graph::op_kind::_add_zps, "zps_op");

    graph::dnnl_impl::fusion_info_t info;
    ASSERT_NO_THROW(info.set_zero_points(zp_op, false, 0));
    ASSERT_EQ(info.get_mutable_zero_points(false, 0), zp_op.get());
}

TEST(test_fusion_info, GetPerGroupMaskWithDegenerateDimension) {
    graph::op_t deq_0(0, graph::op_kind::DynamicDequantize, "deq_0");
    deq_0.set_attr<std::string>(graph::op_attr::qtype, "per_group");
    deq_0.add_input(
            logical_tensor_init(0, {1, 16, 256, 1024}, graph::data_type::s8));
    deq_0.add_input(
            logical_tensor_init(1, {1, 16, 1, 1024}, graph::data_type::f32));

    EXPECT_EQ(graph::dnnl_impl::get_quant_mask(&deq_0), 15);

    graph::op_t deq_1(1, graph::op_kind::DynamicDequantize, "deq_1");
    deq_1.set_attr<std::string>(graph::op_attr::qtype, "per_group");
    deq_1.add_input(
            logical_tensor_init(2, {2, 16, 256, 1024}, graph::data_type::s8));
    deq_1.add_input(
            logical_tensor_init(3, {1, 16, 1, 1024}, graph::data_type::f32));

    EXPECT_EQ(graph::dnnl_impl::get_quant_mask(&deq_1), 14);
}
