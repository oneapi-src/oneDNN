/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include <string>

#include "gtest/gtest.h"

#include "graph/unit/utils.hpp"
#include "interface/c_types_map.hpp"
#include "interface/graph.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/value.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/fake/fake_backend.hpp"

TEST(Graph, GetDnnlPartitions) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t conv {0, op_kind::Convolution, std::string("conv2d")};
    op_t relu {1, op_kind::ReLU, std::string("relu")};
    op_t end {2, op_kind::End, std::string("end")};
    logical_tensor_t conv_src = logical_tensor_init(0, data_type::f32);
    logical_tensor_t conv_wei = logical_tensor_init(1, data_type::f32);
    logical_tensor_t conv_dst = logical_tensor_init(2, data_type::f32);
    logical_tensor_t relu_dst = logical_tensor_init(3, data_type::f32);
    conv.add_input(conv_src);
    conv.add_input(conv_wei);
    conv.add_output(conv_dst);
    conv.set_attr<std::vector<int64_t>>(op_attr::strides, {1, 1});
    conv.set_attr<std::vector<int64_t>>(op_attr::dilations, {1, 1});
    conv.set_attr<std::vector<int64_t>>(op_attr::pads_begin, {0, 0});
    conv.set_attr<std::vector<int64_t>>(op_attr::pads_end, {0, 0});
    conv.set_attr<int64_t>(op_attr::groups, 1);
    conv.set_attr<std::string>(op_attr::data_format, "NCX");
    conv.set_attr<std::string>(op_attr::weights_format, "OIX");
    relu.add_input(conv_dst);
    relu.add_output(relu_dst);
    end.add_input(relu_dst);
    ASSERT_EQ(agraph.add_op(&conv), status::success);
    ASSERT_EQ(agraph.add_op(&relu), status::success);
    ASSERT_EQ(agraph.add_op(&end), status::success);
    ASSERT_EQ(agraph.num_ops(), 3U);

    ASSERT_EQ(agraph.finalize(), status::success);
    auto &dnnl_bkd = dnnl_impl::dnnl_backend::get_singleton();
    dnnl_bkd.get_partitions(agraph, partition_policy::fusion);
    auto &fake_bkd = fake_impl::fake_backend_t::get_singleton();
    fake_bkd.get_partitions(agraph, partition_policy::fusion);
    ASSERT_EQ(agraph.get_num_partitions(), 2U);
    auto p1 = agraph.get_partitions()[0].get();
    ASSERT_NE(p1->get_assigned_backend()->get_name(),
            std::string("fake_backend"));
    ASSERT_TRUE(p1->is_initialized());
    auto p2 = agraph.get_partitions()[1].get();
    ASSERT_EQ(p2->get_assigned_backend()->get_name(),
            std::string("fake_backend"));
    ASSERT_TRUE(p2->is_initialized());
}
