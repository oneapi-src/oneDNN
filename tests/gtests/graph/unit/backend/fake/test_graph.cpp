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

#include "backend/fake/fake_backend.hpp"

TEST(Graph, GetFakePartitions) {
    using namespace dnnl::impl::graph;
    using namespace dnnl::graph::tests::unit::utils;

    graph_t agraph;
    op_t wildcard {0, op_kind::Wildcard, std::string("wildcard")};
    op_t end {2, op_kind::End, std::string("end")};
    logical_tensor_t input = logical_tensor_init(0, data_type::f32);
    logical_tensor_t wildcard_out = logical_tensor_init(1, data_type::f32);

    wildcard.add_input(input);
    wildcard.add_output(wildcard_out);
    end.add_input(wildcard_out);
    ASSERT_EQ(agraph.add_op(&wildcard), status::success);
    ASSERT_EQ(agraph.add_op(&end), status::success);
    ASSERT_EQ(agraph.num_ops(), 2U);

    auto &bkd = fake_impl::fake_backend_t::get_singleton();
    bkd.get_partitions(agraph, partition_policy::fusion);
    ASSERT_EQ(agraph.get_num_partitions(), 2U);
    auto partition = agraph.get_partitions()[0].get();
    ASSERT_EQ(partition->get_assigned_backend()->get_name(),
            std::string("fake_backend"));
    ASSERT_TRUE(partition->is_initialized());
}
