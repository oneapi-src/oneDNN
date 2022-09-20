/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#include <cstdint>

#include <gtest/gtest.h>

#include "oneapi/dnnl/dnnl_graph.hpp"

TEST(APIGraph, GetPartitions) {
    using namespace dnnl::graph;
    dnnl::engine::kind engine_kind = dnnl::engine::kind::cpu;
    graph g(engine_kind);
    op relu_op(0, op::kind::ReLU, "relu");
    logical_tensor lt1 {0, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    logical_tensor lt2 {1, logical_tensor::data_type::f32,
            logical_tensor::layout_type::strided};
    relu_op.add_input(lt1);
    relu_op.add_output(lt2);
    g.add_op(relu_op);
    g.finalize();
    // with default policy: partition::policy::fusion
    auto partitions = g.get_partitions();

    EXPECT_EQ(partitions.size(), 1U);
}

TEST(APIGraph, AddOp) {
    using namespace dnnl::graph;
    dnnl::engine::kind engine_kind = dnnl::engine::kind::cpu;
    graph g(engine_kind);

    op conv(0, op::kind::Convolution, "conv");

    const bool allow_exception = false;
    ASSERT_NE(g.add_op(conv, allow_exception), status::success);
    EXPECT_THROW(g.add_op(conv), dnnl::error);
}
