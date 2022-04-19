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

#include <array>
#include <limits>

#include "gtest/gtest.h"

#include "interface/c_types_map.hpp"
#include "interface/logical_tensor.hpp"
#include "interface/op.hpp"
#include "interface/op_schema.hpp"
#include "interface/partition.hpp"
#include "interface/partition_impl.hpp"

#include "backend/dnnl/dnnl_backend.hpp"
#include "backend/dnnl/internal_ops.hpp"

#include "cpp/unit/utils.hpp"

TEST(Op, FusedOp) {
    using namespace dnnl::graph::impl;
    op_t conv {0, op_kind::Convolution, std::string("convolution")};
    op_t relu {1, op_kind::ReLU, std::string("relu")};
    op_t mul {2, op_kind::Multiply, std::string("multiply")};
    ASSERT_FALSE(conv.is_fused());
    ASSERT_FALSE(relu.is_fused());
    ASSERT_FALSE(mul.is_fused());

    op_t conv_post_ops {4, impl::dnnl_impl::op_kind::conv_post_ops_fusion,
            std::string("conv_post_ops"), true};
    conv_post_ops.add_op_ids({0, 1});
    conv_post_ops.add_op_ids(2);
    ASSERT_TRUE(conv_post_ops.is_fused());

    std::vector<size_t> ret = conv_post_ops.get_op_ids();
    ASSERT_EQ(ret.size(), 3);
    ASSERT_EQ(ret[0], 0);
    ASSERT_EQ(ret[1], 1);
    ASSERT_EQ(ret[2], 2);
}
