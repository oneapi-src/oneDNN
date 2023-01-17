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

#include "gtest/gtest.h"

#include "backend/dnnl/fusion_info.hpp"

#include "interface/c_types_map.hpp"

namespace graph = dnnl::impl::graph;

TEST(FusionInfo, GetMutableZeroPoints) {
    auto zp_op = std::make_shared<graph::op_t>(
            graph::dnnl_impl::op_kind::dnnl_add_zps, "zps_op");

    graph::dnnl_impl::fusion_info_t info;
    ASSERT_NO_THROW(info.set_zero_points(zp_op, false, 0));
    ASSERT_EQ(info.get_mutable_zero_points(false, 0), zp_op.get());
}
