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

#include <cstddef>
#include <cstdint>

#include "gtest/gtest.h"

#include "backend/dnnl/utils.hpp"

#include "cpp/unit/unit_test_common.hpp"
#include "cpp/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(DnnlUtils, TryReverseAxis) {
    auto par1 = std::make_pair<bool, int64_t>(true, 0);
    ASSERT_EQ(graph::dnnl_impl::utils::try_reverse_axis(0, 3), par1);

    auto par2 = std::make_pair<bool, int64_t>(true, 2);
    ASSERT_EQ(graph::dnnl_impl::utils::try_reverse_axis(-1, 3), par2);

    auto par3 = std::make_pair<bool, int64_t>(false, -4);
    ASSERT_EQ(graph::dnnl_impl::utils::try_reverse_axis(-4, 3), par3);

    auto par4 = std::make_pair<bool, int64_t>(false, 4);
    ASSERT_EQ(graph::dnnl_impl::utils::try_reverse_axis(4, 3), par4);
}
