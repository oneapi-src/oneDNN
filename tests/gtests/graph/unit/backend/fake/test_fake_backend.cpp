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

#include "gtest/gtest.h"

#include "interface/tensor.hpp"

#include "backend/fake/fake_backend.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(FakeBackend, GetMemSize) {
    graph::logical_tensor_t lt = utils::logical_tensor_init(
            /* tid= */ 1, {1, 1, 3, 3}, graph::data_type::f32);
    auto &fake_backend = graph::fake_impl::fake_backend_t::get_singleton();

    ASSERT_EQ(fake_backend.get_mem_size(lt), static_cast<size_t>(-1));
}
