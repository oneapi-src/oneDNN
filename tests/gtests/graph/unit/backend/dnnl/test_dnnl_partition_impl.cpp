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

#include "backend/dnnl/dnnl_partition_impl.hpp"
#include "backend/dnnl/kernels/pool.hpp"

#include "interface/partition.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace graph = dnnl::impl::graph;
namespace utils = dnnl::graph::tests::unit::utils;

TEST(DnnlPartitionImpl, InferShape) {
    graph::engine_t &engine = *get_engine();
    size_t id = 0;

    graph::logical_tensor_t lt1
            = utils::logical_tensor_init(id++, graph::data_type::f32);
    graph::logical_tensor_t lt2
            = utils::logical_tensor_init(id++, graph::data_type::f32);
    graph::logical_tensor_t lt3
            = utils::logical_tensor_init(id++, graph::data_type::f32);

    std::vector<const graph::logical_tensor_t *> inputs {&lt1, &lt2};
    std::vector<graph::logical_tensor_t *> outputs {&lt3};

    auto par = std::make_shared<graph::dnnl_impl::dnnl_partition_impl_t>(
            engine.kind(), graph::fpmath_mode::strict,
            graph::partition_kind_t::undef);
    ASSERT_EQ(par->infer_shape(inputs, outputs), graph::status::success);
}
