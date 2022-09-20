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

#include <gtest/gtest.h>

#include "interface/partition.hpp"
#include "interface/tensor.hpp"

#include "backend/fake/fake_partition_impl.hpp"

#include "graph/unit/unit_test_common.hpp"
#include "graph/unit/utils.hpp"

namespace utils = dnnl::graph::tests::unit::utils;
namespace graph = dnnl::impl::graph;

TEST(CompiledPartition, Unsupported) {
    graph::engine_t *eng = get_engine();

    graph::op_t n(graph::op_kind::Wildcard);

    graph::logical_tensor_t lt_in = utils::logical_tensor_init(
            /* tid= */ 1, {1, 1, 3, 3}, graph::data_type::f32);
    graph::logical_tensor_t lt_out = utils::logical_tensor_init(
            /* tid= */ 2, graph::data_type::f32, graph::layout_type::any);

    n.add_input(lt_in);
    n.add_output(lt_out);

    auto pimpl = std::make_shared<graph::fake_impl::fake_partition_impl_t>(
            eng->kind());
    pimpl->init(&n);

    graph::partition_t p;
    p.init(pimpl);

    graph::compiled_partition_t cp(p);

    std::vector<const graph::logical_tensor_t *> lt_ins {&lt_in};
    std::vector<graph::logical_tensor_t *> inferred_output {&lt_out};
    ASSERT_EQ(p.infer_shape(lt_ins, inferred_output),
            graph::status::unimplemented);

    std::vector<const graph::logical_tensor_t *> lt_outs {&lt_out};

    graph::status_t status = p.compile(&cp, lt_ins, lt_outs, eng);
    ASSERT_EQ(status, graph::status::unimplemented);
}
