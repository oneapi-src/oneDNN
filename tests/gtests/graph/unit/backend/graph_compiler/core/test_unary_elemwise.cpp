/*******************************************************************************
 * Copyright 2023 Intel Corporation
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
#include "act_ref.hpp"
#include "compiler/ir/graph/driver.hpp"
#include "compiler/ir/graph/graph.hpp"
#include "compiler/ir/graph/lowering.hpp"
#include "compiler/ir/sc_data_format.hpp"
#include "context.hpp"
#include "gtest/gtest.h"
#include <compiler/jit/jit.hpp>

using namespace dnnl::impl::graph::gc;
TEST(GCCore_unary_elemwise, TestStridedRelu) {
    BUILTIN_REQUIRE_AVX512();
    sc_dims input_dims = {128, 64, 56, 56};
    sc_graph_t g;
    auto ins = g.make_input({graph_tensor::make(input_dims)});
    auto reorderop = g.make("reorder", ins->get_outputs(),
            {graph_tensor::make(input_dims, sc_data_format_t::NCHWc(16))}, {});
    auto out = g.make_output(g.make("relu", reorderop->get_outputs(),
                                      {graph_tensor::make(input_dims)}, {})
                                     ->get_outputs());

    const auto input_size = test_utils::product(input_dims);
    std::vector<float> input_data(input_size), sc_output(input_size),
            ref_output(input_size);
    test_utils::fill_data(&input_data[0], input_size);

    graph_driver(g);
    auto f = lower_graph(get_test_ctx(), g, {ins, out});
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(f);
    fptr->call_default(&input_data[0], &sc_output[0]);

    ref_relu(ref_output.data(), input_data.data(), input_size);
    test_utils::compare_data(sc_output, ref_output, 1e-4, 1e-5);
}
