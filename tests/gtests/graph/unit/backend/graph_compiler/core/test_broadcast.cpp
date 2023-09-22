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

#include <numeric>
#include <vector>

#include "context.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/fusible/broadcast.hpp>
#include <util/math_utils.hpp>

using namespace dnnl::impl::graph::gc;

static void broadcast_ref(const sc_dims &in_plain_dims,
        const sc_dims &out_plain_dims, const std::vector<int> &bc_axis,
        test_buffer<float> &in, std::vector<float> &out) {
    auto extended_in_plain_dims = test_utils::get_extended_plain_dims(
            bc_axis, in_plain_dims, out_plain_dims);
    sc_dims input_strides
            = test_utils::compute_dense_stride(extended_in_plain_dims);
    sc_dims output_strides = test_utils::compute_dense_stride(out_plain_dims);
    const size_t total_size = out.size();
    utils::parallel_for(0, total_size, 1, [&](int64_t i) {
        sc_dims output_idx
                = test_utils::flattened_idx_to_ndims_idx(i, output_strides);
        sc_dims input_idx(output_idx.size());
        for (size_t d = 0; d < input_idx.size(); ++d) {
            input_idx[d] = extended_in_plain_dims[d] == 1 ? 0 : output_idx[d];
        }
        auto prod = math_utils::vector_mul(input_idx, input_strides);
        size_t in_idx = std::accumulate(prod.begin(), prod.end(), 0);
        out[i] = in[in_idx];
    });
}

static void check_broadcast_correctness(const sc_dims &in_plain_dims,
        const sc_dims &out_plain_dims,
        const std::vector<int> &bc_axis = std::vector<int> {}) {
    REQUIRE_AVX2(); // llvm stuck when SSE
    sc_graph_t graph;
    auto input = graph.make_input({std::make_shared<graph_tensor>(
            nullptr, sc_data_format_t(), in_plain_dims, datatypes::f32)});
    auto broadcast = graph.make("broadcast", input->get_outputs(), {},
            {{"output_shape", out_plain_dims}, {"bc_axis", bc_axis}});
    auto output = graph.make_output(broadcast->get_outputs());
    graph_driver(graph, get_test_ctx());
    ir_module_ptr mod = lower_graph(get_test_ctx(), graph, {input, output});
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(mod, true);
    std::vector<generic_val> gargs;
    sc_dim in_size = test_utils::product(in_plain_dims);
    sc_dim out_size = test_utils::product(out_plain_dims);

    test_buffer<float> in = alloc_array<float>(in_size, INIT_RANDOM);
    test_buffer<float> out = alloc_array<float>(out_size, INIT_ZERO);
    std::vector<float> ref_out(out_size, 0.f);
    gargs.emplace_back(in.data());
    gargs.emplace_back(out.data());
    fptr->call_generic_default(gargs.data());
    broadcast_ref(in_plain_dims, out_plain_dims, bc_axis, in, ref_out);
    test_utils::compare_data<float>(out.data(), ref_out.data(), ref_out.size());
}

TEST(GCCore_CPU_broadcast_test, TestCorrectness) {
    check_broadcast_correctness({32}, {32});
    check_broadcast_correctness({34}, {34});
    check_broadcast_correctness({1}, {32});
    check_broadcast_correctness({1}, {2});
    check_broadcast_correctness({2}, {2});
    check_broadcast_correctness({34}, {34});
    check_broadcast_correctness({1}, {34});
    check_broadcast_correctness({1, 2}, {2, 2});
    check_broadcast_correctness({2, 1}, {2, 2});
    check_broadcast_correctness({1, 17}, {17, 17});
    check_broadcast_correctness({17, 1}, {17, 17});
    check_broadcast_correctness({2}, {3, 2});
    check_broadcast_correctness({3}, {3, 2}, {0});
    check_broadcast_correctness({1}, {2, 2, 2, 2});
    check_broadcast_correctness({1, 16, 1, 1}, {4, 16, 8, 4});
    check_broadcast_correctness({4, 1, 1, 4}, {4, 16, 8, 4});
    check_broadcast_correctness({4, 16, 8, 1}, {4, 16, 8, 4});
    check_broadcast_correctness({4, 1, 8, 4}, {4, 16, 8, 4});
    check_broadcast_correctness({4, 1, 1, 1}, {4, 1, 1, 4});
    check_broadcast_correctness({1, 16, 1, 4}, {1, 16, 8, 4});
}
