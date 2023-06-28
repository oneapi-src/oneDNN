/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include <iostream>
#include <numeric>
#include <vector>
#include "context.hpp"
#include "test_utils.hpp"
#include "util/bf16.hpp"
#include "gtest/gtest.h"
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/fusion_mgr.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/sc_data_format.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/index_flatten.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/fusible/memory_movement.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;

template <class T>
static std::vector<T> ref_transpose(const std::vector<T> &data,
        const sc_dims &input_dims, const std::vector<int> &axis) {
    assert(axis.size() == 2);
    const int num_of_loops = input_dims.size();
    sc_dims output_dims = input_dims;
    std::swap(output_dims[axis[0]], output_dims[axis[1]]);
    std::vector<int> lp_vars(num_of_loops);
    std::vector<T> ret(data.size());

    std::function<void(int)> do_for_loop;
    do_for_loop = [&](int lp_index) {
        for (; lp_vars[lp_index] < output_dims[lp_index]; lp_vars[lp_index]++) {
            if (lp_index == num_of_loops - 1) {
                auto input_lp_vars = lp_vars;
                std::swap(input_lp_vars[axis[0]], input_lp_vars[axis[1]]);
                int out_index = 0, in_index = 0;
                for (auto i = 0; i < num_of_loops; i++) {
                    if (i == num_of_loops - 1) {
                        out_index += lp_vars[i];
                        in_index += input_lp_vars[i];
                    } else {
                        out_index
                                = (out_index + lp_vars[i]) * output_dims[i + 1];
                        in_index = (in_index + input_lp_vars[i])
                                * input_dims[i + 1];
                    }
                }
                ret[out_index] = data[in_index];
            } else {
                do_for_loop(lp_index + 1);
            }
        }
        lp_vars[lp_index] = 0;
    };
    do_for_loop(0);
    return ret;
}

void transpose_test(const sc_dims &input_dims, const std::vector<int> &order,
        const std::vector<int> &axis) {
    REQUIRE_AVX2();
    sc_graph_t graph;
    auto in = graph_tensor::make(
            input_dims, sc_data_format_t(), datatypes::f32);
    auto trans = graph.make("transpose", {in}, {}, {{"order", order}});
    auto out = graph.make_output(trans->get_outputs());

    graph_driver(graph, get_test_ctx());
    ir_module_ptr mod = lower_graph(get_test_ctx(), graph,
            {graph.get_input_ops()[0], graph.get_output_ops()[0]});
    auto fptr = jit_engine_t::make(get_test_ctx())->get_entry_func(mod, true);

    const auto input_size = test_utils::product(input_dims);
    std::vector<float> input_data(input_size);
    test_utils::fill_data(input_data.data(), input_size);
    std::vector<float> output_data(input_size);
    std::vector<generic_val> gargs;
    gargs.emplace_back(input_data.data());
    gargs.emplace_back(output_data.data());
    fptr->call_generic_default(gargs.data());
    std::vector<float> ref_output(input_size);
    ref_output = ref_transpose<float>(input_data, input_dims, axis);
    test_utils::compare_data(output_data, ref_output, 1e-4f, 1e-5f);
}

static void check_format_correctness(const sc_dims &plain_dims,
        const std::vector<int> &order, sc_data_format_t input_format,
        sc_data_format_t ref_output_format) {
    sc_graph_t graph;
    auto in = graph_tensor::make(plain_dims, input_format, datatypes::f32);
    auto trans = graph.make("transpose", {in}, {}, {{"order", order}});
    auto out = trans->get_outputs()[0];

    std::vector<std::vector<format_stride_pair>> in_supported_lts,
            out_supported_lts;
    trans->query_format(
            get_default_context(), in_supported_lts, out_supported_lts);
    auto fs_pair = out_supported_lts[0][0];
    out->details_.set_format_and_stride(fs_pair.first, fs_pair.second);
    ASSERT_TRUE(fs_pair.first == ref_output_format);
    ASSERT_TRUE(in->details_.get_blocking_dims()
            == out->details_.get_blocking_dims());
}

TEST(GCCore_CPU_transpose_test, TestQueryFormat) {
    check_format_correctness({1, 2, 3, 4}, {0, 2, 1, 3},
            sc_data_format_t(sc_data_format_kind_t(0, 3, 1, 2)),
            sc_data_format_t(sc_data_format_kind_t(0, 3, 2, 1)));
    check_format_correctness({1, 12, 128, 64}, {0, 1, 3, 2},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3, 2, 3), {12, 64}),
            sc_data_format_t(
                    sc_data_format_kind_t(0, 1, 3, 2, 3, 2), {12, 64}));
    check_format_correctness({1, 12, 128, 64}, {0, 3, 2, 1},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3, 2, 3), {64, 32}),
            sc_data_format_t(
                    sc_data_format_kind_t(0, 3, 2, 1, 2, 1), {64, 32}));
    check_format_correctness({1, 12, 128, 64}, {0, 1, 3, 2},
            sc_data_format_t(
                    sc_data_format_kind_t(0, 1, 2, 3, 2, 3, 2), {32, 16, 2}),
            sc_data_format_t(
                    sc_data_format_kind_t(0, 1, 3, 2, 3, 2, 3), {32, 16, 2}));
    check_format_correctness({1, 12, 128, 64}, {0, 3, 2, 1},
            sc_data_format_t(
                    sc_data_format_kind_t(0, 1, 2, 3, 2, 3, 2), {32, 16, 2}),
            sc_data_format_t(
                    sc_data_format_kind_t(0, 3, 2, 1, 2, 1, 2), {32, 16, 2}));
    check_format_correctness({1, 384, 16, 64}, {0, 2, 3, 1},
            sc_data_format_t(sc_data_format_kind_t(0, 1, 2, 3)),
            sc_data_format_t(sc_data_format_kind_t(0, 3, 1, 2)));
    check_format_correctness({1, 12, 128, 64}, {3, 2, 0, 1},
            sc_data_format_t(
                    sc_data_format_kind_t(0, 3, 1, 2, 3, 2, 3), {32, 16, 2}),
            sc_data_format_t(
                    sc_data_format_kind_t(2, 0, 3, 1, 0, 1, 0), {32, 16, 2}));
}

TEST(GCCore_CPU_transpose_test, TestSingleTranspose) {
    transpose_test({4, 8, 16, 32}, {0, 1, 3, 2}, {2, 3});
    transpose_test({4, 8, 16, 32}, {0, 2, 1, 3}, {1, 2});
}
