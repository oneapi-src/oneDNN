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

#include <iostream>
#include "context.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/codegen/codegen_c.hpp>
#include <compiler/ir/graph/driver.hpp>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/lowering.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/ir/graph/transform/transform.hpp>
#include <compiler/jit/jit.hpp>
#include <ops/fusible/shape_of_tensor.hpp>
#include <runtime/dynamic_dispatch/dynamic_tensor.hpp>
#include <runtime/dynamic_dispatch/ops/config.hpp>
#include <runtime/runtime.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;

auto get_shape_of_tensor_graph
        = [](int shape_idx, padding_shape_etype_t shape_type,
                  bool is_batch
                  = false) -> std::pair<sc_graph_t, std::vector<sc_op_ptr>> {
    sc_graph_t graph;
    const sc_dims input_dims = {-2, -3, 1024, -4};
    auto input = graph.make_input({graph_tensor::make(input_dims)});
    auto shape_of_tsr = graph.make("shape_of_tensor", input->get_outputs(), {},
            {{"shape_idx", shape_idx},
                    {attr_keys::padding_shape_type,
                            static_cast<int>(shape_type)},
                    {attr_keys::shape_of_tensor_is_batch, is_batch}});
    auto output = graph.make_output(shape_of_tsr->get_outputs());
    return std::make_pair(
            std::move(graph), std::vector<sc_op_ptr> {input, output});
};

TEST(GCCore_CPU_shape_of_tensor_test, TestShapeOfTensorWithoutPadding) {
    REQUIRE_AVX2();
    runtime::dynamic_tensor_t in, out;
    sc_dims in_shapes = {16, 32, 1024, 4096}, out_shapes = {0};
    in.dims_ = in_shapes.data();
    out.data_ = out_shapes.data();
    in.ndims_ = 4;
    out.ndims_ = 1;
    for (int i = 0; i < 4; i++) {
        sc_graph_t graph;
        std::vector<sc_op_ptr> args;
        std::tie(graph, args) = get_shape_of_tensor_graph(
                i, padding_shape_etype_t::without_padding);
        auto ctx = get_test_ctx();
        graph_driver(graph, ctx);
        ir_module_ptr mod = lower_graph(ctx, graph, args);
        auto fptr = jit_engine_t::make(ctx)->get_entry_func(mod, true);
        std::vector<generic_val> gargs;
        gargs.emplace_back(&in);
        gargs.emplace_back(&out);
        fptr->call_generic_default(gargs.data());
        EXPECT_EQ(out_shapes[0], in_shapes[i]);
    }
}

TEST(GCCore_CPU_shape_of_tensor_test, TestShapeOfTensorMatmulPadding) {
    REQUIRE_AVX2();
    runtime::dynamic_tensor_t in, out;
    sc_dims in_shapes = {1, 33, 1024, 97}, out_shapes = {0};
    in.dims_ = in_shapes.data();
    out.data_ = out_shapes.data();
    in.ndims_ = 4;
    out.ndims_ = 1;
    for (int i = 0; i < 4; i++) {
        bool is_batch = i == 0;
        sc_graph_t graph;
        std::vector<sc_op_ptr> args;
        std::tie(graph, args) = get_shape_of_tensor_graph(
                i, padding_shape_etype_t::matmul_padding, is_batch);
        auto ctx = get_test_ctx();
        graph_driver(graph, ctx);
        ir_module_ptr mod = lower_graph(ctx, graph, args);
        auto fptr = jit_engine_t::make(ctx)->get_entry_func(mod, true);
        std::vector<generic_val> gargs;
        gargs.emplace_back(&in);
        gargs.emplace_back(&out);
        fptr->call_generic_default(gargs.data());
        int block = get_matmul_dyn_cfg_single(in_shapes[i], is_batch);
        EXPECT_EQ(out_shapes[0],
                static_cast<int>(utils::divide_and_ceil(in_shapes[i], block))
                        * block);
    }
}
