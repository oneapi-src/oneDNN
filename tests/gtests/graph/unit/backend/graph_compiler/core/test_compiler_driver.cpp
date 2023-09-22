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
#include <utility>
#include "context.hpp"
#include "test_graph.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/graph/pass/graph_code_cache.hpp>
#include <compiler/ir/graph/pass/pass.hpp>
#include <compiler/jit/compiler_driver.hpp>
#include <unordered_set>

using namespace dnnl::impl::graph::gc;

static sc_graph_t make_graph(size_t id1, size_t id2, float data) {
    sc_graph_t g;
    auto in = g.make_input({make_tensor({128, 128})})->get_outputs()[0];
    auto weight = g.make_input({make_tensor({128, 128})},
                           {{"constant", const_kind::local_const},
                                   {"temp.tensor_id", id1}})
                          ->get_outputs()[0];
    auto weight_bias = g.make_input({make_tensor({1, 128})},
                                {{"constant", const_kind::local_const},
                                        {"temp.tensor_id", id2}})
                               ->get_outputs()[0];
    auto c_1 = g.make("constant", {}, {graph_tensor::make({1})},
                        {{"values",
                                 std::make_shared<static_data_t>(
                                         std::vector<float> {data})},
                                {"dtype", datatypes::f32},
                                {"plain_dims", sc_dims {1}}})
                       ->get_outputs()[0];

    auto weightx2 = g.make("add", {weight, weight}, {}, {})->get_outputs()[0];
    auto addout
            = g.make("add", {weight, weight_bias}, {}, {})->get_outputs()[0];
    auto mm = g.make("matmul_core", {in, addout}, {}, {})->get_outputs()[0];
    auto add2 = g.make("add", {mm, weightx2}, {}, {})->get_outputs()[0];
    auto add3 = g.make("mul", {add2, c_1}, {}, {});
    g.make_output(add3->get_outputs());
    return g;
}

static std::unordered_set<std::shared_ptr<cached_const_graph_tensor>> to_set(
        const std::vector<std::shared_ptr<cached_const_graph_tensor>> &v) {
    return std::unordered_set<std::shared_ptr<cached_const_graph_tensor>> {
            v.begin(), v.end()};
}

TEST(GCCore_CPU_compiler_driver, TestCodeReuse) {
    REQUIRE_AVX2();
    auto ctx = std::make_shared<context_t>(*get_test_ctx());
    ctx->flags_.const_share_ = true;
    ctx->flags_.mixed_fusion_ = true;

    auto g1 = make_graph(0, 1, 1.23f);
    auto mod1
            = compiler_driver(ctx, g1, std::vector<sc_op_ptr> {})->get_module();

    {
        // g1 is exactly the same as g2
        auto g2 = make_graph(0, 1, 1.23f);
        auto mod2 = compiler_driver(ctx, g2, std::vector<sc_op_ptr> {})
                            ->get_module();
        EXPECT_EQ(mod1->code_, mod2->code_);
        ASSERT_EQ(mod1->globals_.shared_tensors_.size(), 3UL);
        ASSERT_EQ(mod2->globals_.shared_tensors_.size(), 3UL);
        EXPECT_EQ(mod1->globals_.shared_tensors_[1],
                mod2->globals_.shared_tensors_[1]);
    }

    {
        // cannot share code with g1. the lazy init tensors are split into two
        // pieces
        // hash=4342983318926907364, base tsr=[0, 1, 2], offsets=[0, 0, 65536]
        auto g2 = make_graph(0, 2, 1.23f);
        auto mod2 = compiler_driver(ctx, g2, std::vector<sc_op_ptr> {})
                            ->get_module();
        EXPECT_NE(mod1->code_, mod2->code_);
        EXPECT_NE(to_set(mod1->globals_.shared_tensors_),
                to_set(mod2->globals_.shared_tensors_));
        EXPECT_TRUE(mod2->code_->graph_cache_handle_);

        // g3 can share code with g2: the lazy init tensors are split into two
        // pieces
        // hash=4342983318926907364, base tsr=[0, 1, 2], offsets=[0, 0, 65536]
        auto g3 = make_graph(0, 4, 1.23f);
        auto mod3 = compiler_driver(ctx, g3, std::vector<sc_op_ptr> {})
                            ->get_module();
        EXPECT_EQ(mod3->code_, mod2->code_);
        EXPECT_NE(to_set(mod3->globals_.shared_tensors_),
                to_set(mod2->globals_.shared_tensors_));

        // hash=4342983318926853943, base tsr=[0, 1, 1], offsets=[0, 0, 65536]
        auto g4 = make_graph(5, 1, 1.23f);
        auto mod4 = compiler_driver(ctx, g4, std::vector<sc_op_ptr> {})
                            ->get_module();
        EXPECT_EQ(mod4->code_, mod1->code_);
    }
    {
        // can share code with g1, because the values of the pure constants are
        // not considered in graph compare
        auto g2 = make_graph(0, 1, 10.23f);
        auto mod2 = compiler_driver(ctx, g2, std::vector<sc_op_ptr> {})
                            ->get_module();
        EXPECT_EQ(mod1->code_, mod2->code_);
        ASSERT_EQ(mod2->globals_.shared_tensors_.size(), 3UL);
        EXPECT_EQ(mod1->globals_.shared_tensors_[1],
                mod2->globals_.shared_tensors_[1]);
    }

    {
        // g1 is exactly the same as g2, but cannot share code because of
        // different contexts
        auto ctx2 = std::make_shared<context_t>(*ctx);
        auto g2 = make_graph(0, 1, 1.23f);
        auto mod2 = compiler_driver(ctx2, g2, std::vector<sc_op_ptr> {})
                            ->get_module();
        EXPECT_NE(mod1->code_, mod2->code_);
    }

    // release the compiled module and check that the cached code has been
    // removed
    mod1 = nullptr;
    EXPECT_EQ(query_cached_code_of_context(ctx), 0UL);
}
