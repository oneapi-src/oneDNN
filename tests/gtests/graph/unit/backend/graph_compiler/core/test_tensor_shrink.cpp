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
#include "exception_util.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_tensor_shrink_cpp, TestTensorShrink) {
    builder::ir_builder_t builder;
    builder.push_scope();
    {
        _tensor_(tsr1, datatypes::f32, 100, 200);
        _tensor_(tsr2, datatypes::f32, 100, 200);
        _tensor_(tsr3, datatypes::f32, 100, 200);
        _var_(ptr, datatypes::pointer);
        expr ptr2;
        _for_(i, 0, 100) {
            auto placeholder
                    = builder::make_stmts_unattached({}).checked_as<stmts>();
            builder.get_current_scope().body.emplace_back(placeholder);
            tsr1[{i, 10}] = 10;
            tsr2[{i, 10}] = 10;
            tsr3[{i, 10}] = 10;
            ptr2 = builder::tensor_ptr(tsr1, {i, 20});
            ptr = ptr2;
            ptr = tsr2;
            ptr2[{i, 10}] = 10;
            tsr1->attr()[tensor_shrinker_attrs::should_shrink]
                    = tensor_shrinker_t::shrink_info_t {
                            /*base*/ {i, 10}, /*shape*/ {1, 190}, stmts()};
            tsr3->attr()[tensor_shrinker_attrs::should_shrink]
                    = tensor_shrinker_t::shrink_info_t {/*base*/ {i, 10},
                            /*shape*/ {1, 190}, placeholder};
        }
    }
    auto body = builder.pop_scope();
    tensor_shrinker_t pass;
    auto after_body = pass(body);

    builder.push_scope();
    {
        _tensor_(tsr1_sh, datatypes::f32, 1, 190);
        _tensor_(tsr2, datatypes::f32, 100, 200);
        builder.get_current_scope().body.emplace_back(
                builder::make_stmts_unattached({}));
        _var_(ptr, datatypes::pointer);
        expr ptr2;
        _for_(i, 0, 100) {
            _tensor_(tsr3_sh, datatypes::f32, 1, 190);
            tsr1_sh[{i - i, expr(10) - 10}] = 10;
            tsr2[{i, 10}] = 10;
            tsr3_sh[{i - i, expr(10) - 10}] = 10;
            ptr = builder::tensor_ptr(tsr1_sh, {i - i, expr(20) - 10});
            ptr = tsr2;
            ptr2 = builder::tensor_ptr(tsr1_sh, {i, expr(20)});
            ptr2[{0, 0}] = 10;
        }
    }
    auto expected = builder.pop_scope();
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(after_body, expected));
}

TEST(GCCore_CPU_tensor_shrink_cpp, TestTensorShrinkUnroll) {
    builder::ir_builder_t builder;
    for_loop loop;
    builder.push_scope();
    {
        _tensor_(tsr1, datatypes::f32, 100, 200);
        _named_for_(loop, i, 0, 1) {
            auto placeholder
                    = builder::make_stmts_unattached({}).checked_as<stmts>();
            builder.get_current_scope().body.emplace_back(placeholder);
            tsr1[{i, 10}] = i;
            tsr1->attr()[tensor_shrinker_attrs::should_shrink]
                    = tensor_shrinker_t::shrink_info_t {
                            /*base*/ {i, 10}, /*shape*/ {1, 190}, placeholder};
            placeholder->attr()[tensor_shrinker_attrs::tensor_for_placerholder]
                    = std::weak_ptr<expr_base>(tsr1.get().impl);
        }
    }
    auto body = builder.pop_scope();
    loop->unroll(0, body);

    tensor_shrinker_t pass;
    auto after_body = pass(body);

    builder.push_scope();
    {
        builder.get_current_scope().emit(builder::make_stmts_unattached({}));
        builder.push_scope();
        {
            _tensor_(tsr1_sh, datatypes::f32, 1, 190);
            tsr1_sh[{expr(0UL) + UINT64_C(0) - (expr(0UL) + UINT64_C(0)),
                    expr(10) - 10}]
                    = (expr(0UL) + UINT64_C(0));
        }
        auto bd = builder.pop_scope();
        builder.get_current_scope().emit(bd);
    }
    auto expectedbody = builder.pop_scope();

    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(after_body, expectedbody));
}

TEST(GCCore_CPU_tensor_shrink_cpp, TestTensorShrinkFail) {
    builder::ir_builder_t builder;
    builder.push_scope();
    {
        _tensor_(tsr1, datatypes::f32, 100, 200);
        _var_(ptr, datatypes::pointer);
        _for_(i, 0, 100) {
            tsr1[{i, 10}] = 10;
            ptr = tsr1;
            tsr1->attr()[tensor_shrinker_attrs::should_shrink]
                    = tensor_shrinker_t::shrink_info_t {
                            /*base*/ {i, 10}, /*shape*/ {1, 190}, stmts()};
        }
    }
    auto body = builder.pop_scope();
    tensor_shrinker_t pass;
    EXPECT_SC_ERROR(
            pass(body), "The shrinked tensor is referenced without indexing");

    builder.push_scope();
    {
        auto tsr1 = builder::make_tensor("tsr1", {100, 200}, datatypes::f32);
        tsr1[{0, 10}] = 10;
        tsr1->attr()[tensor_shrinker_attrs::should_shrink]
                = tensor_shrinker_t::shrink_info_t {
                        /*base*/ {0, 10}, /*shape*/ {1, 190}, stmts()};
    }
    body = builder.pop_scope();
    EXPECT_SC_ERROR(pass(body), "Tensor used before definition");

    builder.push_scope();
    {
        _tensor_(tsr1, datatypes::f32, 100, 200);
        tsr1[{0, 10, 0}] = 10;
        tsr1->attr()[tensor_shrinker_attrs::should_shrink]
                = tensor_shrinker_t::shrink_info_t {
                        /*base*/ {0, 10}, /*shape*/ {1, 190}, stmts()};
    }
    body = builder.pop_scope();
    EXPECT_SC_ERROR(pass(body), "Bad number of dimensions for indexing access");

    builder.push_scope();
    {
        _tensor_(tsr1, datatypes::f32, 100, 200);
        tsr1[{0, 10}] = 10;
        tsr1->attr()[tensor_shrinker_attrs::should_shrink]
                = tensor_shrinker_t::shrink_info_t {
                        /*base*/ {0, 10}, /*shape*/ {1, 190, 20}, stmts()};
    }
    body = builder.pop_scope();
    EXPECT_SC_ERROR(pass(body), "Bad shape for shrinking the tensor:");

    builder.push_scope();
    {
        _tensor_(tsr1, datatypes::f32, 100, 200);
        builder.get_current_scope().body.back().checked_as<define>()->linkage_
                = linkage::static_local;
        tsr1[{0, 10}] = 10;
        tsr1->attr()[tensor_shrinker_attrs::should_shrink]
                = tensor_shrinker_t::shrink_info_t {
                        /*base*/ {0, 10}, /*shape*/ {1, 190}, stmts()};
    }
    body = builder.pop_scope();
    EXPECT_SC_ERROR(pass(body), "And it should be a local tensor");
}

TEST(GCCore_CPU_tensor_shrink_cpp, TestTensorShrinkBRGEMM) {
    builder::ir_builder_t builder;
    for_loop loop;
    builder.push_scope();
    {
        _tensor_(A, datatypes::s32, 100, 200, 300, 400);
        _tensor_(B, datatypes::s32, 100, 200, 400, 300);
        _tensor_(C, datatypes::s32, 100, 300, 200, 300);
        _for_(i, 0, 100) {
            _for_(j, 0, 200) {
                auto placeholder = builder::make_stmts_unattached({})
                                           .checked_as<stmts>();
                builder.get_current_scope().body.emplace_back(placeholder);
                expr orig_LDC = 200 * 300;
                builtin::brgemm_update(builder::tensor_ptr(A, {i, j, 0, 0}),
                        builder::tensor_ptr(B, {i, j, 0, 0}),
                        // discontinuous memory access for C buffer
                        builder::tensor_ptr(C, {i, 0, j, 0}), 20, 300, 300, 400,
                        /*LDA*/ 400, /*LDB*/ 300, /*LDC*/ orig_LDC,
                        /*stride_A*/ 300 * 400, /*stride_B*/ 400 * 300,
                        datatypes::s32, datatypes::s32);
                C->attr()[tensor_shrinker_attrs::should_shrink]
                        = tensor_shrinker_t::shrink_info_t {
                                /*base*/ {i, 0, j, 0},
                                /*shape*/ {1, 300, 1, 300}, placeholder};
                placeholder
                        ->attr()[tensor_shrinker_attrs::tensor_for_placerholder]
                        = std::weak_ptr<expr_base>(C.get().impl);
            }
        }
    }
    auto body = builder.pop_scope();

    tensor_shrinker_t pass;
    auto after_body = pass(body);

    builder.push_scope();
    {
        _tensor_(A, datatypes::s32, 100, 200, 300, 400);
        _tensor_(B, datatypes::s32, 100, 200, 400, 300);
        builder.get_current_scope().emit(builder::make_stmts_unattached({}));
        _for_(i, 0, 100) {
            _for_(j, 0, 200) {
                _tensor_(C_shr, datatypes::s32, 1, 300, 1, 300);
                builtin::brgemm_update(builder::tensor_ptr(A, {i, j, 0, 0}),
                        builder::tensor_ptr(B, {i, j, 0, 0}),
                        builder::tensor_ptr(C_shr,
                                {i - i, expr(0) - expr(0), j - j,
                                        expr(0) - expr(0)}),
                        20, 300, 300, 400, /*LDA*/ 400, /*LDB*/ 300,
                        /*LDC*/
                        300 /*instead of 200*300 */,
                        /*stride_A*/ 300 * 400, /*stride_B*/ 400 * 300,
                        datatypes::s32, datatypes::s32);
            }
        }
    }
    auto expectedbody = builder.pop_scope();
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(after_body, expectedbody));
}
