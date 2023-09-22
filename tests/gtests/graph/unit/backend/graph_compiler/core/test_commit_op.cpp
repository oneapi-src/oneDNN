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

#include "context.hpp"
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <ops/templates/commit_op.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
constexpr auto f32 = datatypes::f32;

TEST(GCCore_CPU_commit_op, TestCommitOP) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, ccc, _arg_("A", f32, {512, 1024}),
            _arg_("B", f32, {32, 128, 16, 16}),
            _arg_("C", f32, {32, 128, 16, 16})) {
        _bind_(A, B, C);
        _tensor_(tmp, f32, 32, 128, 16, 16);
        _for_(i, 0, 32) {
            _for_(j, 0, 32) {
                tmp->attr()[tensor_shrinker_attrs::should_shrink]
                        = tensor_shrinker_t::shrink_info_t {
                                /*base*/ {i, j, 0, 0}, /*shape*/ {1, 1, 16, 16},
                                /*move def*/ stmts()};
                // think that A is output of a plain matmul
                ops::commit_op(get_test_ctx(), "reorder",
                        /*inslice*/
                        {tensor_slice(A, {{i * 16, 16}, {j * 16, 16}})},
                        /*outslice*/
                        {tensor_slice(tmp, {{i, 1}, {j, 1}, {0, 16}, {0, 16}})},
                        /*in*/
                        {graph_tensor::make(
                                {512, 1024}, sc_data_format_t::MK())},
                        /*out*/ {},
                        /*attr*/
                        {{"out_format", sc_data_format_t::MKmk(16, 16)}});
                ops::commit_op(get_test_ctx(), "add",
                        /*inslice*/
                        {tensor_slice(tmp, {{i, 1}, {j, 1}, {0, 16}, {0, 16}}),
                                tensor_slice(
                                        B, {{i, 1}, {j, 1}, {0, 16}, {0, 16}})},
                        {tensor_slice(C, {{i, 1}, {j, 1}, {0, 16}, {0, 16}})},
                        {graph_tensor::make({32, 128, 16, 16}),
                                graph_tensor::make({32, 128, 16, 16})});
            }
        }
    }
    auto constexpr UL16 = UINT64_C(16);
    int simd_len = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    auto build_expected = [&](bool shrink) {
        _function_(datatypes::void_t, expected, _arg_("A", f32, {512, 1024}),
                _arg_("B", f32, {32, 128, 16, 16}),
                _arg_("C", f32, {32, 128, 16, 16})) {
            _bind_(A, B, C);
            _tensor_(tmp, f32, shrink ? 1 : 32, shrink ? 1 : 128, 16, 16);
            _for_(i, 0, 32) {
                _for_(j, 0, 32) {
                    _for_(f0, 0, 16) {
                        _for_(f1, 0, 16, simd_len) {
                            std::vector<expr> idx = {(f0 + i * UL16) / 16,
                                    (f1 + j * UL16) / 16, (f0 + i * UL16) % 16,
                                    (f1 + j * UL16) % 16};
                            if (shrink) {
                                idx[0] = idx[0] - i;
                                idx[1] = idx[1] - j;
                                idx[2] = idx[2] - 0;
                                idx[3] = idx[3] - 0;
                            }
                            tmp[span_t(idx, simd_len)] = builder::tensor_ptr(A,
                                    {i * UL16, j * UL16}, {},
                                    true)[span_t({f0, f1}, simd_len)];
                        }
                    }
                    _for_(f4, 0, 16) {
                        _for_(f5, 0, 16, simd_len) {
                            auto c_ptr = builder::tensor_ptr(
                                    C, {i, j, 0, 0}, {}, true);
                            std::vector<expr> idx = {i, j, 0, 0};
                            if (shrink) {
                                idx[0] = idx[0] - i;
                                idx[1] = idx[1] - j;
                                idx[2] = idx[2] - 0;
                                idx[3] = idx[3] - 0;
                            }
                            auto tmp_ptr
                                    = builder::tensor_ptr(tmp, idx, {}, true);
                            tmp_ptr.checked_as<tensorptr>()->shape_
                                    = {32, 128, 16, 16};
                            auto B_ptr = builder::tensor_ptr(
                                    B, {i, j, 0, 0}, {}, true);
                            c_ptr[span_t({0, 0, f4, f5}, simd_len)]
                                    = tmp_ptr[span_t({0, 0, f4, f5}, simd_len)]
                                    + B_ptr[span_t({0, 0, f4, f5}, simd_len)];
                        }
                    }
                }
            }
        }
        return expected;
    };

    ir_comparer cmper(true);
    EXPECT_TRUE(cmper.compare(ccc, build_expected(false)));
    EXPECT_TRUE(cmper.compare(
            tensor_shrinker_t()(ccc), build_expected(true), false));
}
