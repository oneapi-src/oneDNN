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
#include <iostream>
#include "context.hpp"
#include "test_utils.hpp"
#include "gtest/gtest.h"
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/ir_module.hpp>
#include <compiler/ir/transform/parallel_workload_attr.hpp>
#include <compiler/ir/transform/tensor_init.hpp>

using namespace dnnl::impl::graph::gc;
TEST(GCCore_CPU_tensor_init_cpp, TestTensorInit) {
    REQUIRE_PARALLEL();
    REQUIRE_AVX();
    builder::ir_builder_t builder;

    _function_(datatypes::s32, bbb, _arg_("a", datatypes::f32, {128})) {
        _bind_(a);
        a.static_as<tensor>()->init_value_
                = tensor_node::make_tensor_initializer(2.123f);
        _tensor_(b, datatypes::f32, 260);
        b.static_as<tensor>()->init_value_
                = tensor_node::get_zero_tensor_initializer();
        _var_(vvv, datatypes::index);
        _tensor_(c, datatypes::f32, vvv);
        c.static_as<tensor>()->init_value_
                = tensor_node::get_zero_tensor_initializer();
        _for_(t, 0, 100, 1, for_type::PARALLEL) {
            _tensor_(d, datatypes::f32, 128);
            d.static_as<tensor>()->init_value_
                    = tensor_node::get_zero_tensor_initializer();
        }
        _return_(1);
    }

    tensor_init_t pass {get_test_ctx()};
    auto f = pass(bbb);

    uint64_t simdlen = get_test_ctx()->get_max_vector_lanes(sc_data_etype::F32);
    _function_(datatypes::s32, expected, _arg_("a", datatypes::f32, {128})) {
        _bind_(a);
        a.static_as<tensor>()->init_value_
                = tensor_node::make_tensor_initializer(2.123f);
        _for_(i, UINT64_C(0), UINT64_C(128), simdlen, for_type::PARALLEL) {
            a[span_t({i}, simdlen)]
                    = make_expr<constant_node>(union_val(2.123f),
                            sc_data_type_t(sc_data_etype::F32, simdlen));
        }
        _tensor_(b, datatypes::f32, 260);
        b.static_as<tensor>()->init_value_
                = tensor_node::get_zero_tensor_initializer();
        _for_(i, UINT64_C(0), UINT64_C(260) / simdlen * simdlen, simdlen,
                for_type::PARALLEL) {
            b[span_t({i}, simdlen)] = make_expr<constant_node>(union_val(0.0f),
                    sc_data_type_t(sc_data_etype::F32, simdlen));
        }
        _for_(i, UINT64_C(260) / simdlen * simdlen, UINT64_C(260),
                UINT64_C(1)) {
            b[i] = 0.0f;
        }
        _var_(vvv, datatypes::index);
        _tensor_(c, datatypes::f32, vvv);
        c.static_as<tensor>()->init_value_
                = tensor_node::get_zero_tensor_initializer();
        _for_(i, UINT64_C(0), vvv, UINT64_C(1), for_type::PARALLEL) {
            c[i] = 0.0f;
        }

        _for_(t, 0, 100, 1, for_type::PARALLEL) {
            _tensor_(d, datatypes::f32, 128);
            d.static_as<tensor>()->init_value_
                    = tensor_node::get_zero_tensor_initializer();
            _for_(i, UINT64_C(0), UINT64_C(128), simdlen) {
                d[span_t({i}, simdlen)]
                        = make_expr<constant_node>(union_val(0.0f),
                                sc_data_type_t(sc_data_etype::F32, simdlen));
            }
        }
        _return_(1);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(f, expected, false));
    auto assign_nd = f->body_.checked_as<stmts>()
                             ->seq_.at(0)
                             .checked_as<for_loop>()
                             ->body_.checked_as<stmts>()
                             ->seq_.at(0);
    EXPECT_EQ(assign_nd->attr().get_or_else(
                      parallel_workload::attr_workload_number, size_t(0)),
            4UL);
}
