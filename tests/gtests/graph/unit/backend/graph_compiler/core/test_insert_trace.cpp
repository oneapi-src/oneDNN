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

#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/insert_trace.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
extern int get_last_trace_func_id();

}
} // namespace graph
} // namespace impl
} // namespace dnnl

TEST(GCCore_CPU_insert_trace, TestInsertTrace) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, ccc1, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        _tensor_(B, datatypes::f32, 100);
        A[1010] = A[3010];
        B[span_t({1000}, 8)] = A[span_t({0}, 8)];
    }
    _function_(datatypes::void_t, ccc2, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        A[1010] = A[3010];
        _return_();
    }
    _function_(datatypes::void_t, ccc3, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        A[1010] = A[3010];
        _if_(A[1010] == 1) { _return_(); }
        _return_();
    }

    ir_module_ptr mod = std::make_shared<ir_module_t>(get_default_context());
    mod->add_func({ccc1, ccc2, ccc3});
    auto outmod = trace_inserter_t()(mod);

    int func_id = get_last_trace_func_id();
    _function_(datatypes::void_t, eccc1, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        builder.push_evaluate(builtin::make_trace(func_id - 2, 0, 0));
        _tensor_(B, datatypes::f32, 100);
        A[1010] = A[3010];
        B[span_t({1000}, 8)] = A[span_t({0}, 8)];
        builder.push_evaluate(builtin::make_trace(func_id - 2, 1, 0));
    }
    _function_(datatypes::void_t, eccc2, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        builder.push_evaluate(builtin::make_trace(func_id - 1, 0, 0));
        A[1010] = A[3010];
        builder.push_scope();
        {
            builder.push_evaluate(builtin::make_trace(func_id - 1, 1, 0));
            _return_();
        }
        builder.emit(builder.pop_scope());
    }
    _function_(datatypes::void_t, eccc3, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        builder.push_evaluate(builtin::make_trace(func_id, 0, 0));
        A[1010] = A[3010];
        _if_(A[1010] == 1) {
            builder.push_scope();
            {
                builder.push_evaluate(builtin::make_trace(func_id, 1, 0));
                _return_();
            }
            builder.emit(builder.pop_scope());
        }
        builder.push_scope();
        {
            builder.push_evaluate(builtin::make_trace(func_id, 1, 0));
            _return_();
        }
        builder.emit(builder.pop_scope());
    }
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(outmod->get_contents()[0], eccc1));
    EXPECT_TRUE(cmper.compare(outmod->get_contents()[1], eccc2));
    EXPECT_TRUE(cmper.compare(outmod->get_contents()[2], eccc3));
}
