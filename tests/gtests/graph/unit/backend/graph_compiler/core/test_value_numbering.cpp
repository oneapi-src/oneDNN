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

#include <functional>
#include <iostream>
#include <vector>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <compiler/ir/transform/value_numbering.hpp>
#include <gtest/gtest.h>
#include <util/utils.hpp>

using namespace dnnl::impl::graph::gc;

static constexpr auto s32 = datatypes::s32;

TEST(GCCore_CPU_value_numbering, TestValueNumbering) {
    builder::ir_builder_t bld;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        A[a] = g + 1;
        A[a] = g + 1;
        A[a] = A[a] + 1;
        A[a] = A[a] + 1;
        A[a + b] = a + 1 + b;
        A[a + b + 1] = 1 + a + b; // check that a+1 and 1+a can be folded
        A[a + b + 2] = a + 1 + b + 2;
        _return_(a);
    }

    auto ssa_ccc = ssa_transform_t()(ccc);
    auto out = value_numbering_t()(ssa_ccc);
    out = ssa_visitor_t().top_level_dispatch(out);

    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        _var_(g, s32);
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        _var_init_(t0, s32, g);
        _var_init_(t2, s32, t0 + 1);
        A[a] = t2;
        _var_init_(t3, s32, g);
        _var_init_(t5, s32, t3 + 1);
        A[a] = t5;

        _var_init_(t6, s32, A[a]);
        _var_init_(t8, s32, t6 + 1);
        A[a] = t8;
        _var_init_(t9, s32, A[a]);
        _var_init_(t11, s32, t9 + 1);
        A[a] = t11;

        _var_init_(t12, s32, a + b);
        _var_init_(t14, s32, a + 1);
        _var_init_(t15, s32, t14 + b);
        A[t12] = t15;
        _var_init_(t18, s32, t12 + 1);
        A[t18] = t15;
        _var_init_(t24, s32, t12 + 2);
        _var_init_(t29, s32, t15 + 2);
        A[t24] = t29;
        _return_(a);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_value_numbering, TestValueNumberingScopes) {
    builder::ir_builder_t bld;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        A[a] = b + 1;
        _var_(c, s32);
        _if_(a == 0) {
            A[a] = a + b;
            A[b] = b + 1;
            c = 1;
            _if_(a == 0) { A[b] = a + b + 2; }
            _else_ { A[a] = a + b + 2; }
            A[b] = a + b + 2;
        }
        _else_ {
            A[a] = a + b;
            A[b] = b + 1;
            c = 2; // check that phi nodes are not altered by const propagation
        }
        A[b] = a + b + 2;
        // values in for loops are unsafe to hoist
        _for_(i, 0, 100) { A[b] = a + 3; }
        A[b] = a + 3;
        _return_(a + b + c);
    }

    auto ssa_ccc = ssa_transform_t()(ccc);
    auto out = value_numbering_t()(ssa_ccc);
    out = ssa_visitor_t().top_level_dispatch(out);

    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        _var_init_(t1, s32, b + 1);
        A[a] = t1;
        _var_init_(t3, datatypes::boolean, a == 0);
        _var_init_(t4, s32, a + b);
        _var_init_(t12, s32, t4 + 2);
        expr c0_c, c1_c;
        _if_(t3) {
            A[a] = t4;
            A[b] = t1;
            _var_init_(c0, s32, 1);
            c0_c = c0;

            _if_(t3) { A[b] = t12; }
            _else_ { A[a] = t12; }
            A[b] = t12;
        }
        _else_ {
            A[a] = t4;
            A[b] = t1;
            _var_init_(c1, s32, 2);
            c1_c = c1;
        }

        _var_init_(c2, s32, builder::make_phi({c0_c, c1_c}));
        A[b] = t12;
        _for_(i, 0, 100) {
            _var_init_(t33, s32, a + 3);
            A[b] = t33;
        }
        _var_init_(t35, s32, a + 3);
        A[b] = t35;
        _var_init_(t16, s32, t4 + c2);
        _return_(t16);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_value_numbering, TestValueNumberingScopes2) {
    builder::ir_builder_t bld;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        {
            bld.push_scope();
            A[a] = a + b;
            bld.emit(bld.pop_scope());
        }
        {
            bld.push_scope();
            A[a] = a + b;
            bld.emit(bld.pop_scope());
        }
        _return_(a);
    }
    auto ssa_ccc = ssa_transform_t()(ccc);
    auto out = value_numbering_t()(ssa_ccc);
    out = ssa_visitor_t().top_level_dispatch(out);

    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        _var_init_(t4, s32, a + b);
        {
            bld.push_scope();
            A[a] = t4;
            bld.emit(bld.pop_scope());
        }
        {
            bld.push_scope();
            A[a] = t4;
            bld.emit(bld.pop_scope());
        }
        _return_(a);
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_value_numbering, TestValueNumberingFor) {
    builder::ir_builder_t bld;
    _function_(s32, ccc, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        _var_ex_(c, s32, linkage::local, 0);
        _var_ex_(d, s32, linkage::local, 0);
        _for_(i, 0, 100) {
            A[builder::make_cast(s32, i) * 100 + 2]
                    = builder::make_cast(s32, i) + a + b;
            A[builder::make_cast(s32, i) * 100]
                    = builder::make_cast(s32, i) + a + b;
            d = d + 1;
            c = c + 1;
        }
        _return_(c + d);
    }

    auto ssa_ccc = ssa_transform_t()(ccc);
    auto out = value_numbering_t()(ssa_ccc);
    out = ssa_visitor_t().top_level_dispatch(out);

    _function_(s32, expected, _arg_("A", s32, {10000}), _arg_("a", s32),
            _arg_("b", s32)) {
        _bind_(A, a, b);
        _var_init_(c, s32, 0);
        _var_init_(d, s32, 0);
        expr c5_c, d3_c;
        _for_(i, 0, 100) {
            _var_init_(t3, s32, builder::make_cast(s32, i));
            _var_init_(t5, s32, t3 * 100);
            _var_init_(t7, s32, t5 + 2);
            _var_init_(t10, s32, t3 + a);
            _var_init_(t12, s32, t10 + b);
            A[t7] = t12;
            A[t5] = t12;

            _var_init_(d2, s32, 0);
            auto d2_def = bld.get_current_scope().body.back();
            _var_init_(d3, s32, d2 + 1);
            d2_def.checked_as<define>()->init_
                    = builder::make_phi({d, d3}, true);

            _var_init_(c4, s32, 0);
            auto c4_def = bld.get_current_scope().body.back();
            _var_init_(c5, s32, c4 + 1);
            c4_def.checked_as<define>()->init_
                    = builder::make_phi({c, c5}, true);
            c5_c = c5;
            d3_c = d3;
        }

        _var_init_(c6, s32, builder::make_phi({c, c5_c}));
        _var_init_(d7, s32, builder::make_phi({d, d3_c}));
        _var_init_(t27, s32, c6 + d7);
        _return_(t27);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_value_numbering, TestValueNumberingFuncCall) {
    builder::ir_builder_t bld;
    _function_(s32, normal_func) { _return_(0); }
    _function_(s32, ccc, _arg_("A", s32, {10000})) {
        _bind_(A);
        _var_ex_(c, s32, linkage::local, // pure func
                builder::make_call(builtin::get_thread_id_func(), {}));
        _var_ex_(d, s32, linkage::local, // pure func
                builder::make_call(builtin::get_thread_id_func(), {}));
        _var_ex_(e, s32, linkage::local, builder::make_call(normal_func, {}));
        _var_ex_(f, s32, linkage::local, builder::make_call(normal_func, {}));
        _for_(i, 0, 100) {
            A[builder::make_cast(s32, i) * 100 + 2] = c + d + e + f;
        }
        _return_(c + d);
    }

    auto ssa_ccc = ssa_transform_t()(ccc);
    auto out = value_numbering_t()(ssa_ccc);
    out = ssa_visitor_t().top_level_dispatch(out);

    _function_(s32, expected, _arg_("A", s32, {10000})) {
        _bind_(A);
        _var_ex_(c, s32, linkage::local, // pure func
                builder::make_call(builtin::get_thread_id_func(), {}));
        _var_ex_(e, s32, linkage::local, builder::make_call(normal_func, {}));
        _var_ex_(f, s32, linkage::local, builder::make_call(normal_func, {}));
        _for_(i, 0, 100) {
            _var_init_(t3, s32, builder::make_cast(s32, i));
            _var_init_(t5, s32, t3 * 100);
            _var_init_(t7, s32, t5 + 2);
            _var_init_(t10, s32, c + c);
            _var_init_(t12, s32, t10 + e);
            _var_init_(t13, s32, t12 + f);
            A[t7] = t13;
        }
        _var_init_(t27, s32, c + c);
        _return_(t27);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
