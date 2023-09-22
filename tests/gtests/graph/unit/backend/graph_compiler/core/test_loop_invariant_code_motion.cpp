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

#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/ssa_visitor.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/loop_invariant_code_motion.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <util/any_map.hpp>
#include <util/def.hpp>

#if SC_BUILTIN_JIT_ENABLED
#include <compiler/jit/xbyak/ir/transform/indexing_transform.hpp>
#endif

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
constexpr auto s32 = datatypes::s32;
constexpr auto f32 = datatypes::f32;
constexpr auto idx = datatypes::index;
TEST(GCCore_CPU_licm_transform, TestLICMTransformSingleLoop) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {100}), _arg_("B", s32, {100}),
            _arg_("a", s32)) {
        _bind_(A, B, a);
        _var_init_(c, s32, 20);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        g = 1;
        _for_(i, 0, 100) {
            _var_init_(d, s32, 10);
            _var_init_(e, s32, a);
            g = d + 2;
            g = e + 3;
            _var_init_(
                    f, sc_data_type_t::s32(16), builder::make_broadcast(2, 16));
            _var_init_(h, sc_data_type_t::s32(16),
                    builder::make_broadcast(builder::make_cast(s32, i), 16));
            B[span_t({0}, 16)] = f + h;
            _tensor_(C, s32, {100});
            A[i] = i;
            B[i] = 3;
            c = C[0];
        }
        _return_(c);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    loop_invariant_code_motion_t licm;
    out = licm(out);

    _function_(s32, expected, _arg_("A", s32, {100}), _arg_("B", s32, {100}),
            _arg_("a", s32)) {
        _bind_(A, B, a);
        _var_init_(c, s32, 20);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        _var_init_(t0, s32, 1);
        g = t0;
        _var_init_(t1, s32, 0);
        _var_init_(t2, s32, 100);
        _var_init_(t3, s32, 1);
        _var_init_(d, s32, 10);
        _var_init_(a_0, s32, builder::make_phi({a}));
        _var_init_(e, s32, a_0);
        _var_init_(t5, s32, 2);
        _var_init_(t6, s32, d + t5);
        _var_init_(t7, s32, 3);
        _var_init_(t8, s32, e + t7);
        _var_init_(t9, s32, 2);
        _var_init_(f, sc_data_type_t::s32(16), builder::make_broadcast(t9, 16));
        _var_init_(t11, s32, 0);
        _var_init_(t13, s32, 3);
        _var_init_(t14, s32, 0);

        expr c1_;
        _for_(i, t1, t2, t3) {
            g = t6;
            g = t8;
            _var_init_(t10, s32, builder::make_cast(s32, i));
            _var_init_(h, sc_data_type_t::s32(16),
                    builder::make_broadcast(t10, 16));
            _var_init_(t12, sc_data_type_t::s32(16), f + h);
            B[span_t({t11}, 16)] = t12;
            _tensor_(C, s32, {100});
            A[i] = i;
            B[i] = t13;
            _var_init_copy_(c1, s32, C[t14]);
        }
        _var_init_(c2, s32, builder::make_phi({c, c1_}));
        _return_(c2);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_licm_transform, TestLICMTransformComplexLoop) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", idx, {100, 200, 300}),
            _arg_("B", idx, {100, 200, 300}), _arg_("C", idx, {100, 200, 300}),
            _arg_("D", idx, {100, 200, 300})) {
        _bind_(A, B, C, D);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        g = 1;
        _for_(l0, 0, 100) {
            // out of l0
            _var_init_(c0, s32, 8);
            _for_(l1, 0, 100) {
                // out of l1
                _var_init_(c1, idx, l0 + c0);
                // don't hoist
                _var_init_(v1, idx, c1 + l1);
                // don't hoist
                g = c0;
                // don't hoist
                _var_init_(r1, idx, 0);
                _for_(l2, 0, 300) {
                    A[l0 + l1] = B[l0];
                    C[l0] = c1;
                    // don't hoist
                    D[l2 + v1] = c1;
                    // don't hoist
                    g = c1;
                    // don't hoist
                    r1 = r1 + UINT64_C(1);
                }
                // don't hoist
                g = r1;
            }
            _for_(l3, 0, 100) {
                // out of l0
                _var_init_(c2, s32, 13);
                // out of l3
                _var_init_(v2, idx, l0 + c2);
                A[c2] = c2;
                B[v2] = c2;
                // don't hoist
                C[v2 + l3] = c2;
                // don't hoist
                g = c2;
            }
            // don't hoist
            A[0] = g;
        }
        _return_(g);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    loop_invariant_code_motion_t licm;
    out = licm(out);

    // expected
    _function_(s32, expected, _arg_("A", idx, {100, 200, 300}),
            _arg_("B", idx, {100, 200, 300}), _arg_("C", idx, {100, 200, 300}),
            _arg_("D", idx, {100, 200, 300})) {
        _bind_(A, B, C, D);
        _var_(g, s32); // simulate global var
        g->attr()[attr_keys::module_global_offset] = size_t(1);
        _var_init_(t0, s32, 1);
        g = t0;
        _var_init_(t1, s32, 0);
        _var_init_(t2, s32, 100);
        _var_init_(t3, s32, 1);
        _var_init_(c0, s32, 8);
        _var_init_(t4, s32, 0);
        _var_init_(t5, s32, 100);
        _var_init_(t6, s32, 1);
        _var_init_(c0_1, s32, builder::make_phi({c0}));
        _var_init_(t9, s32, 0);
        _var_init_(t10, s32, 300);
        _var_init_(t11, s32, 1);
        _var_init_(tr1, idx, UINT64_C(1));
        _var_init_(t19, s32, 0);
        _var_init_(t20, s32, 100);
        _var_init_(t21, s32, 1);
        _var_init_(c2, s32, 13);
        _var_init_(t24, s32, 0);
        _var_init_(t25, s32, g);
        _for_(l0, t1, t2, t3) {
            _var_init_(l0_0, idx, builder::make_phi({l0}));
            _var_init_(c1, idx, l0_0 + c0_1);
            _var_init_(l0_2, idx, builder::make_phi({l0_0}));
            _var_init_(t15, idx, B[l0_2]);
            _var_init_(c1_4, idx, builder::make_phi({c1}));
            _for_(l1, t4, t5, t6) {
                _var_init_(v1, idx, c1 + l1);
                g = c0_1;
                _var_init_(r1, idx, 0);
                _var_init_(l1_3, idx, builder::make_phi({l1}));
                _var_init_(t14, idx, l0_2 + l1_3);
                _var_init_(v1_5, idx, builder::make_phi({v1}));
                expr r1_7 = builder::make_var(idx, "r1_7");
                _for_(l2, t9, t10, t11) {
                    A[t14] = t15;
                    C[l0_2] = c1_4;
                    _var_init_(t18, idx, l2 + v1_5);
                    D[t18] = c1_4;
                    g = c1_4;
                    _var_init_(r1_6, idx, builder::make_phi({r1, r1_7}, true));
                    builder::get_current_builder()->push_var_tensor_def(
                            r1_7, linkage::local, r1_6 + tr1);
                }
                _var_init_(r1_8, idx, builder::make_phi({r1, r1_7}));
                g = r1_8;
            }
            _var_init_(l0_6, idx, builder::make_phi({l0}));
            _var_init_(v2, idx, l0_6 + c2);
            _for_(l3, t19, t20, t21) {
                A[c2] = c2;
                B[v2] = c2;
                _var_init_(t23, idx, v2 + l3);
                C[t23] = c2;
                g = c2;
            }
            A[t24] = t25;
        }
        _var_init_(t26, s32, g);
        _return_(t26);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_licm_transform, TestLICMTransformIfNodeHoist) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", s32, {10000})) {
        _bind_(A);
        // can't hoist
        _var_(a, s32);
        _for_(i, 0, 10) {
            _if_(true) { a = 1; }
            _else_ { a = A[i]; }
            A[a] = a;
        }
        // can't hoist
        _var_(c, s32);
        _for_(i, 0, 10) {
            _if_(true) { c = A[i]; }
            A[c] = c;
        }
        // can't hoist
        _var_(e, s32);
        _var_init_(f, s32, 1);
        _for_(i, 0, 10) {
            _if_(i == 0) { f = 3; }
            _if_(f == 0) { e = 1; }
            _else_ { e = 2; }
            A[e] = e;
        }
        // can hoist
        _var_(m, s32);
        _for_(i, 0, 10) {
            _if_(true) {
                _var_init_(tmp, s32, 1);
                m = 1;
            }
            _else_ { m = 2; }
            A[m] = m;
        }
        // can hoist inside if scope
        _var_init_(d, s32, 0);
        _for_(i, 0, 10) {
            _if_(true) {
                _for_(i1, 0, 10) {
                    _var_init_(tmp, s32, d + 2);
                    A[i1] = tmp;
                }
            }
            _else_ {
                _for_(i2, 0, 10) {
                    _var_init_(tmp2, s32, d + 1);
                    A[i2] = tmp2;
                }
            }
        }
        // can hoist
        _var_init_(m1, s32, 0);
        _var_init_(m2, s32, 0);
        _for_(i, 0, 10) {
            _if_(true) {
                _for_(j, 0, 10) {
                    _var_init_(tmp, s32, m2 + 1);
                    m1 = tmp;
                }
            }
            _else_ { m1 = 2; }
            A[m1] = m1;
        }

        _return_(0);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    loop_invariant_code_motion_t licm;
    out = licm(out);

    _function_(s32, expected, _arg_("A", s32, {10000})) {
        _bind_(A);
        //
        _var_init_(t0, s32, 0);
        _var_init_(t1, s32, 10);
        _var_init_(t2, s32, 1);
        _var_init_(t3, datatypes::boolean, true);
        expr a_1_, a_2_;
        _for_(i, t0, t1, t2) {
            _if_(t3) { _var_init_copy_(a_1, s32, 1); }
            _else_ { _var_init_copy_(a_2, s32, A[i]); }
            _var_init_(a_3, s32, builder::make_phi({a_1_, a_2_}));
            A[a_3] = a_3;
        }
        //
        _var_init_(t8, s32, 0);
        _var_init_(t9, s32, 10);
        _var_init_(t10, s32, 1);
        _var_init_(t11, datatypes::boolean, true);
        _for_(i, t8, t9, t10) {
            expr c_6 = builder::make_var(s32, "c6");
            _var_init_(c_5, s32, builder::make_phi({0, c_6}, true));
            expr c_4_;
            _if_(t11) { _var_init_copy_(c_4, s32, A[i]); }
            builder::get_current_builder()->push_var_tensor_def(
                    c_6, linkage::local, builder::make_phi({c_5, c_4_}));
            A[c_6] = c_6;
        }
        //
        _var_init_(f, s32, 1);
        _var_init_(t15, s32, 0);
        _var_init_(t16, s32, 10);
        _var_init_(t17, s32, 1);
        _var_init_(t18, s32, 0);
        _var_init_(t22, s32, 0);
        _for_(i, t15, t16, t17) {
            expr f_10 = builder::make_var(s32, "f10");
            _var_init_(t19, datatypes::boolean, i == t18);
            _var_init_(f_9, s32, builder::make_phi({f, f_10}, true));
            expr e_1_, e_2_, f_7_;
            _if_(t19) { _var_init_copy_(f_7, s32, 3); }
            builder::get_current_builder()->push_var_tensor_def(
                    f_10, linkage::local, builder::make_phi({f_9, f_7_}));
            _var_init_(t23, datatypes::boolean, f_10 == t22);
            _if_(t23) { _var_init_copy_(e_1, s32, 1); }
            _else_ { _var_init_copy_(e_2, s32, 2); }
            _var_init_(e_3, s32, builder::make_phi({e_1_, e_2_}));
            A[e_3] = e_3;
        }
        //
        _var_init_(t29, s32, 0);
        _var_init_(t30, s32, 10);
        _var_init_(t31, s32, 1);
        _var_init_(t32, datatypes::boolean, true);
        expr m_1_, m_2_;
        _if_(t32) { _var_init_copy_(m_1, s32, 1); }
        _else_ { _var_init_copy_(m_2, s32, 2); }
        _var_init_(m_3, s32, builder::make_phi({m_1_, m_2_}));
        _for_(i, t29, t30, t31) { A[m_3] = m_3; }
        //
        _var_init_(d, s32, 0);
        _var_init_(t37, s32, 0);
        _var_init_(t38, s32, 10);
        _var_init_(t39, s32, 1);
        _var_init_(d_18, s32, builder::make_phi({d}));
        _var_init_(t40, datatypes::boolean, true);
        _for_(i, t37, t38, t39) {
            _if_(t40) {
                _var_init_(t41, s32, 0);
                _var_init_(t42, s32, 10);
                _var_init_(t43, s32, 1);
                _var_init_(d_19, s32, builder::make_phi({d_18}));
                _var_init_(t46, s32, 2);
                _var_init_(tmp, s32, (d_19 + t46));
                _for_(i1, t41, t42, t43) { A[i1] = tmp; }
            }
            _else_ {
                _var_init_(t47, s32, 0);
                _var_init_(t48, s32, 10);
                _var_init_(t49, s32, 1);
                _var_init_(d_20, s32, builder::make_phi({d_18}));
                _var_init_(t51, s32, 1);
                _var_init_(tmp2, s32, (d_20 + t51));
                _for_(i2, t47, t48, t49) { A[i2] = tmp2; }
            }
        }
        //
        _var_init_(m100, s32, 0);
        _var_init_(m200, s32, 0);
        _var_init_(t50, s32, 0);
        _var_init_(t51, s32, 10);
        _var_init_(t52, s32, 1);
        _var_init_(m201, s32, builder::make_phi({m200}));
        _var_init_(t53, datatypes::boolean, true);
        expr m103_, m104_;
        _if_(t53) {
            _var_init_(t54, s32, 0);
            _var_init_(t55, s32, 10);
            _var_init_(t56, s32, 1);
            _var_init_(m202, s32, builder::make_phi({m201}));
            _var_init_(t59, s32, 1);
            _var_init_(m102, s32, (m202 + t59));
            _for_(j, t54, t55, t56) {}
            _var_init_copy_(m103, s32, builder::make_phi({m100, m102}));
        }
        _else_ { _var_init_copy_(m104, s32, 2); }
        _var_init_(m105, s32, builder::make_phi({m103_, m104_}));
        _for_(i, t50, t51, t52) { A[m105] = m105; }
        //
        _var_init_(t33, s32, 0);
        _return_(t33);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
    std::cout << cmper;
}

TEST(GCCore_CPU_licm_transform, TestLICMTransformIndexing) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", datatypes::s32, {10000}),
            _arg_("B", datatypes::s32, {10000})) {
        _bind_(A, B);
        _for_(i, 0, 10) {
            _for_(j, 0, 10) {
                A[i] = B[0] + builder::make_cast(s32, i); // can hoist B[0]
                A[j] = B[i]; // can hoist B[i]
            }
        }
        _for_(i, 0, 10) {
            B[i] = 0; // canot hoist
            _for_(j, 0, 10) {
                A[i + j] = B[0]; // can hoist B[0] only outside loop j
            }
        }
        _return_(0);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    loop_invariant_code_motion_t licm;
    out = licm(out);

    _function_(s32, expected, _arg_("A", datatypes::s32, {10000}),
            _arg_("B", datatypes::s32, {10000})) {
        _bind_(A, B);
        _var_init_(t0, s32, 0);
        _var_init_(t1, s32, 10);
        _var_init_(t2, s32, 1);
        _var_init_(t3, s32, 0);
        _var_init_(t4, s32, 10);
        _var_init_(t5, s32, 1);
        _var_init_(t7, s32, 0);
        _var_init_(t8, s32, B[t7]);
        _for_(i, t0, t1, t2) {
            _var_init_(i_0, idx, builder::make_phi({i}));
            _var_init_(t9, s32, builder::make_cast(s32, i_0));
            _var_init_(t10, s32, t8 + t9);
            _var_init_(t11, s32, B[i_0]);
            _for_(j, t3, t4, t5) {
                A[i_0] = t10;
                A[j] = t11;
            }
        }

        _var_init_(t12, s32, 0);
        _var_init_(t13, s32, 10);
        _var_init_(t14, s32, 1);
        _var_init_(t15, s32, 0);
        _var_init_(t16, s32, 0);
        _var_init_(t17, s32, 10);
        _var_init_(t18, s32, 1);
        _var_init_(t21, s32, 0);
        _for_(i, t12, t13, t14) {
            B[i] = t15;
            _var_init_(i_1, idx, builder::make_phi({i}));
            _var_init_(t22, s32, B[t21]);
            _for_(j, t16, t17, t18) {
                _var_init_(t20, idx, i_1 + j);
                A[t20] = t22;
            }
        }
        _var_init_(t23, s32, 0);
        _return_(t23);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_licm_transform, TestLICMTransformLoopWithFuncCall) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", datatypes::f32, {10000}),
            _arg_("B", datatypes::f32, {10000}), _arg_("m", datatypes::index),
            _arg_("n", datatypes::index)) {
        _bind_(A, B, m, n);
        _for_(i, 0, 10) {
            _for_(j, 0, 10) {
                _for_(k, 0, 10) {
                    // can hoist
                    A[m + n + i + j + k] = B[0] + 1.f;
                }
            }
            // loop contain func call, don't promote beyond this loop
            _evaluate_call_(builtin::get_barrier_arrive_func(), A);
            _for_(j, 0, 10) {
                // loop contain func call, don't promote beyond this loop
                _evaluate_call_(builtin::get_barrier_arrive_func(), B);
                A[n + i + j] = 2.f;
            }
        }
        _return_(0);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    loop_invariant_code_motion_t licm;
    out = licm(out);

    _function_(s32, expected, _arg_("A", datatypes::f32, {10000}),
            _arg_("B", datatypes::f32, {10000}), _arg_("m", datatypes::index),
            _arg_("n", datatypes::index)) {
        _bind_(A, B, m, n);
        _var_init_(t0, s32, 0);
        _var_init_(t1, s32, 10);
        _var_init_(t2, s32, 1);
        _for_(i, t0, t1, t2) {
            _var_init_(m_0, idx, builder::make_phi({m}));
            _var_init_(n_3, idx, builder::make_phi({n}));
            _var_init_(t3, s32, 0);
            _var_init_(t4, s32, 10);
            _var_init_(t5, s32, 1);
            _var_init_(m_1, idx, builder::make_phi({m_0}));
            _var_init_(n_4, idx, builder::make_phi({n_3}));
            _var_init_(i_6, idx, builder::make_phi({i}));
            _var_init_(t6, s32, 0);
            _var_init_(t7, s32, 10);
            _var_init_(t8, s32, 1);
            _var_init_(m_2, idx, builder::make_phi({m_1}));
            _var_init_(n_5, idx, builder::make_phi({n_4}));
            _var_init_(t11, idx, (m_2 + n_5));
            _var_init_(i_7, idx, builder::make_phi({i_6}));
            _var_init_(t13, idx, (t11 + i_7));
            _var_init_(t17, s32, 0);
            _var_init_(t18, f32, B[t17]);
            _var_init_(t19, f32, 1.f);
            _var_init_(t20, f32, (t18 + t19));
            _for_(j, t3, t4, t5) {
                _var_init_(j_3, idx, builder::make_phi({j}));
                _var_init_(t15, idx, (t13 + j_3));
                _for_(k, t6, t7, t8) {
                    _var_init_(t16, idx, (t15 + k));
                    A[t16] = t20;
                }
            }
            _var_init_(t21, datatypes::void_t,
                    builder::make_call(
                            builtin::get_barrier_arrive_func(), {A}));
            builder.push_evaluate(t21);
            _var_init_(t22, s32, 0);
            _var_init_(t23, s32, 10);
            _var_init_(t24, s32, 1);
            _for_(j, t22, t23, t24) {
                _var_init_(t25, datatypes::void_t,
                        builder::make_call(
                                builtin::get_barrier_arrive_func(), {B}));
                builder.push_evaluate(t25);
                _var_init_(n_4, idx, builder::make_phi({n_3}));
                _var_init_(i_5, idx, builder::make_phi({i}));
                _var_init_(t28, idx, (n_4 + i_5));
                _var_init_(t29, idx, (t28 + j));
                _var_init_(t30, f32, 2.f);
                A[t29] = t30;
            }
        }
        _var_init_(t31, s32, 0);
        _return_(t31);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_licm_transform, TestLICMTransformAliasTensor) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", datatypes::s32, {10000}),
            _arg_("B", datatypes::s32, {10000}),
            _arg_("C", datatypes::s32, {10000}),
            _arg_("D", datatypes::s32, {10000})) {
        _bind_(A, B, C, D);
        {
            auto clique = std::make_shared<alias_info::alias_set_t>();
            alias_info::get_or_create_alias_info(*A.get())->add_to_clique(
                    clique);
            alias_info::get_or_create_alias_info(*B.get())->add_to_clique(
                    clique);
        }
        _for_(i, 0, 10) {
            _for_(j, 0, 10) {
                C[i] = B[0]; // alias A volatile, cannot hoist B[0]
                A[j] = D[0]; // A volatile, D non-volatile
            }
        }
        _return_(0);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    loop_invariant_code_motion_t licm;
    out = licm(out);

    _function_(s32, expected, _arg_("A", datatypes::s32, {10000}),
            _arg_("B", datatypes::s32, {10000}),
            _arg_("C", datatypes::s32, {10000}),
            _arg_("D", datatypes::s32, {10000})) {
        _bind_(A, B, C, D);
        _var_init_(t0, s32, 0);
        _var_init_(t1, s32, 10);
        _var_init_(t2, s32, 1);
        _var_init_(t3, s32, 0);
        _var_init_(t4, s32, 10);
        _var_init_(t5, s32, 1);
        _var_init_(t7, s32, 0);
        _var_init_(t9, s32, 0);
        _var_init_(t10, s32, D[t9]);
        _for_(i, t0, t1, t2) {
            _var_init_(i_0, idx, builder::make_phi({i}));
            _for_(j, t3, t4, t5) {
                _var_init_(t8, s32, B[t7]);
                C[i_0] = t8;
                A[j] = t10;
            }
        }

        _var_init_(t23, s32, 0);
        _return_(t23);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_licm_transform, TestLICMNonLoopPHI) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", datatypes::s32, {10000}),
            _arg_("B", datatypes::s32, {10000})) {
        _bind_(A, B);
        _for_(i, 0, 10) {
            _var_init_(a, datatypes::s32, 0);
            _if_(i > 0) { a = 2; }
            A[0] = a;
            _var_init_(c, datatypes::s32, 1);
            _var_init_(d, datatypes::s32, 2);
            _var_init_(e, datatypes::s32, c + d);
            _if_(0) { e = 5; }
            A[1] = e;
            _var_init_(f, datatypes::s32, 0);
            _if_(0) {
                B[1] = 2;
                f = B[0];
            }
            A[2] = f;
        }
        _return_(0);
    }
    ssa_transform_t s;
    auto out = s(ccc);

    loop_invariant_code_motion_t licm;
    out = licm(out);

    _function_(s32, expected, _arg_("A", datatypes::s32, {10000}),
            _arg_("B", datatypes::s32, {10000})) {
        _bind_(A, B);
        _var_init_(t0, s32, 0);
        _var_init_(t1, s32, 10);
        _var_init_(t2, s32, 1);
        _var_init_(t3, s32, 0);
        _var_init_(t7, s32, 0);
        _var_init_(c, s32, 1);
        _var_init_(d, s32, 2);
        _var_init_(e, s32, c + d);
        expr e2_;
        _var_init_(t8, s32, 0);
        _if_(t8) { _var_init_copy_(e2, s32, 5); }
        _var_init_(e3, s32, builder::make_phi({e, e2_}));
        _var_init_(t12, s32, 1);
        _var_init_(t13, s32, 0);
        _var_init_(t19, s32, 2);

        _for_(i, t0, t1, t2) {
            _var_init_(a, s32, 0);
            _var_init_(t4, datatypes::boolean, i > t3);
            expr a0_;
            _if_(t4) { _var_init_copy_(a0, s32, 2); }
            _var_init_(a1, s32, builder::make_phi({a, a0_}));
            A[t7] = a1;
            A[t12] = e3;
            _var_init_(f, s32, 0);
            expr f4_;
            _if_(t13) {
                _var_init_(t14, s32, 1);
                _var_init_(t15, s32, 2);
                B[t14] = t15;
                _var_init_(t16, s32, 0);
                _var_init_copy_(f4, s32, B[t16]);
            }
            _var_init_(f5, s32, builder::make_phi({f, f4_}));
            A[t19] = f5;
        }

        _var_init_(t20, s32, 0);
        _return_(t20);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

#if SC_BUILTIN_JIT_ENABLED
TEST(GCCore_CPU_licm_transform, TestIndexingTransformLICM) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, original, _arg_("A", f32, {65536UL}),
            _arg_("B", f32, {65536UL}), _arg_("C", f32, {65536UL})) {
        _bind_(A, B, C);
        _for_(idx1, 0UL, 128UL, 1UL) {
            _for_(idx2, 0UL, 512UL, 1UL) {
                _var_init_(a, f32, A[(UINT64_C(512) * idx1) + idx2]);
                _var_init_(b, f32, B[(UINT64_C(512) * idx1) + idx2]);
                C[(UINT64_C(512) * idx1) + idx2] = a + b;
            }
        }
    }

    constant_folder_t c1(false);
    auto out = c1(original);
    xbyak::indexing_transform_t t;
    out = t(out);
    constant_folder_t c2(false);
    out = c2(out);
    ssa_transform_t s;
    out = s(out);
    loop_invariant_code_motion_t licm;
    out = licm(out);

    auto f32_ptr = f32.get_pointerof();
    _function_(datatypes::void_t, expected, _arg_("A", f32, {65536UL}),
            _arg_("B", f32, {65536UL}), _arg_("C", f32, {65536UL})) {
        _bind_(A, B, C);
        _var_init_(t0, idx, 0UL);
        _var_init_(t1, idx, 128UL);
        _var_init_(t2, idx, 1UL);
        _var_init_(t3, idx, 0UL);
        _var_init_(t4, idx, 512UL);
        _var_init_(t5, idx, 1UL);
        _var_init_(t6, idx, builder::make_cast(idx, A));
        _var_init_(t8, idx, 2048UL);
        _var_init_(t12, idx, builder::make_cast(idx, B));
        _var_init_(t13, idx, 2048UL);
        _var_init_(t17, idx, builder::make_cast(idx, C));
        _var_init_(t18, idx, 2048UL);
        _for_(idx1, t0, t1, t2) {
            _var_init_(i1, idx, builder::make_phi({idx1}));
            _var_init_(t9, idx, (i1 * t8));
            _var_init_(t10, idx, (t6 + t9));
            _var_init_(ptr_a, f32_ptr, builder::make_cast(f32_ptr, t10));
            _var_init_(t14, idx, (i1 * t13));
            _var_init_(t15, idx, (t12 + t14));
            _var_init_(ptr_b, f32_ptr, builder::make_cast(f32_ptr, t15));
            _var_init_(t19, idx, (i1 * t18));
            _var_init_(t20, idx, (t17 + t19));
            _var_init_(ptr_c, f32_ptr, builder::make_cast(f32_ptr, t20));
            _for_(idx2, t3, t4, t5) {
                _var_init_(a, f32, ptr_a[idx2]);
                _var_init_(b, f32, ptr_b[idx2]);
                _var_init_(t22, f32, (a + b));
                ptr_c[idx2] = t22;
            }
        }
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
#endif
