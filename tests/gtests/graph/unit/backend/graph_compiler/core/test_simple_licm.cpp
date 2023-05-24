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
#include <compiler/ir/transform/loop_function_motion.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/simple_licm.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
static constexpr auto s32 = datatypes::s32;
TEST(GCCore_CPU_simple_licm_cpp, TestSimpleLICMTransform) {
    builder::ir_builder_t builder;
    auto dim1 = builder::make_var(s32, "dim1");
    dim1->attr().set(attr_key::const_attr, true);
    auto dim2 = builder::make_var(s32, "dim2");
    _function_(s32, ccc, _arg_("A", s32, {100}), _arg_("B", s32, {100})) {
        _bind_(A, B);

        _for_(i, 0, 100) {
            _tensor_(c, s32, {100});
            _var_init_(d, s32, 3);
            d = c[0] + 3;
            _tensor_(e, s32, {dim1, 100});
            e[{0, 0}] = d;
            _tensor_(g, s32, {100});
            g.get().checked_as<tensor>()->init_value_
                    = tensor_node::get_zero_tensor_initializer();
            _for_(j, 0, 100) {
                _tensor_(f, s32, {200});
                _tensor_(h, s32, {i, 100});
                _tensor_(l, s32, {dim2, 100});
                f[1] = d;
                g[j] = g[j] + 1;
                h[0] = 2;
            }
        }
        _return_(1);
    }

    simple_loop_invariant_code_motion_t licm;
    auto out = licm(ccc);

    _function_(s32, expected, _arg_("A", s32, {100}), _arg_("B", s32, {100})) {
        _bind_(A, B);
        builder.push_scope();
        _tensor_(c, s32, {100});
        _tensor_(e, s32, {dim1, 100});
        _tensor_(f, s32, {200});
        _for_(i, 0, 100) {
            builder.push_scope();
            builder.emit(builder.pop_scope());
            _var_init_(d, s32, 3);
            d = c[0] + 3;
            builder.push_scope();
            builder.emit(builder.pop_scope());
            e[{0, 0}] = d;
            _tensor_(g, s32, {100});
            g.get().checked_as<tensor>()->init_value_
                    = tensor_node::get_zero_tensor_initializer();
            _for_(j, 0, 100) {
                builder.push_scope();
                builder.emit(builder.pop_scope());
                _tensor_(h, s32, {i, 100});
                _tensor_(l, s32, {dim2, 100});
                f[1] = d;
                g[j] = g[j] + 1;
                h[0] = 2;
            }
        }
        builder.emit(builder.pop_scope());
        _return_(1);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_simple_licm_cpp, TestSimpleLICMMultiScope) {
    builder::ir_builder_t builder;
    auto dim1 = builder::make_var(s32, "dim1");
    dim1->attr().set(attr_key::const_attr, true);
    auto dim2 = builder::make_var(s32, "dim2");
    _function_(s32, ccc, _arg_("A", s32, {100}), _arg_("B", s32, {100})) {
        _bind_(A, B);

        _for_(i, 0, 100, 1, for_type::PARALLEL) {
            _for_(j, 0, 100, 1, for_type::PARALLEL) {
                _tensor_(c, s32, {100});
                _for_(k, 0, 100) {
                    _tensor_(d, s32, {i, 100});
                    _for_(l, 0, 100) {
                        _tensor_(e, s32, {dim1, 100});
                        _for_(m, 0, 100) {
                            _tensor_(f, s32, {dim2});
                            _tensor_(g, s32, {100});
                            g.get().checked_as<tensor>()->init_value_
                                    = tensor_node::
                                            get_zero_tensor_initializer();
                            _tensor_(h, s32, {i, k});
                        }
                        _for_(n, 0, 100) {
                            _tensor_(q, s32, {i, 100});
                            _tensor_(t, s32, {100});
                        }
                    }
                }
            }
        }
        _return_(1);
    }

    simple_loop_invariant_code_motion_t licm;
    auto out = licm(ccc);

    _function_(s32, expected, _arg_("A", s32, {100}), _arg_("B", s32, {100})) {
        _bind_(A, B);

        _for_(i, 0, 100, 1, for_type::PARALLEL) {
            _for_(j, 0, 100, 1, for_type::PARALLEL) {
                _tensor_(c, s32, {100});
                builder.push_scope();
                _tensor_(e, s32, {dim1, 100});
                _tensor_(t, s32, {100});
                _for_(k, 0, 100) {
                    _tensor_(d, s32, {i, 100});
                    _for_(l, 0, 100) {
                        builder.push_scope();
                        builder.emit(builder.pop_scope());
                        _for_(m, 0, 100) {
                            _tensor_(f, s32, {dim2});
                            _tensor_(g, s32, {100});
                            g.get().checked_as<tensor>()->init_value_
                                    = tensor_node::
                                            get_zero_tensor_initializer();
                            _tensor_(h, s32, {i, k});
                        }
                        _for_(n, 0, 100) {
                            _tensor_(q, s32, {i, 100});
                            builder.push_scope();
                            builder.emit(builder.pop_scope());
                        }
                    }
                }
                builder.emit(builder.pop_scope());
            }
        }
        _return_(1);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_loop_function_motion_cpp, TestPureFunctionMotion) {
    builder::ir_builder_t builder;
    _function_(s32, ccc, _arg_("A", datatypes::pointer, {10000})) {
        _bind_(A);
        _var_init_(t0, datatypes::s32,
                builder::make_call(builtin::get_thread_id_func(), {}));
        _for_(i, 0, 10) {
            builder.push_scope();
            {
                _var_init_(f0, datatypes::boolean,
                        builder::make_call(builtin::get_is_in_parallel_func(),
                                {})); // can hoist
            }
            builder.emit(builder.pop_scope());
            builder.push_scope();
            {
                _evaluate_call_(builtin::get_barrier_arrive_func(),
                        A[0]); // can't hoist
            }
            builder.emit(builder.pop_scope());
        }
        _for_(i, 0, 10) {
            _for_(j, 0, 10) {
                A[builder::make_call(builtin::get_thread_id_func(), {}) + i + j]
                        = 0; // can hoist
            }
        }
        _return_(0);
    }

    simple_loop_function_motion_t licm;
    auto out = licm(ccc);

    _function_(s32, expected, _arg_("A", datatypes::pointer, {10000})) {
        _bind_(A);
        _var_init_(t0, datatypes::s32,
                builder::make_call(builtin::get_thread_id_func(), {}));
        builder.push_scope();
        _var_init_(call_var_sc_is_in_parallel, datatypes::s32,
                builder::make_call(builtin::get_is_in_parallel_func(), {}));
        _for_(i, 0, 10) {
            builder.push_scope();
            { _var_init_(f0, datatypes::boolean, call_var_sc_is_in_parallel); }
            builder.emit(builder.pop_scope());
            builder.push_scope();
            { _evaluate_call_(builtin::get_barrier_arrive_func(), A[0]); }
            builder.emit(builder.pop_scope());
        }
        builder.emit(builder.pop_scope());

        builder.push_scope();
        _var_init_(call_var_sc_get_thread_id, datatypes::s32,
                builder::make_call(builtin::get_thread_id_func(), {}));
        _for_(i, 0, 10) {
            _for_(j, 0, 10) { A[call_var_sc_get_thread_id + i + j] = 0; }
        }
        builder.emit(builder.pop_scope());
        _return_(0);
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_simple_licm_cpp, TestSimpleLICMVarMotion) {
    builder::ir_builder_t builder;
    auto dim1 = builder::make_var(s32, "dim1");
    dim1->attr().set(attr_key::const_attr, true);
    auto dim2 = builder::make_var(s32, "dim2");
    _function_(s32, ccc, _arg_("A", s32, {100}), _arg_("B", s32, {100})) {
        _bind_(A, B);

        _for_(i, 0, 100) {
            _var_(var1, s32);
            _tensor_(c, s32, {100});
            _var_init_(d, s32, 3);
            d = c[0] + 3;
            _tensor_(e, s32, {dim1, 100});
            e[{0, 0}] = d;
            _tensor_(g, s32, {100});
            var1 = 10;
            g.get().checked_as<tensor>()->init_value_
                    = tensor_node::get_zero_tensor_initializer();
            _for_(j, 0, 100) {
                _tensor_(f, s32, {200});
                _tensor_(h, s32, {i, 100});
                _tensor_(l, s32, {dim2, 100});
                var1 = var1 + 1;
                f[1] = d;
                g[j] = g[j] + 1;
                _var_(var2, s32);
                var2 = 5;
                h[0] = 2;
            }
        }
        _return_(1);
    }

    simple_loop_invariant_code_motion_t licm;
    auto out = licm(ccc);

    _function_(s32, expected, _arg_("A", s32, {100}), _arg_("B", s32, {100})) {
        _bind_(A, B);
        builder.push_scope();
        _var_(var1, s32);
        _tensor_(c, s32, {100});
        _tensor_(e, s32, {dim1, 100});
        _tensor_(f, s32, {200});
        _var_(var2, s32);
        _for_(i, 0, 100) {
            builder.push_scope();
            builder.emit(builder.pop_scope());
            builder.push_scope();
            builder.emit(builder.pop_scope());
            _var_init_(d, s32, 3);
            d = c[0] + 3;
            builder.push_scope();
            builder.emit(builder.pop_scope());
            e[{0, 0}] = d;
            _tensor_(g, s32, {100});
            var1 = 10;
            g.get().checked_as<tensor>()->init_value_
                    = tensor_node::get_zero_tensor_initializer();
            _for_(j, 0, 100) {
                builder.push_scope();
                builder.emit(builder.pop_scope());
                _tensor_(h, s32, {i, 100});
                _tensor_(l, s32, {dim2, 100});
                var1 = var1 + 1;
                f[1] = d;
                g[j] = g[j] + 1;
                builder.push_scope();
                builder.emit(builder.pop_scope());
                var2 = 5;
                h[0] = 2;
            }
        }
        builder.emit(builder.pop_scope());
        _return_(1);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
