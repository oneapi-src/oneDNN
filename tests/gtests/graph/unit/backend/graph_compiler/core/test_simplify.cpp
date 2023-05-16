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

#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_ir_simplify, TestSimplify) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, ccc, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        // no collapse
        _if_(A[0] == 0) {}
        _else_ { A[0] = 0; }

        // remove
        builder.push_scope();
        {
            _var_(aaa, datatypes::s32);
            aaa = 4;
            // remove
            builder.push_scope();
            {}
            builder.emit(builder.pop_scope());
            // promote
            builder.push_scope();
            {
                aaa = 4;
                aaa = 5;
            }
            builder.emit(builder.pop_scope());
        }
        builder.emit(builder.pop_scope());
    }
    ir_simplifier_t s {false};
    auto out = s(ccc);

    _function_(
            datatypes::void_t, expected, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        // no collapse
        _if_(!(A[0] == 0)) { A[0] = 0; }

        _var_(aaa, datatypes::s32);
        aaa = 4;
        aaa = 4;
        aaa = 5;
    }
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_ir_simplify, TestVarRename) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, ccc, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);
        _var_(var0, datatypes::f32);
        _tensor_(var1, datatypes::s32, {1});
        var0 = 1;
        var1[0] = 2;
        // no collapse
        builder.push_scope();
        {
            _var_(var1, datatypes::s32);
            var1 = 4;
            // remove
            builder.push_scope();
            {}
            builder.emit(builder.pop_scope());
            // promote and remove
            builder.push_scope();
            {
                _tensor_(var0, datatypes::f32, {1});
                _var_(var2, datatypes::s32);
                var0[0] = 3;
                var2 = 6;
                var1 = 4;
                var1 = 5;
            }
            builder.emit(builder.pop_scope());
            _var_(var2, datatypes::f32);
            var2 = 7;
        }
        builder.emit(builder.pop_scope());
    }
    ir_simplifier_t s {false};
    auto out = s(ccc);

    _function_(
            datatypes::void_t, expected, _arg_("A", datatypes::f32, {10000})) {
        _bind_(A);

        _var_(var0, datatypes::f32);
        _tensor_(var1, datatypes::s32, {1});
        var0 = 1;
        var1[0] = 2;

        _var_(var1_1, datatypes::s32);
        var1_1 = 4;

        _tensor_(var0_2, datatypes::f32, {1});
        _var_(var2, datatypes::s32);
        var0_2[0] = 3;
        var2 = 6;
        var1_1 = 4;
        var1_1 = 5;

        _var_(var2_3, datatypes::f32);
        var2_3 = 7;
    }
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_ir_simplify, TestLoopVarRename) {
    builder::ir_builder_t builder;
    _function_(datatypes::void_t, ccc, _arg_("A", datatypes::index, {300})) {
        _bind_(A);
        _for_(iter, 0, 100) {
            auto &l1_iter = iter;
            _for_(iter, 0, 100) {
                auto &l2_iter = iter;
                _for_(iter, 0, 100) {
                    auto &l3_iter = iter;
                    A[l1_iter + l2_iter + l3_iter]
                            = l1_iter + l2_iter + l3_iter;
                }
            }
        }
    }
    ir_simplifier_t s {false};
    auto out = s(ccc);

    _function_(
            datatypes::void_t, expected, _arg_("A", datatypes::index, {300})) {
        _bind_(A);
        _for_(iter, 0, 100) {
            _for_(iter_1, 0, 100) {
                _for_(iter_2, 0, 100) {
                    A[iter + iter_1 + iter_2] = iter + iter_1 + iter_2;
                }
            }
        }
    }
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(out, expected));
}

TEST(GCCore_CPU_ir_simplify, TestSimplifyLoopEliminateWithConstantFolding) {
    builder::ir_builder_t builder;

    _function_(datatypes::void_t, ccc, _arg_("A", datatypes::f32, {10})) {
        _bind_(A);
        _var_(idx, datatypes::index);
        idx = builder::make_cast(datatypes::index, A[0]);
        _var_(idx2, datatypes::index);
        idx2 = idx + 1;
        // Case 1
        _for_(i, idx, idx2) { A[i] = A[i] + 1; }
    }

    auto result = auto_caster_t()(ccc);
    result = constant_folder_t(false)(result);
    result = ir_simplifier_t(true)(result);

    _function_(datatypes::void_t, expected, _arg_("A", datatypes::f32, {10})) {
        _bind_(A);
        _var_(idx, datatypes::index);
        idx = builder::make_cast(datatypes::index, A[0]);
        _var_(idx2, datatypes::index);
        idx2 = idx + UINT64_C(1);
        A[idx] = A[idx] + 1.0f;
    }

    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(result, expected));
}

TEST(GCCore_CPU_ir_simplify, TestSimplifyLoopEliminate) {
    builder::ir_builder_t builder;

    _function_(
            datatypes::void_t, ccc, _arg_("A", datatypes::f32, {10, 20, 32})) {
        _bind_(A);
        for_loop outer;
        // Case 1
        _for_(i, 5, 6) {
            _for_(j, 11, 12) {
                _for_(k, 0, 32, 16) {
                    A[span_t({i, j, k}, 16)] = A[span_t({i, j, k}, 16)]
                            + builder::make_broadcast(1, 16);
                }
            }
        }
        // Case 2
        _for_(i, 5, 6) {
            _for_(j, i, 18) {
                _for_(k, 0, 32, 16) {
                    A[span_t({i, j, k}, 16)] = A[span_t({i, j, k}, 16)]
                            + builder::make_broadcast(1, 16);
                }
            }
        }
        // Case 3
        _for_(i, 5, 6) {
            _for_(j, 11, 12) {
                _for_(k, 16, 32, 16) {
                    A[span_t({i, j, k}, 16)] = A[span_t({i, j, k}, 16)]
                            + builder::make_broadcast(1, 16);
                }
            }
        }
        // Case 4
        _for_(i, 5, 6) {
            _for_(j, 11, 12) {
                _for_(k, 0, 32, 16) {
                    A[span_t({i, j, k}, 16)] = A[span_t({i, j, k}, 16)]
                            + builder::make_broadcast(1, 16);
                }
                _for_(k, 0, 32, 16) {
                    // expected to be eliminated
                }
            }
        }
        // Case 5
        _for_(i, 5, 6) {
            _for_(j, 11, 12) {
                _for_(k, 0, 32, 16) {
                    A[span_t({i, j, k}, 16)] = A[span_t({i, j, k}, 16)]
                            + builder::make_broadcast(1, 16);
                }
                _for_(k, 16, 32, 16) {
                    A[span_t({i, j, k}, 16)] = A[span_t({i, j, k}, 16)]
                            + builder::make_broadcast(1, 16);
                }
            }
        }
        // Case 6
        _named_for_(outer, i, 5, 6) {
            _for_(j, 11, 12) {
                _for_(k, 0, 32, 16) {
                    A[span_t({i, j, k}, 16)] = A[span_t({i, j, k}, 16)]
                            + builder::make_broadcast(1, 16);
                }
            }
        }
        outer->attr()[stmt_attr_key::merge_loop] = true;
    }
    ir_simplifier_t s {false};
    auto out = s(ccc);

    _function_(datatypes::void_t, expected,
            _arg_("A", datatypes::f32, {10, 20, 32})) {
        _bind_(A);
        for_loop new_outer;
        // Case 1
        _for_(k, 0, 32, 16) {
            A[span_t({5, 11, k}, 16)] = A[span_t({5, 11, k}, 16)]
                    + builder::make_broadcast(1, 16);
        }

        // Case 2
        _for_(j, 5, 18) {
            _for_(k, 0, 32, 16) {
                A[span_t({5, j, k}, 16)] = A[span_t({5, j, k}, 16)]
                        + builder::make_broadcast(1, 16);
            }
        }

        // Case 3
        A[span_t({5, 11, 16}, 16)]
                = A[span_t({5, 11, 16}, 16)] + builder::make_broadcast(1, 16);

        // Case 4
        _for_(k, 0, 32, 16) {
            A[span_t({5, 11, k}, 16)] = A[span_t({5, 11, k}, 16)]
                    + builder::make_broadcast(1, 16);
        }

        // Case 5
        _for_(k, 0, 32, 16) {
            A[span_t({5, 11, k}, 16)] = A[span_t({5, 11, k}, 16)]
                    + builder::make_broadcast(1, 16);
        }
        A[span_t({5, 11, 16}, 16)]
                = A[span_t({5, 11, 16}, 16)] + builder::make_broadcast(1, 16);

        // Case 6
        _named_for_(new_outer, k, 0, 32, 16) {
            A[span_t({5, 11, k}, 16)] = A[span_t({5, 11, k}, 16)]
                    + builder::make_broadcast(1, 16);
        }
        new_outer->attr()[stmt_attr_key::merge_loop] = true;
    }
    ir_comparer cmper;

    EXPECT_TRUE(cmper.compare(out, expected));
    // check attr pass down for last gtest case
    EXPECT_TRUE(ccc->body_.checked_as<stmts>()
                        ->seq_.back()
                        .checked_as<for_loop>()
                        ->attr_->get_or_else(stmt_attr_key::merge_loop, false)
            == true);
}

TEST(GCCore_CPU_ir_simplify, TestSimplifyIfAndElseEliminate) {
    builder::ir_builder_t builder;
    _function_(
            datatypes::void_t, ccc, _arg_("A", datatypes::f32, {10, 20, 32})) {
        _bind_(A);
        // Case 1:
        _if_(A[0] == 0) {}
        A[0] = 0;
        // Case 2:
        _if_(A[0] == 0) {}
        _else_ { A[0] = 0; }
        // Case 3:
        _if_(A[0] != 0) { A[0] = 0; }
        _else_ {}
        // Case 4:
        _if_(1 > 0) { A[0] = 1; }
        _else_ { A[0] = 0; }
        // Case 5:
        _if_(1 < 0) { A[0] = 0; }
        _else_ { A[0] = 1; }
        // Case 6:
        _if_(A[0] != 0) {
            _if_(A[0] > 0) {}
            _else_ { A[0] = 1; }
        }
        _else_ {}
        // Case 7:
        _if_(true) {
            _if_(A[0] > 0) {}
            _else_ { A[0] = 1; }
        }
        // Case 8:
        _if_(true) {
            _if_(true) { A[0] = 1; }
            _else_ { A[0] = -1; }
        }
    }
    ir_simplifier_t s {false};
    auto out = s(ccc);

    _function_(datatypes::void_t, expected,
            _arg_("A", datatypes::f32, {10, 20, 32})) {
        _bind_(A);
        // Case 1:
        A[0] = 0;
        // Case 2:
        _if_(!(A[0] == 0)) { A[0] = 0; }
        // Case 3:
        _if_(A[0] != 0) { A[0] = 0; }
        // Case 4:
        A[0] = 1;
        // Case 5:
        A[0] = 1;
        // Case 6:
        _if_(A[0] != 0) {
            _if_(!(A[0] > 0)) { A[0] = 1; }
        }
        // Case 7:
        _if_(!(A[0] > 0)) { A[0] = 1; }
        // Case 8:
        A[0] = 1;
    }
    ir_comparer cmper;
    EXPECT_TRUE(cmper.compare(out, expected));
}
