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
#include <compiler/ir/transform/index2var.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <util/any_map.hpp>

#include <iostream>
#include "gtest/gtest.h"

using namespace dnnl::impl::graph::gc;
using namespace dnnl::impl::graph::gc::builder;
TEST(GCCore_CPU_index2var_cpp, TestIndex2Var) {
    builder::ir_builder_t builder;
    _decl_func_(datatypes::s32, functest);
    _function_(datatypes::s32, aaa, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        _tensor_(tmp, datatypes::f32, 100, 200);
        tmp[{0, 0}] = tmp[{0, 0}] + 1;
        _for_(i, 0, len) {
            A[{i, 10}] = A[{i, 10}] + A[{i, 10}];
            B[{i, 10}] = B[{i, 10}] + A[{i, 10}];
            i = i + 1; // changed index dependency, should evict

            A[{i, 10}] = A[{i, 10}] + B[{i, 10}];

            // index changed, should evict above
            A[{i, i}] = A[{i, 10}] + B[{i, i}];

            // read the value cache of parent scope, should evict (TODO(xxx): we
            // can avoid evict that)
            A[{i, i}] = tmp[{0, 0}];
        }

        _if_(true) { tmp[{0, 0}] = 1.2f; }
        _else_ {
            tmp[{0, 0}] = 1.2f;
            // touched the cached tensor outside of the scope, should evict
            tmp[{0, 1}] = 1.2f;
        }
        _for_(i, 0, len) {
            A[{i, 10}] = 3;
            // should evict above, use it in function
            builtin::mem_zero(A, 0, datatypes::f32);
            A[{i, 10}] = 4;
            // should evict above, tensor_ptr
            builtin::mem_zero(
                    builder::tensor_ptr(A, {0, 0}), 0, datatypes::f32);
            A[{i, 10}] = 5;
        }
        _for_(i, 0, len) {
            A[{i, 10}] = 3;
            // should evict above, indexing as index.
            A[{B[{i, 0}], 0}] = 3;
            A[{i, 10}] = 4;
            // should evict above, func call as index
            A[{functest(), 0}] = 3;
            A[{i, 10}] = 5;
        }

        _for_(i, 0, len) {
            A[span_t({i, 10}, 16)]
                    = A[span_t({i, 10}, 16)] + A[span_t({i, 10}, 16)];
            B[span_t({i, 10}, 16)] = A[span_t({i, 10}, 16)];
        }

        _return_(12);
    }

    _function_(datatypes::s32, expected, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        _tensor_(tmp, datatypes::f32, 100, 200);
        _var_init_(cached0, datatypes::f32, (tmp[{0, 0}]));
        {
            builder.push_scope();
            cached0 = cached0 + 1;
            tmp[{0, 0}] = cached0;
            builder.emit(builder.pop_scope());
        }
        _for_(i, 0, len) {
            _var_init_(cached1, datatypes::f32, (A[{i, 10}]));
            {
                builder.push_scope();
                cached1 = cached1 + cached1;
                A[{i, 10}] = cached1;
                builder.emit(builder.pop_scope());
            }
            _var_init_(cached2, datatypes::f32, (B[{i, 10}]));
            {
                builder.push_scope();
                cached2 = cached2 + cached1;
                B[{i, 10}] = cached2;
                builder.emit(builder.pop_scope());
            }
            i = i + 1; // changed index dependency, should evict

            _var_init_(cached3, datatypes::f32, (A[{i, 10}]));
            _var_init_(cached4, datatypes::f32, (B[{i, 10}]));
            {
                builder.push_scope();
                cached3 = cached3 + cached4;
                A[{i, 10}] = cached3;
                builder.emit(builder.pop_scope());
            }

            _var_init_(cached5, datatypes::f32, (B[{i, i}]));
            _var_(cached6, datatypes::f32);
            {
                builder.push_scope();
                cached6 = cached3 + cached5;
                builder.emit(builder.pop_scope());
            }
            {
                builder.push_scope();
                cached6 = cached0;
                A[{i, i}] = cached6;
                builder.emit(builder.pop_scope());
            }
        }

        _if_(true) {
            _var_(cached7, datatypes::f32);
            {
                builder.push_scope();
                cached7 = 1.2f;
                tmp[{0, 0}] = cached7;
                builder.emit(builder.pop_scope());
            }
        }
        _else_ {
            _var_(cached7, datatypes::f32);
            {
                builder.push_scope();
                cached7 = 1.2f;
                tmp[{0, 0}] = cached7;
                builder.emit(builder.pop_scope());
            }
            _var_(cached9, datatypes::f32);
            {
                builder.push_scope();
                cached9 = 1.2f;
                tmp[{0, 1}] = cached9;
                builder.emit(builder.pop_scope());
            }
        }
        _for_(i, 0, len) {
            _var_(cached10, datatypes::f32);
            {
                builder.push_scope();
                cached10 = 3;
                A[{i, 10}] = cached10;
                builder.emit(builder.pop_scope());
            }
            // should evict above, use it in function
            builtin::mem_zero(A, 0, datatypes::f32);
            _var_(cached11, datatypes::f32);
            {
                builder.push_scope();
                cached11 = 4;
                A[{i, 10}] = cached11;
                builder.emit(builder.pop_scope());
            }
            // should evict above, tensor_ptr
            builtin::mem_zero(
                    builder::tensor_ptr(A, {0, 0}), 0, datatypes::f32);
            _var_(cached12, datatypes::f32);
            {
                builder.push_scope();
                cached12 = 5;
                A[{i, 10}] = cached12;
                builder.emit(builder.pop_scope());
            }
        }
        _for_(i, 0, len) {
            _var_(cached10, datatypes::f32);
            {
                builder.push_scope();
                cached10 = 3;
                A[{i, 10}] = cached10;
                builder.emit(builder.pop_scope());
            }

            _var_init_(cached14, datatypes::f32, (B[{i, 0}]));
            _var_(cached15, datatypes::f32);
            {
                builder.push_scope();
                cached15 = 3;
                A[{cached14, 0}] = cached15;
                builder.emit(builder.pop_scope());
            }
            _var_(cached11, datatypes::f32);
            {
                builder.push_scope();
                cached11 = 4;
                A[{i, 10}] = cached11;
                builder.emit(builder.pop_scope());
            }
            A[{functest(), 0}] = 3;
            _var_(cached12, datatypes::f32);
            {
                builder.push_scope();
                cached12 = 5;
                A[{i, 10}] = cached12;
                builder.emit(builder.pop_scope());
            }
        }

        _for_(i, 0, len) {
            _var_init_(cached18, sc_data_type_t::f32(16),
                    (A[span_t({i, 10}, 16)]));
            {
                builder.push_scope();
                cached18 = cached18 + cached18;
                A[span_t({i, 10}, 16)] = cached18;
                builder.emit(builder.pop_scope());
            }
            _var_(cached19, sc_data_type_t::f32(16));
            {
                builder.push_scope();
                cached19 = cached18;
                B[span_t({i, 10}, 16)] = cached19;
                builder.emit(builder.pop_scope());
            }
        }

        _return_(12);
    }
    index2var_t f;
    ir_comparer cmp(true);
    EXPECT_TRUE(cmp.compare(f(aaa), expected, false));
}

// a test case which have pointer alias
TEST(GCCore_CPU_index2var_cpp, TestIndex2VarAlias) {
    builder::ir_builder_t builder;
    _function_(datatypes::s32, aaa, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        {
            auto clique = std::make_shared<alias_info::alias_set_t>();
            alias_info::get_or_create_alias_info(*A.get())->add_to_clique(
                    clique);
            alias_info::get_or_create_alias_info(*B.get())->add_to_clique(
                    clique);
        }
        A[{0, 1}] = 1.0f;
        A[{0, 1}] = A[{0, 1}] + 1.0f;
        _var_(a, datatypes::f32);
        a = B[{0, 1}];
        A[{0, 1}] = A[{0, 1}] + 1.0f;
        a = B[{0, 1}] + 1;
        B[{0, 1}] = B[{0, 1}] + 1;
        _return_(A[{0, 1}] + B[{0, 1}]);
    }

    index2var_t f;
    auto out = f(aaa);

    _function_(datatypes::s32, expected, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        _var_(cached0, sc_data_type_t::f32());
        {
            builder.push_scope();
            cached0 = 1.0f;
            builder.emit(builder.pop_scope());
        }
        {
            builder.push_scope();
            cached0 = cached0 + 1.0f;
            builder.emit(builder.pop_scope());
        }
        _var_(a, datatypes::f32);
        _var_init_(cached1, sc_data_type_t::f32(), (B[{0, 1}]));
        a = cached1;
        {
            // original code: A[{0, 1}] = A[{0, 1}] + 1.0f;
            // B is read, OK to keep cache of A
            builder.push_scope();
            cached0 = cached0 + 1.0f;
            A[{0, 1}] = cached0;
            builder.emit(builder.pop_scope());
        }
        // A is written, need to evict and re-load B
        _var_init_(cached2, sc_data_type_t::f32(), (B[{0, 1}]));
        a = cached2 + 1;
        {
            // original code: B[{0, 1}] = B[{0, 1}] + 1.0f;
            builder.push_scope();
            cached2 = cached2 + 1;
            B[{0, 1}] = cached2;
            builder.emit(builder.pop_scope());
        }
        // B is written, need to reload A
        _var_init_(cached3, sc_data_type_t::f32(), (A[{0, 1}]));
        _return_(cached3 + cached2);
    }

    ir_comparer cmp(true);
    EXPECT_TRUE(cmp.compare(out, expected, false));
}

TEST(GCCore_CPU_index2var_cpp, TestIndex2VarLoopLift) {
    builder::ir_builder_t builder;
    _function_(datatypes::s32, aaa, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        // can lift
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) {
                    _var_(i2, datatypes::index);
                    i2 = i + 1;
                    A[{i2, 0}] = 0;
                }
            }
        }
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) {
                    _var_(mask, datatypes::u8);
                    mask = make_constant({UINT64_C(1)}, datatypes::u8);
                    _var_(tmp, sc_data_type_t::f32(8));
                    tmp = B[span_t({i, 0}, 8, mask)];
                }
            }
        }
        // not nested for
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                len = len + 1;
                _for_(k, 0, 100, 1) { A[{i, 0}] = 0; }
            }
        }
        // flush before
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) {
                    A[{0, 0}] = 0;
                    A[{i, 0}] = 0;
                }
            }
        }
        // complex
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) { A[{len, 0}] = 0; }
            }
        }
        // tensor defined in inner loop
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) {
                    _tensor_(C, datatypes::s32, {10, 20});
                    C[{i, 0}] = 0;
                }
            }
        }
        _return_(0);
    }

    _function_(datatypes::s32, expected, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        // can lift
        _for_(i, 0, 100, 1) {
            _var_(i2, datatypes::index);
            i2 = i + 1;
            _var_(cache0, datatypes::f32);
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) {
                    builder.push_scope();
                    cache0 = 0;
                    builder.emit(builder.pop_scope());
                }
            }
            A[{i2, 0}] = cache0;
        }
        _for_(i, 0, 100, 1) {
            _var_(mask, datatypes::u8);
            mask = make_constant({UINT64_C(1)}, datatypes::u8);
            _var_init_(
                    cache1, sc_data_type_t::f32(8), B[span_t({i, 0}, 8, mask)]);
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) {
                    _var_(tmp, sc_data_type_t::f32(8));
                    tmp = cache1;
                }
            }
        }
        // not nested for
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                len = len + 1;
                _var_(cache0, datatypes::f32);
                _for_(k, 0, 100, 1) {
                    builder.push_scope();
                    cache0 = 0;
                    builder.emit(builder.pop_scope());
                }
                A[{i, 0}] = cache0;
            }
        }
        // flush before
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) {
                    _var_(cache0, datatypes::f32);
                    builder.push_scope();
                    cache0 = 0;
                    A[{0, 0}] = cache0;
                    builder.emit(builder.pop_scope());

                    _var_(cache1, datatypes::f32);
                    builder.push_scope();
                    cache1 = 0;
                    A[{i, 0}] = cache1;
                    builder.emit(builder.pop_scope());
                }
            }
        }
        // complex
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) {
                    _var_(cache0, datatypes::f32);
                    builder.push_scope();
                    cache0 = 0;
                    A[{len, 0}] = cache0;
                    builder.emit(builder.pop_scope());
                }
            }
        }
        // tensor defined in inner loop
        _for_(i, 0, 100, 1) {
            _for_(j, 0, 100, 1) {
                _for_(k, 0, 100, 1) {
                    _tensor_(C, datatypes::s32, {10, 20});
                    _var_(cache5, datatypes::s32);
                    builder.push_scope();
                    cache5 = 0;
                    C[{i, 0}] = cache5;
                    builder.emit(builder.pop_scope());
                }
            }
        }
        _return_(0);
    }

    index2var_t f;
    auto out = f(aaa);
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}

TEST(GCCore_CPU_index2var_cpp, TestIndex2VarMask) {
    builder::ir_builder_t builder;
    _function_(datatypes::f32, aaa, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        expr mask = builder::make_constant({UINT64_C(16)}, datatypes::u16);
        A[span_t {{len}, 16, mask}] = B[span_t {{len}, 16}];
        _var_(a, sc_data_type_t::f32(16));
        _var_(b, sc_data_type_t::f32(16));
        b = A[span_t {{len}, 16}];
        a = A[span_t {{len}, 16, mask}];
        _return_(builder::make_reduce_add(a));
    }

    _function_(datatypes::f32, expected, _arg_("A", datatypes::f32, {123, 321}),
            _arg_("B", datatypes::f32, {111, 222}),
            _arg_("len", datatypes::s32)) {
        _bind_(A, B, len);
        expr mask = builder::make_constant({UINT64_C(16)}, datatypes::u16);
        _var_init_(cached0, sc_data_type_t::f32(16), (B[span_t {{len}, 16}]));
        _var_(cached1, sc_data_type_t::f32(16));
        {
            // original code: A[{0, 1}] = A[{0, 1}] + 1.0f;
            // B is read, OK to keep cache of A
            builder.push_scope();
            // mask > 0 is true, index2var will use cached0.
            cached1 = cached0;
            A[span_t {{len}, 16, mask}] = cached1;
            builder.emit(builder.pop_scope());
        }
        _var_(a, sc_data_type_t::f32(16));
        _var_(b, sc_data_type_t::f32(16));
        _var_init_(cached2, sc_data_type_t::f32(16), (A[span_t {{len}, 16}]));
        b = cached2;
        _var_init_(cached3, sc_data_type_t::f32(16),
                (A[span_t {{len}, 16, mask}]));
        a = cached3;
        _return_(builder::make_reduce_add(a));
    }

    index2var_t f;
    auto out = f(aaa);
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
}
