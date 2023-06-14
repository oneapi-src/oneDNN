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
#include <utility>
#include "gtest/gtest.h"
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/pass/graph_constant_cache.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/cpu/local_tensor_lower.hpp>
#include <compiler/ir/transform/pointer_alias_info.hpp>
#include <runtime/const_cache_wrapper.hpp>
#include <util/any_map.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_local_tensor_lower, TestLocalTensorLowering) {
    builder::ir_builder_t builder;
    local_tensor_lowering_cpu_t pass {128};

    _function_(datatypes::void_t, aaa, _arg_("stream", datatypes::pointer),
            _arg_("globals", datatypes::s8, {0}),
            _arg_("args", datatypes::f32)) {
        _bind_(ctx, globals, args);
        _var_(a, datatypes::s32);
        _tensor_(b1, datatypes::f32, 200);
        _tensor_(dyn1, datatypes::f32, a);
        _tensor_(b2, datatypes::f32, 10);
        b1[0] = args;
        builder.push_scope();
        {
            _tensor_(b3, datatypes::u8, 1000);
            b3[0] = 9;
            _return_();
        }
        builder.emit(builder.pop_scope());
        builder.push_scope();
        {
            _tensor_(b4, datatypes::u8, 1000);
            _tensor_(b5, datatypes::u8, 1000);
            b4[0] = 9;
            builder.push_scope();
            {
                _tensor_(b6, datatypes::u8, 1000);
                b4[0] = 9;
            }
            builder.emit(builder.pop_scope());
        }
        builder.emit(builder.pop_scope());

        _tensor_(b5, datatypes::u8, 1000);

        _for_(i, 0, 100, 1, for_type::PARALLEL) {
            _tensor_(b6, datatypes::u8, 1000);
            b6->attr()["is_thread_buffer"] = true;
        }
    }

    auto mod2 = pass(aaa);

    _function_(datatypes::void_t, expected, _arg_("stream", datatypes::pointer),
            _arg_("globals", datatypes::s8, {0}),
            _arg_("args", datatypes::f32)) {
        _bind_(ctx, globals, args);

        auto set_buffer = [&builder, &ctx](bool is_parallel, expr sz) {
            builder.get_current_scope().body.back().checked_as<define>()->init_
                    = get_cpu_temp_malloc_func(is_parallel)(ctx, std::move(sz));
        };

        auto release_buffer = [&ctx](bool is_parallel, expr sz) {
            _evaluate_call_(
                    get_cpu_temp_free_func(is_parallel), ctx, std::move(sz));
        };

        _var_(a, datatypes::s32);
        _tensor_(b1, datatypes::f32, 200);
        set_buffer(false, 800UL);
        _tensor_(dyn1, datatypes::f32, a);
        set_buffer(
                false, builder::make_cast(datatypes::index, a) * UINT64_C(4));
        _tensor_(b2, datatypes::f32, 10);
        b1[0] = args;
        builder.push_scope();
        {
            _tensor_(b3, datatypes::u8, 1000);
            set_buffer(false, 1000UL);
            b3[0] = 9;
            release_buffer(false, b3);
            _return_();
        }
        builder.emit(builder.pop_scope());
        builder.push_scope();
        {
            _tensor_(b4, datatypes::u8, 1000);
            set_buffer(false, 1000UL);
            _tensor_(b5, datatypes::u8, 1000);
            set_buffer(false, 1000UL);
            b4[0] = 9;
            builder.push_scope();
            {
                _tensor_(b6, datatypes::u8, 1000);
                set_buffer(false, 1000UL);
                b4[0] = 9;
                release_buffer(false, b6);
            }
            builder.emit(builder.pop_scope());

            release_buffer(false, b5);
            release_buffer(false, b4);
        }
        builder.emit(builder.pop_scope());

        _tensor_(b5, datatypes::u8, 1000);
        set_buffer(false, 1000UL);

        _for_(i, 0, 100, 1, for_type::PARALLEL) {
            _tensor_(b6, datatypes::u8, 1000);
            set_buffer(true, 1000UL);
            release_buffer(true, b6);
        }
        release_buffer(false, b5);
        release_buffer(false, dyn1);
        release_buffer(false, b1);
    }
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(mod2, expected, false));
}

TEST(GCCore_CPU_local_tensor_lower, TestSharedConst) {
    builder::ir_builder_t builder;
    local_tensor_lowering_cpu_t pass {128};

    auto dummy_buffer = std::make_shared<int>();
    // compile-time const buffer
    auto base1 = std::make_shared<runtime::const_cache_proxy>(
            dummy_buffer, dummy_buffer.get(), 32, false);
    auto dummy_buffer2 = std::make_shared<int>();
    auto base2 = std::make_shared<runtime::const_cache_proxy>(
            dummy_buffer2, dummy_buffer2.get(), 256, true);

    auto graph_tsr1
            = std::make_shared<cached_const_graph_tensor>(nullptr, 32, nullptr);
    graph_tsr1->buf_base_ = base1;
    graph_tsr1->offset_ = 0;

    auto graph_tsr2 = std::make_shared<cached_const_graph_tensor>(
            nullptr, 256, nullptr);
    graph_tsr2->buf_base_ = base2;
    graph_tsr2->offset_ = 0;

    _function_(datatypes::void_t, aaa, _arg_("stream", datatypes::pointer),
            _arg_("mod_data", datatypes::s8, {0UL}),
            _arg_("args", datatypes::f32)) {
        _bind_(stream, moddata, args);
        _tensor_(__shared_const_handle, datatypes::index, UINT64_C(2));
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(moddata, {0UL});
        args = 1.0f;
        _tensor_(__is_init, datatypes::s32, 1);
        __is_init[0] = 1;

        _tensor_(normal, datatypes::u8, UINT64_C(256));
        _tensor_(base0, datatypes::u8, UINT64_C(32));
        auto &attr1 = builder.get_current_scope()
                              .body.back()
                              .checked_as<define>()
                              ->var_->attr();
        attr1[attr_keys::shared_const] = graph_tsr1;
        attr1[attr_keys::shared_const_base_idx] = size_t(0);
        _tensor_(base1, datatypes::u8, UINT64_C(256));
        auto &attr2 = builder.get_current_scope()
                              .body.back()
                              .checked_as<define>()
                              ->var_->attr();
        attr2[attr_keys::shared_const] = graph_tsr2;
        attr2[attr_keys::shared_const_base_idx] = size_t(1);
        _tensor_(normal2, datatypes::u8, UINT64_C(256));
    }

    auto out = pass(aaa);

    _function_(datatypes::void_t, expected, _arg_("stream", datatypes::pointer),
            _arg_("mod_data", datatypes::s8, {0UL}),
            _arg_("args", datatypes::f32)) {
        _bind_(stream, moddata, args);
        _tensor_(__shared_const_handle, datatypes::index, UINT64_C(2));
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(moddata, {0UL});
        args = 1.0f;
        _tensor_(__is_init, datatypes::s32, 1);
        __is_init[0] = 1;

        _tensor_(normal, datatypes::u8, UINT64_C(256));
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = get_cpu_temp_malloc_func(false)(stream.get(), UINT64_C(256));
        _tensor_(base0, datatypes::u8, UINT64_C(32));
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::make_reinterpret(
                        __shared_const_handle[UINT64_C(0)], datatypes::pointer);
        _tensor_(base1, datatypes::u8, UINT64_C(256));
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = get_acquire_const_cache_func()(stream,
                        __shared_const_handle[UINT64_C(1)], UINT64_C(256),
                        __is_init);
        _tensor_(normal2, datatypes::u8, UINT64_C(256));
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = get_cpu_temp_malloc_func(false)(stream.get(), UINT64_C(256));
        _evaluate_call_(get_cpu_temp_free_func(false), stream, normal2);
        _evaluate_call_(get_release_const_cache_func(), stream,
                __shared_const_handle[UINT64_C(1)], base1);
        _evaluate_call_(get_cpu_temp_free_func(false), stream, normal);
    }

    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(out, expected, false));
    // prevent unregistering the buffer to graph API cache manager
    base2->is_lazy_ = false;
    base1->is_lazy_ = false;
}

TEST(GCCore_CPU_local_tensor_lower, TestAlias) {
    builder::ir_builder_t builder;
    local_tensor_lowering_cpu_t pass {128};

    auto set_buffer = [&builder](const expr &base, uint64_t sz) {
        auto def = builder.get_current_scope().body.back().checked_as<define>();
        def->init_ = builder::tensor_ptr(base, {sz});
    };

    _function_(datatypes::void_t, aaa, _arg_("stream", datatypes::pointer),
            _arg_("globals", datatypes::s8, {0}),
            _arg_("args", datatypes::f32)) {
        _bind_(ctx, globals, args);
        _tensor_(base, datatypes::u8, 200);
        base->attr()["can_be_scheduled"] = true;
        //[0,15]
        _tensor_(b1, datatypes::u8, 16);
        set_buffer(base, 0);
        //[16,79]
        _tensor_(b2, datatypes::f32, 16);
        set_buffer(base, 16);
        //[32,95]
        _tensor_(b3, datatypes::f32, 16);
        set_buffer(base, 32);
        //[80,143], should not alias with b2
        _tensor_(b4, datatypes::index, 8);
        set_buffer(base, 80);

        // indirect base
        _tensor_(base2, datatypes::u8, 64);
        set_buffer(base, 144);
        base2->attr()["can_be_scheduled"] = true;

        //[144,...]
        _tensor_(b5, datatypes::index, 8);
        set_buffer(base2, 0);
        //[150,...]
        _tensor_(b6, datatypes::index, 8);
        set_buffer(base, 150);

        // check for hoist_and_schedule result
        // hoisted base
        _tensor_(hbase, datatypes::u8, 200);
        set_buffer(base, 214);
        hbase->attr()["hoisted"] = true;

        // hoisted scheduled base
        _tensor_(base0, datatypes::u8, 100);
        base0->attr()["can_be_scheduled"] = true;
        set_buffer(hbase, 10); // in reality, should be based on get_gid()*10

        _tensor_(b7, datatypes::index, 2);
        set_buffer(base0, 0);

        _tensor_(b8, datatypes::index, 2);
        set_buffer(base0, 16);

        // should be alias with b8 and b7
        _tensor_(b9, datatypes::u8, 128);
        set_buffer(base, 214);

        _tensor_(b10_hoisted, datatypes::u8, 32);
        set_buffer(base0, 32);
        b10_hoisted->attr()["hoisted"] = true;

        _tensor_(base_b10, datatypes::u8, 16);
        set_buffer(b10_hoisted, 0);
        base_b10->attr()["can_be_scheduled"] = true;

        _tensor_(b11, datatypes::u8, 16);
        set_buffer(base_b10, 0);
    }

    auto mod2 = pass(aaa);
    auto body = mod2->body_.as<stmts>();
    std::vector<std::vector<int64_t>> expected = {
            {1}, // b1
            {2}, // b2
            {2, 3}, // b3
            {3}, // b4
            {4}, // base2
            {4}, // b5
            {4}, // b6
            {5, 10, 11, 12, 14}, // hbase
            // b7 b8 and b11 are not alias
            {6, 10}, // b7
            {7, 11}, // b8
            // b7 b8 and b11 are alias with b9
            {5, 10, 11, 12, 14}, // b9
            {8, 12, 13}, // b10_hoisted
            {9, 13, 14}, // b11
    };
    int cur_tensor = 0;
    std::unordered_set<alias_info::tensor_alias_identity_t *> idset;
    for (auto &s : body->seq_) {
        if (s.isa<define>()) {
            auto def = s.static_as<define>();
            if (auto alias_id = alias_info::get_alias_info(*def->var_)) {
                idset.insert(alias_id);
                auto &cur_expected = expected.at(cur_tensor);
                cur_tensor++;
                std::vector<int64_t> ids;
                for (auto &cli : alias_id->alias_cliques_) {
                    ids.emplace_back(cli->id_);
                }
                EXPECT_EQ(ids, cur_expected);
            }
        }
    }
    ASSERT_TRUE(mod2->attr_);
    auto a_ids = mod2->attr_->get_or_null<
            std::vector<std::shared_ptr<alias_info::tensor_alias_identity_t>>>(
            "alias_sets");
    ASSERT_TRUE(a_ids);
    ASSERT_EQ(a_ids->size(), idset.size());
    for (auto &v : *a_ids) {
        ASSERT_TRUE(idset.count(v.get()));
    }
}
