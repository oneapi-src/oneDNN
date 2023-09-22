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
#include "gtest/gtest.h"
#include <compiler/ir/attr_keys.hpp>
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/graph/pass/graph_constant_cache.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/cpu/closurize.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <runtime/config.hpp>
#include <runtime/const_cache_wrapper.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_CPU_module_globals_resolver_t, TestGlobalTensorExtract) {
    builder::ir_builder_t builder;
    module_globals_resolver_t pass {};
    ir_module_ptr mod = std::make_shared<ir_module_t>(get_default_context());
    _global_tensor_(mod, gv1, datatypes::f32, 30);
    _global_tensor_(mod, gv2, datatypes::s32, 1);
    _global_tensor_(mod, gv3, datatypes::f32, 1);
    _global_var_(mod, scalar_gv, datatypes::f32, 1.0f);

    auto u64_0 = make_expr<constant_node>(UINT64_C(0));
    auto pointer_0 = make_expr<constant_node>(UINT64_C(0), datatypes::pointer);
    auto u8_pointer_0 = make_expr<constant_node>(
            UINT64_C(0), datatypes::s8.get_pointerof());

    int data = 123;
    gv2.static_as<tensor>()->init_value_
            = std::make_shared<static_data_t>(&data, sizeof(data));
    float data2 = 123.0f;
    gv3.static_as<tensor>()->init_value_
            = std::make_shared<static_data_t>(&data2, sizeof(data2));

    auto null_v = make_expr<constant_node>(UINT64_C(0), datatypes::pointer);
    auto brg_func = builtin::get_brgemm_creator_and_call_func(
            builtin::brgemm_mode::stride, scflags_t::brgemm_t::dnnl, false)
                            .second;
    _function_(datatypes::void_t, aaa, _arg_("args", datatypes::f32)) {
        _bind_(args);
        scalar_gv = 2.0f;
        gv1[0] = 1.0f;
        _evaluate_call_(get_parallel_call_with_env_func(false),
                builder::make_func_addr(get_parallel_call_with_env_func(false)),
                u64_0, pointer_0, u8_pointer_0, 0UL, 1UL, 0UL, gv1);
        _evaluate_call_(brg_func, null_v, null_v, null_v, null_v, 100, null_v);
    }
    mod->add_func({aaa});

    auto mod2 = pass(mod);
    auto &globals = *mod2->attr_.get<std::shared_ptr<statics_table_t>>(
            ir_module_t::attr_key_t::MODULE_DATA_BUFFERS);
    EXPECT_EQ(globals.impl_.size(), 4UL);

    ASSERT_EQ(12UL, globals.initialized_size_);
    auto offset = globals.impl_.find("scalar_gv");
    ASSERT_TRUE(offset != globals.impl_.end());
    EXPECT_EQ(offset->second, 0UL);

    offset = globals.impl_.find("gv2");
    ASSERT_TRUE(offset != globals.impl_.end());
    EXPECT_EQ(offset->second, 4UL);

    offset = globals.impl_.find("gv3");
    ASSERT_TRUE(offset != globals.impl_.end());
    EXPECT_EQ(offset->second, 8UL);

    offset = globals.impl_.find("gv1");
    ASSERT_TRUE(offset != globals.impl_.end());
    EXPECT_EQ(offset->second, 64UL);

    EXPECT_EQ(*(int *)globals.get("gv2"), data);
    EXPECT_EQ(*(float *)globals.get("gv3"), data2);

    EXPECT_EQ(globals.data_.size_, 64UL + (1 * 10 * 3) * 4);

    _function_(datatypes::void_t, expected, _arg_("stream", datatypes::pointer),
            _arg_("mod_data", datatypes::s8, {0UL}),
            _arg_("args", datatypes::f32)) {
        _bind_(ctx, mod_data, args);
        _var_(scalar_gv, datatypes::f32);
        _tensor_(gv1, datatypes::f32, 30);
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(mod_data, {64UL});
        scalar_gv = 2.0f;
        gv1[0] = 1.0f;
        _evaluate_call_(get_parallel_call_with_env_func(false),
                builder::make_func_addr(get_parallel_call_with_env_func(false)),
                u64_0, ctx, mod_data, 0UL, 1UL, 0UL, gv1);
        _evaluate_call_(brg_func, null_v, null_v, null_v, null_v, 100, ctx);
    }
    expected->params_[1].checked_as<tensor>()->dims_[0] = 0UL;
    auto func_aaa = mod2->get_func("aaa");
    ASSERT_TRUE(func_aaa);
    ir_comparer cmper {true};
    EXPECT_TRUE(cmper.compare(expected, func_aaa, false));

    globals.save_to_file("globals.bin");
    auto loaded = statics_table_t::load_from_file("globals.bin");
    ASSERT_TRUE(loaded.initialized_size_ == globals.initialized_size_);
    ASSERT_TRUE(loaded.data_.size_ == globals.data_.size_);
    ASSERT_EQ(memcmp(loaded.data_.data_, globals.data_.data_,
                      loaded.initialized_size_),
            0);
}

TEST(GCCore_CPU_module_globals_resolver_t, TestGlobalSharedTensor) {
    auto dummy_buffer = std::make_shared<int>();
    // compile-time const buffer
    auto base1 = std::make_shared<runtime::const_cache_proxy>(
            dummy_buffer, dummy_buffer.get(), 128, false);
    auto dummy_buffer2 = std::make_shared<int>();
    auto base2 = std::make_shared<runtime::const_cache_proxy>(
            dummy_buffer2, dummy_buffer2.get(), 128, true);
    auto dummy_buffer3 = std::make_shared<int>();
    auto base3 = std::make_shared<runtime::const_cache_proxy>(
            dummy_buffer3, dummy_buffer3.get(), 128, true);

    auto graph_tsr1
            = std::make_shared<cached_const_graph_tensor>(nullptr, 64, nullptr);
    graph_tsr1->buf_base_ = base1;
    graph_tsr1->offset_ = 0;
    auto graph_tsr2
            = std::make_shared<cached_const_graph_tensor>(nullptr, 64, nullptr);
    graph_tsr2->buf_base_ = base1;
    graph_tsr2->offset_ = 64;

    auto graph_tsr3
            = std::make_shared<cached_const_graph_tensor>(nullptr, 64, nullptr);
    graph_tsr3->buf_base_ = base2;
    graph_tsr3->offset_ = 0;
    auto graph_tsr4
            = std::make_shared<cached_const_graph_tensor>(nullptr, 64, nullptr);
    graph_tsr4->buf_base_ = base2;
    graph_tsr4->offset_ = 64;

    auto graph_tsr5
            = std::make_shared<cached_const_graph_tensor>(nullptr, 64, nullptr);
    graph_tsr5->buf_base_ = base3;
    graph_tsr5->offset_ = 64;

    builder::ir_builder_t builder;
    module_globals_resolver_t pass {};
    _function_(datatypes::void_t, aaa, _arg_("args", datatypes::f32)) {
        _bind_(args);
        args = 1.0f;
        _tensor_(__is_init, datatypes::s32, 1);
        builder.get_current_scope()
                .body.back()
                ->attr()[attr_keys::is_shared_const_init_stmt]
                = true;
        __is_init[0] = 1;
        builder.get_current_scope()
                .body.back()
                ->attr()[attr_keys::is_shared_const_init_stmt]
                = true;

        _tensor_(A, datatypes::f32, 16);
        A->attr()[attr_keys::shared_const] = graph_tsr1;
        _tensor_(B, datatypes::f32, 16);
        B->attr()[attr_keys::shared_const] = graph_tsr2;
        _tensor_(C, datatypes::f32, 16);
        C->attr()[attr_keys::shared_const] = graph_tsr3;
        _tensor_(D, datatypes::f32, 16);
        D->attr()[attr_keys::shared_const] = graph_tsr4;
        _tensor_(E, datatypes::f32, 16);
        E->attr()[attr_keys::shared_const] = graph_tsr5;
    }
    auto mod = ir_module_t::from_entry_func(get_default_context(), aaa);
    mod->attr_[ir_module_t::attr_key_t::SHARED_CONST_BASES]
            = std::vector<std::shared_ptr<runtime::const_cache_proxy>> {
                    base1, base2, base3};

    auto out_mod = pass(mod);
    _function_(datatypes::void_t, expected, _arg_("stream", datatypes::pointer),
            _arg_("mod_data", datatypes::s8, {0UL}),
            _arg_("args", datatypes::f32)) {
        _bind_(stream, moddata, args);
        _tensor_(handles, datatypes::index, UINT64_C(3));
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(moddata, {0UL});
        args = 1.0f;
        _tensor_(__is_init, datatypes::s32, 1);
        __is_init[0] = 1;

        _tensor_(base0, datatypes::u8, UINT64_C(128));
        _tensor_(base1, datatypes::u8, UINT64_C(128));
        _tensor_(base2, datatypes::u8, UINT64_C(128));

        _tensor_(A, datatypes::f32, 16);
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(base0, {0UL});
        _tensor_(B, datatypes::f32, 16);
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(base0, {64UL});
        _tensor_(C, datatypes::f32, 16);
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(base1, {0UL});
        _tensor_(D, datatypes::f32, 16);
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(base1, {64UL});
        _tensor_(E, datatypes::f32, 16);
        builder.get_current_scope().body.back().checked_as<define>()->init_
                = builder::tensor_ptr(base2, {64UL});
    }

    ir_comparer cmper {true};
    ASSERT_TRUE(cmper.compare(out_mod->get_entry_func(), expected, true));
    auto &globals = *out_mod->attr_.get<std::shared_ptr<statics_table_t>>(
            ir_module_t::attr_key_t::MODULE_DATA_BUFFERS);
    EXPECT_EQ(globals.impl_.size(), 1UL);
    auto handles_buf = (void **)globals.get_or_null("__shared_const_handle");
    ASSERT_TRUE(handles_buf);
    // const buffer, directly use buffer ptr
    EXPECT_EQ(handles_buf[0], dummy_buffer.get());
    // lazy const buffer, use buffer proxy
    EXPECT_EQ(handles_buf[1], base2.get());
    // lazy const buffer, use buffer proxy
    EXPECT_EQ(handles_buf[2], base3.get());

    auto &body = out_mod->get_entry_func()->body_.checked_as<stmts>()->seq_;
    auto is_ok = [&body](int index, size_t expected,
                         cached_const_graph_tensor *shared) {
        return body.at(index)
                .cast<define>()
                .filter([expected, shared](const define &v) {
                    auto pshared = any_map_t::fetch_or_null<
                            std::shared_ptr<cached_const_graph_tensor>>(
                            v->var_->attr_.get(), attr_keys::shared_const);
                    if (!pshared) { return false; }
                    return any_map_t::fetch_or_else(v->var_->attr_.get(),
                                   attr_keys::shared_const_base_idx,
                                   size_t(10000))
                            == expected
                            && pshared->get() == shared;
                })
                .has_value();
    };
    // check shared_const_base_idx attr
    EXPECT_TRUE(is_ok(4, 0, graph_tsr1.get()));
    EXPECT_TRUE(is_ok(5, 1, graph_tsr3.get()));
    EXPECT_TRUE(is_ok(6, 2, graph_tsr5.get()));
    // prevent unregistering the buffer to graph API cache manager
    base2->is_lazy_ = false;
    base3->is_lazy_ = false;
}
