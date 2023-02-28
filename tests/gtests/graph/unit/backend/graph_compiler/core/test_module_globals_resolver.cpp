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
#include <compiler/ir/builtin.hpp>
#include <compiler/ir/easy_build.hpp>
#include <compiler/ir/ir_comparer.hpp>
#include <compiler/ir/transform/cpu/closurize.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <runtime/config.hpp>

using namespace dnnl::impl::graph::gc;

TEST(GCCore_module_globals_resolver_t, TestGlobalTensorExtract) {
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
