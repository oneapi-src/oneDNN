/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#include <fstream>
#include <utility>
#include <vector>
#include <unordered_map>
#include <util/utils.hpp>

#include <compiler/jit/symbol_resolver.hpp>
#include <compiler/jit/xbyak/xbyak_jit_engine.hpp>
#include <compiler/jit/xbyak/xbyak_jit_module.hpp>
#include <compiler/jit/xbyak/xbyak_lowering_viewer.hpp>

#include <compiler/codegen/precodegen_passes.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/dessa_transform.hpp>
#include <compiler/ir/transform/loop_invariant_code_motion.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <compiler/ir/transform/value_numbering.hpp>

#include <compiler/jit/xbyak/ir/pass/ir_data_initializer.hpp>
#include <compiler/jit/xbyak/ir/pass/ir_indexer.hpp>
#include <compiler/jit/xbyak/ir/pass/live_interval.hpp>
#include <compiler/jit/xbyak/ir/transform/call_transform.hpp>
#include <compiler/jit/xbyak/ir/transform/constant_optimizer.hpp>
#include <compiler/jit/xbyak/ir/transform/constant_propagation.hpp>
#include <compiler/jit/xbyak/ir/transform/module_var_resolver.hpp>
#include <compiler/jit/xbyak/ir/transform/register_allocation.hpp>
#include <compiler/jit/xbyak/ir/transform/x86_intrinsics_lowering.hpp>
#include <compiler/jit/xbyak/ir/xbyak_printer.hpp>

namespace sc {
namespace sc_xbyak {

sequential_module_pass_t get_xbyak_precodegen_passes(
        const context_ptr &ctx, const x86_64::target_profile_t &profile) {
    std::vector<module_pass_ptr> ret;

    ret.emplace_back(utils::make_unique<constant_folder_t>());
    ret.emplace_back(module_function_pass_t::make<module_var_resolver_t>());
    ret.emplace_back(module_function_pass_t::make<constant_optimizer_t>());
    ret.emplace_back(module_function_pass_t::make<call_transform_t>(profile));

    ret.emplace_back(module_function_pass_t::make<ssa_transform_t>());
    ret.emplace_back(module_function_pass_t::make<value_numbering_t>());
    ret.emplace_back(
            module_function_pass_t::make<loop_invariant_code_motion_t>());
    ret.emplace_back(module_function_pass_t::make<dessa_transform_t>());

    ret.emplace_back(module_function_pass_t::make<constant_propagation_t>());
    ret.emplace_back(module_function_pass_t::make<x86_intrinsics_lowering_t>(
            ctx->machine_));

    ret.emplace_back(module_function_pass_t::make<ir_data_initializer_t>());
    ret.emplace_back(module_function_pass_t::make<ir_indexer_t>());
    ret.emplace_back(module_function_pass_t::make<live_interval_t>());
    ret.emplace_back(
            module_function_pass_t::make<register_allocation_t>(profile));

    return sequential_module_pass_t(std::move(ret));
}

void *xbyak_external_symbol_resolve(const std::string &name) {
    static std::unordered_map<std::string, void *> table = {
            {"memset", (void *)memset},
    };
    // Find function in local table first, then external table
    auto itr = table.find(name);
    if (itr != table.end()) { return itr->second; }
    return default_external_symbol_resolve(name);
}

xbyak_jit_engine::xbyak_jit_engine(context_ptr context)
    : jit_engine_t(std::move(context))
    , external_symbol_resolver_(xbyak_external_symbol_resolve) {}

xbyak_jit_engine::~xbyak_jit_engine() = default;

std::shared_ptr<jit_module> xbyak_jit_engine::make_jit_module(
        const_ir_module_ptr ir_mod, bool generate_wrapper) {
    assert(ir_mod);
    COMPILE_ASSERT(generate_wrapper, "Wrapper is required by xbyak backend.");
    COMPILE_ASSERT(ir_mod->ctx_->flags_.ssa_passes_ == false,
            "SC_SSA_PASSES is redundant for xbyak backend.");

    //========================================================================
    //  Default passes
    //========================================================================
    auto default_passes
            = get_default_precodegen_passes(ir_mod->ctx_, generate_wrapper);
    auto ir_mod1 = run_precodegen_passes(default_passes, ir_mod);
    assert(ir_mod1);

    //========================================================================
    // Xbyak passes
    //========================================================================
    x86_64::target_profile_t target_profile
            = x86_64::get_target_profile(ir_mod1->ctx_->machine_);

    auto xbyak_passes
            = get_xbyak_precodegen_passes(ir_mod1->ctx_, target_profile);
    auto ir_mod2 = xbyak_passes(ir_mod1);
    assert(ir_mod2);

    if (ir_mod2->ctx_->flags_.xbyak_jit_asm_listing_) {
        xbyak_printer_t printer(ir_mod2, target_profile);
        std::ofstream f("xbyak_ir.txt");
        f << printer.get_stream().rdbuf();
    }

    //========================================================================
    // Xbyak code-gen
    //========================================================================
    xbyak_lowering_viewer xlv {*this, *ir_mod2, target_profile};
    auto jit_output_ = xlv.get_jit_output();

    //========================================================================
    // Execute __sc_init__
    //========================================================================
    auto &attr_table = *ir_mod2->attr_.get<std::shared_ptr<statics_table_t>>(
            ir_module_t::attr_key_t::MODULE_DATA_BUFFERS);
    typedef void (*init_func_t)(void *ctx, void *mod);
    auto init_func = reinterpret_cast<init_func_t>(
            jit_output_->get_func_address("__sc_init__"));
    if (init_func) { init_func(nullptr, attr_table.data_.data_); }

    //========================================================================
    // Make xbyak_jit_module
    //========================================================================
    bool use_managed_tp = ir_mod2->attr_.get<bool>(
            ir_module_t::attr_key_t::MANAGED_THREAD_POOL);
    auto ret = std::shared_ptr<xbyak_jit_module>(new xbyak_jit_module(
            std::move(jit_output_), std::move(attr_table), use_managed_tp));
    ret->update_runtime_op_tables(ir_mod);
    return ret;
}

} // namespace sc_xbyak
} // namespace sc
