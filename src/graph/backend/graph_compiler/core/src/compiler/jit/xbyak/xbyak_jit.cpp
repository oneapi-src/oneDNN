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

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_map>

#include <compiler/jit/symbol_resolver.hpp>
#include <compiler/jit/xbyak/backend/xbyak_jit_generator.hpp>
#include <compiler/jit/xbyak/backend/xbyak_lowering_viewer.hpp>
#include <compiler/jit/xbyak/ir/xbyak_printer.hpp>
#include <runtime/config.hpp>
#include <util/utils.hpp>

#include <compiler/codegen/precodegen_passes.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/dessa_transform.hpp>
#include <compiler/ir/transform/loop_function_motion.hpp>
#include <compiler/ir/transform/loop_invariant_code_motion.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <compiler/ir/transform/value_numbering.hpp>

#include <compiler/jit/xbyak/ir/pass/ir_data_initializer.hpp>
#include <compiler/jit/xbyak/ir/pass/ir_indexer.hpp>
#include <compiler/jit/xbyak/ir/pass/live_interval.hpp>
#include <compiler/jit/xbyak/ir/transform/avx2_legalizer.hpp>
#include <compiler/jit/xbyak/ir/transform/call_transform.hpp>
#include <compiler/jit/xbyak/ir/transform/constant_optimizer.hpp>
#include <compiler/jit/xbyak/ir/transform/indexing_transform.hpp>
#include <compiler/jit/xbyak/ir/transform/intrinsics_combine.hpp>
#include <compiler/jit/xbyak/ir/transform/low_level_legalizer.hpp>
#include <compiler/jit/xbyak/ir/transform/module_var_resolver.hpp>
#include <compiler/jit/xbyak/ir/transform/register_allocation.hpp>
#include <compiler/jit/xbyak/ir/transform/x86_intrinsics_lowering.hpp>

#include "xbyak_jit.hpp"

SC_MODULE(xbyakjit.xbyak_jit)

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace xbyak {

sequential_module_pass_t get_xbyak_precodegen_passes(
        const context_ptr &ctx, const x86_64::target_profile_t &profile) {
    std::vector<module_pass_ptr> ret;

    ret.emplace_back(module_function_pass_t::make<module_var_resolver_t>());
    ret.emplace_back(module_function_pass_t::make<indexing_transform_t>());
    ret.emplace_back(utils::make_unique<constant_folder_t>(false));
    ret.emplace_back(utils::make_unique<auto_caster_t>());
    ret.emplace_back(
            module_function_pass_t::make<low_level_legalizer_t>(ctx->machine_));
    ret.emplace_back(module_function_pass_t::make<constant_optimizer_t>());
    ret.emplace_back(
            module_function_pass_t::make<avx2_legalizer_t>(ctx->machine_));
    ret.emplace_back(
            module_function_pass_t::make<simple_loop_function_motion_t>());
    ret.emplace_back(module_function_pass_t::make<ir_simplifier_t>(false));

    ret.emplace_back(module_function_pass_t::make<ssa_transform_t>());
    ret.emplace_back(module_function_pass_t::make<value_numbering_t>());
    ret.emplace_back(
            module_function_pass_t::make<loop_invariant_code_motion_t>());
    ret.emplace_back(module_function_pass_t::make<intrinsics_combine_t>());
    ret.emplace_back(module_function_pass_t::make<value_numbering_t>());
    ret.emplace_back(module_function_pass_t::make<dessa_transform_t>());
    ret.emplace_back(utils::make_unique<constant_folder_t>(false));

    ret.emplace_back(module_function_pass_t::make<call_transform_t>(profile));
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

} // namespace xbyak

// ================== //
//     xbyak_jit      //
// ================== //
using namespace gc::xbyak;

xbyak_jit::xbyak_jit(context_ptr context)
    : jit_engine_t(std::move(context))
    , external_symbol_resolver_(xbyak_external_symbol_resolve) {}

xbyak_jit::~xbyak_jit() = default;

std::shared_ptr<jit_module> xbyak_jit::make_jit_module(
        const_ir_module_ptr ir_mod, bool generate_wrapper) {
    COMPILE_ASSERT(ir_mod->ctx_->machine_.cpu_flags_.fAVX2,
            "Builtin codegen currently only support AVX2 and AVX512");
    assert(ir_mod);
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

    if (utils::compiler_configs_t::get().xbyak_jit_asm_listing_) {
        std::ofstream f("xbyak_ir.txt");
        xbyak_printer_t printer(f, ir_mod2, target_profile);
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
    auto ret = std::shared_ptr<xbyak_jit_module_code>(
            new xbyak_jit_module_code(std::move(jit_output_), use_managed_tp));
    ret->postprocess(ir_mod2, attr_table);
    return std::make_shared<jit_module>(std::move(attr_table), ret);
}

// ================== //
//  xbyak_jit_module  //
// ================== //

xbyak_jit_module_code::xbyak_jit_module_code(
        std::shared_ptr<xbyak_jit_generator> jit_output,
        bool managed_thread_pool)
    : jit_module_code(managed_thread_pool)
    , jit_output_(std::move(jit_output)) {}

void *xbyak_jit_module_code::get_address_of_symbol(const std::string &name) {
    return jit_output_->get_func_address(name);
}

void *xbyak_jit_module_code::get_function(
        const std::string &name, void *&wrapper) {
    void *fun = jit_output_->get_func_address(name);
    wrapper = jit_output_->get_func_address(name + "_0wrapper");
    return fun;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
