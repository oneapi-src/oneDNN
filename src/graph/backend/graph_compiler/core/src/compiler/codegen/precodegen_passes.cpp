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
#include <memory>
#include <utility>
#include <vector>

#include "precodegen_passes.hpp"
#include <compiler/ir/pass/validator.hpp>
#include <compiler/ir/pass_manager.hpp>
#include <compiler/ir/sequential_function_pass.hpp>
#include <compiler/ir/transform/auto_cast.hpp>
#include <compiler/ir/transform/bf16_fp16_legalize.hpp>
#include <compiler/ir/transform/buffer_reschedule_tensor_hoist.hpp>
#include <compiler/ir/transform/buffer_schedule.hpp>
#include <compiler/ir/transform/concat_memory_planning.hpp>
#include <compiler/ir/transform/constant_fold.hpp>
#include <compiler/ir/transform/cpu/closurize.hpp>
#include <compiler/ir/transform/cpu/kernel_lower.hpp>
#include <compiler/ir/transform/cpu/local_tensor_lower.hpp>
#include <compiler/ir/transform/cpu/target_specific_lower.hpp>
#include <compiler/ir/transform/dead_func_eliminate.hpp>
#include <compiler/ir/transform/dead_write_eliminate.hpp>
#include <compiler/ir/transform/dessa_transform.hpp>
#include <compiler/ir/transform/dyn_tsr_transform.hpp>
#include <compiler/ir/transform/func_inline.hpp>
#include <compiler/ir/transform/index2var.hpp>
#include <compiler/ir/transform/index_flatten.hpp>
#include <compiler/ir/transform/insert_trace.hpp>
#include <compiler/ir/transform/interface_generalize.hpp>
#include <compiler/ir/transform/loop_invariant_code_motion.hpp>
#include <compiler/ir/transform/loop_merge.hpp>
#include <compiler/ir/transform/loop_unroll.hpp>
#include <compiler/ir/transform/module_globals_resolve.hpp>
#include <compiler/ir/transform/nested_parallel_flatten.hpp>
#include <compiler/ir/transform/parallel_merge.hpp>
#include <compiler/ir/transform/parallel_workload_dispatch.hpp>
#include <compiler/ir/transform/simple_licm.hpp>
#include <compiler/ir/transform/simplify.hpp>
#include <compiler/ir/transform/ssa_transform.hpp>
#include <compiler/ir/transform/tensor2var.hpp>
#include <compiler/ir/transform/tensor_init.hpp>
#include <compiler/ir/transform/tensor_inplace.hpp>
#include <compiler/ir/transform/tensor_shrink.hpp>
#include <compiler/ir/transform/value_numbering.hpp>
#include <compiler/ir/util_module_passes.hpp>
#include <runtime/config.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

sequential_module_pass_t get_default_precodegen_passes(
        const context_ptr &ctx, bool gen_wrapper) {
    std::vector<module_pass_ptr> ret;
    ret.reserve(64);
    ret.emplace_back(utils::make_unique<dyn_tensor_transformer_t>());
    if (gen_wrapper) {
        ret.emplace_back(utils::make_unique<interface_generalizer_t>());
    }
    ret.emplace_back(utils::make_unique<tensor_shrinker_t>());
    if (ctx->flags_.concat_optimization_) {
        ret.emplace_back(utils::make_unique<concat_memory_planning_t>());
    }
    ret.emplace_back(utils::make_unique<index_flattener_t>());
    ret.emplace_back(utils::make_unique<auto_caster_t>());
    ret.emplace_back(module_function_pass_t::make<bf16_fp16_legalizer_t>(ctx));
    ret.emplace_back(utils::make_unique<validator_t>());
    if (ctx->flags_.trace_) {
        ret.emplace_back(utils::make_unique<trace_inserter_t>());
    }

    ret.emplace_back(utils::make_unique<func_inliner_t>());
    ret.emplace_back(utils::make_unique<constant_folder_t>());
    ret.emplace_back(module_function_pass_t::make<ir_simplifier_t>(true, true));
    ret.emplace_back(module_function_pass_t::make<loop_merger_t>());
    ret.emplace_back(module_function_pass_t::make<ir_simplifier_t>(true));

    if (ctx->flags_.buffer_schedule_ > 0 && ctx->flags_.tensor_inplace_) {
        ret.emplace_back(utils::make_unique<tensor_inplace_t>(ctx));
    }
    ret.emplace_back(module_function_pass_t::make<tensor_init_t>(ctx));

    ret.emplace_back(
            module_function_pass_t::make<parallel_workload_dispatcher_t>());
    ret.emplace_back(module_function_pass_t::make<
            simple_loop_invariant_code_motion_t>());
    ret.emplace_back(utils::make_unique<constant_folder_t>(false));
    if (ctx->flags_.index2var_) {
        ret.emplace_back(module_function_pass_t::make<index2var_t>());
    }
    if (ctx->flags_.dead_write_elimination_) {
        ret.emplace_back(
                module_function_pass_t::make<dead_write_eliminator_t>());
    }
    ret.emplace_back(module_function_pass_t::make<ir_simplifier_t>(true));

    ret.emplace_back(
            module_function_pass_t::make<buffer_rescheduling_tensor_hoisting_t>(
                    ctx, true, ctx->flags_.tensor_inplace_));
    ret.emplace_back(
            module_function_pass_t::make<nested_parallel_flattener_t>());
    ret.emplace_back(utils::make_unique<constant_folder_t>(false));
    ret.emplace_back(module_function_pass_t::make<ir_simplifier_t>(true));

    ret.emplace_back(utils::make_unique<parallel_merge_t>());
    ret.emplace_back(utils::make_unique<dead_func_eliminate_t>());
    ret.emplace_back(module_function_pass_t::make<bf16_fp16_eliminator_t>(ctx));
    ret.emplace_back(utils::make_unique<target_specific_lowering_cpu_t>(ctx));
    ret.emplace_back(utils::make_unique<func_inliner_t>());
    ret.emplace_back(utils::make_unique<dead_func_eliminate_t>());
    ret.emplace_back(module_function_pass_t::make<loop_unroller_t>());
    ret.emplace_back(module_function_pass_t::make<ir_simplifier_t>(false));

    if (ctx->flags_.tensor2var_) {
        ret.emplace_back(module_function_pass_t::make<tensor2var_t>());
    }
    if (ctx->flags_.buffer_schedule_ > 0) {
        ret.emplace_back(module_function_pass_t::make<buffer_scheduler_t>(
                ctx, true, ctx->flags_.tensor_inplace_));
    }
    ret.emplace_back(utils::make_unique<kernel_lowering_cpu_t>(
            ctx->flags_.kernel_optim_));
    ret.emplace_back(utils::make_unique<closurizer_cpu_t>(
            runtime_config_t::get().get_num_threads() == 1));
    ret.emplace_back(module_function_pass_t::make<ir_simplifier_t>(false));

    ret.emplace_back(utils::make_unique<module_globals_resolver_t>());
    ret.emplace_back(
            module_function_pass_t::make<local_tensor_lowering_cpu_t>(128));
    if (ctx->flags_.ssa_passes_) {
        ret.emplace_back(module_function_pass_t::make<ssa_transform_t>());
        ret.emplace_back(module_function_pass_t::make<value_numbering_t>());
        ret.emplace_back(
                module_function_pass_t::make<loop_invariant_code_motion_t>());
        ret.emplace_back(module_function_pass_t::make<dessa_transform_t>());
    }
    validate_pass_order(ctx, ret, gen_wrapper);
    return sequential_module_pass_t(std::move(ret));
}

const_ir_module_ptr run_precodegen_passes(
        module_pass_t &pass, const_ir_module_ptr mod) { // NOLINT
    func_t init_func = mod->make_init_func();
    auto mod_with_init = std::make_shared<ir_module_t>(*mod);
    if (init_func) { mod_with_init->add_func({init_func}); }
    // todo: use attr in function to skip some of the passes
    auto mod_cpy = pass(mod_with_init);
    if (mod->ctx_->flags_.print_ir_) { std::cerr << mod_cpy; }
    return mod_cpy;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
