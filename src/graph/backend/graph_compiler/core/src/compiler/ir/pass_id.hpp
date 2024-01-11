/*******************************************************************************
 * Copyright 2022-2024 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_ID_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASS_ID_HPP

#include <stdint.h>
#include <vector>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace tir_pass {

enum pass_id {
    undef = 0,
    dyn_tensor_transformer,
    interface_generalizer,
    concat_memory_planning,
    index_flattener,
    auto_caster,
    validator,
    trace_inserter,
    constant_folder,
    dead_func_eliminate,
    tensor_inplace,
    target_specific_lowering_cpu,
    kernel_lowering_cpu,
    closurizer_cpu,
    module_globals_resolver,
    parallel_merge,

    FUNCTION_PASS_START,
    tensor_shrinker = FUNCTION_PASS_START,
    bf16_fp16_legalizer,
    dynamic_parallel_transform,
    buffer_rescheduling_tensor_hoisting,
    nested_parallel_flattener,
    func_inliner,
    ir_simplifier,
    loop_merger,
    tensor_init,
    parallel_workload_dispatcher,
    simple_loop_invariant_code_motion,
    simple_loop_function_motion,
    index2var,
    bf16_fp16_eliminator,
    loop_unroller,
    dead_write_eliminator,
    tensor2var,
    buffer_scheduler,
    dyn_boundary_check,
    local_tensor_lowering_cpu,
    loop_splitter,
    ssa_transform,
    value_numbering,
    loop_invariant_code_motion,
    dessa_transform,

    MAX_ID_PLUS_1
};

// the states in the pass manager. Initialized with 0
enum state {
    CONST_FOLDED,
    IR_SIMPLIFIED,
    FUNC_INLINED,
    SSA_STAGE,

    NUM_STATES
};
} // namespace tir_pass

#ifndef NDEBUG
struct tir_pass_dependency_t {
    tir_pass::pass_id id_;
    // the passes to run before this pass
    std::vector<tir_pass::pass_id> depending_;
    // the required bits in the global pass state. If a bit in
    // required_state_ is 1, the corresponding bit must be 1 in global pass
    // state
    uint64_t required_state_;
    // the required unset bits in the global pass state. If a bit in
    // required_not_state_ is 1, the corresponding bit must be 0 in global pass
    // state
    uint64_t required_not_state_;
    // the bit mask to set to the global pass state after the pass
    uint64_t set_state_;
    // the bit mask to unset to the global pass state after the pass
    uint64_t unset_state_;

    tir_pass_dependency_t(tir_pass::pass_id id = tir_pass::pass_id::undef,
            const std::vector<tir_pass::pass_id> &depending = {},
            uint64_t required_state = 0, uint64_t required_not_state = 0,
            uint64_t set_state = 0, uint64_t unset_state = 0)
        : id_(id)
        , depending_(depending)
        , required_state_(required_state)
        , required_not_state_(required_not_state)
        , set_state_(set_state)
        , unset_state_(unset_state) {}
};
#endif

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
