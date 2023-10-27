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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BUFFER_SCHEDULE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_BUFFER_SCHEDULE_HPP

#include <utility>
#include <vector>
#include "../function_pass.hpp"
#include <compiler/config/context.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace attr_keys {
// the buffer scheduler type: 0 - no buffer schedule, 1 - whole buffer reuse, 2
// - static memory planner (minimize size), 3 - static memory planner (hot
// memory first)
constexpr const char *buf_sched_type = "pass.buf_sched_type";
// hint tick info for tensors in loops from graph or fusion mgr: int64_t. It
// guides the buffer scheduler to correctly compute the tensor life time. Buffer
// scheduler will it add to current tick to calculate final tick
constexpr const char *hint_first_access_tick = "pass.hint_first_access_tick";
constexpr const char *hint_last_access_tick = "pass.hint_last_access_tick";
constexpr const char *tsr_dont_buf_sched = "pass.tsr_dont_buf_sched";
// applied on functions. If true, the func has already been processed by
// buffer_scheduler_t
constexpr const char *already_buf_sched = "pass.already_buf_sched";
constexpr int BUF_SCHED_NONE = 0;
constexpr int BUF_SCHED_WHOLE = 1;
constexpr int BUF_SCHED_SIZE = 2;
constexpr int BUF_SCHED_HOT = 3;
// the tensor inplace hint, applied on temp tensors. It should be a
// vector<temp_tensor_inplace_info_t>.
constexpr const char *tensor_inplace_hint = "pass.tensor_inplace_hint";
// applied on for-loops. If a for-loop is attached with this attr = true, buffer
// scheduler will treat it as a parallel-for and as a independent scope
constexpr const char *buf_sched_top_scope = "pass.buf_sched_top_level_scope";
// whether the tensor can be scheduled even if there are complex accesses to it
constexpr const char *can_be_scheduled = "can_be_scheduled";
} // namespace attr_keys

/**
 * Schedule tensor buffers to reuse them if they are no longer needed.
 * This pass should only work on 1D tensors. It should be placed after
 * index_flatten
 *
 * 1) We sort all the expressions by execution order and all exprs are assigned
 * a tick. A greater tick means that the expr will be executed later than other
 * expr with less tick.
 *
 * 2) First collect the last-read-tick (LRT), all write ticks (in writes_ set)
 * and first-access-tick (FAT), creation tick, deletion tick for each tensor. We
 * collect these ticks on indexing_nodes, and functions calls. To distinguish
 * writes from reads, we also process assign_nodes (lvalues are written). The
 * function arguments can be annotated with "read_buffer" and "write_buffer" in
 * the function declaration. If no annotation is applied on an argument, the
 * tensor is considered read-written. Special case for "for_loop":
 * the tensors in a for-loop will be accessed mutiple times in "body_" and
 * "iter_end_". We manually set ticks of all tensors accessed in a for-loop to
 * the tick at the end of the loop.
 *
 * 3) Optionally (if eliminate_dead_writes_=true), remove all writes to local
 * tensors which is no longer read, where tick > tensor.LRT
 *
 * 4) Schedule the tensors. For each defined local tensors (in tensor creation
 * order), say, "cur", find another local defined/ function arg tensor, say
 * "candidate", where:
 *  1. cur.FAT > candidate.LRT && cur.FAT >= candidate.creation_tick &&
 * cur.deletion_tick <= candidate.deletion_tick.
 *  2. in the tick set candidate.writes, there are no writes to the candidates
 * that happens between [cur.FAT, cur.LRT].
 *  3. If the candidate is an function argument, make sure that cur writes will
 * not overwrite the candidate's final values: cur.last_write < candidate.FAT
 *
 * If such candidate is found, replace cur with the candidate
 *
 * 5) if "cur" is larger than "candidate" in size, extend candidate
 * */
class buffer_scheduler_t : public function_pass_t {
public:
    context_ptr ctx_;
    // if only transforming the func body, this field should be set to correctly
    // handle the func args
    func_c top_level_ = nullptr;
    bool eliminate_dead_writes_;
    bool do_inplace_opt_;
    buffer_scheduler_t(context_ptr ctx, bool eliminate_dead_writes,
            bool do_inplace_opt = false)
        : ctx_(std::move(ctx))
        , eliminate_dead_writes_(eliminate_dead_writes)
        , do_inplace_opt_(do_inplace_opt) {}
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c f) const;
    SC_DECL_PASS_INFO_FUNC();
};
// todo: if the buffer ("candidate") is larger than the "cur" tensor, we can
// split "candidate" tensor into two and reuse the remaining of it for other
// tensors

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
