/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#include "fusion_cost_model.hpp"
#include <utility>
#include "fusible_op_utils.hpp"
#include "fusion_anchor.hpp"
#include "mixed_partition.hpp"

SC_MODULE(graph.fusion_cost_model);

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static int op_num(mixed_parti_t *parti) {
    int num = parti->ops.size();
    return num;
}

static int loop_parallelism(mixed_parti_t *parti) {
    auto outer_loops = parti->get_outer_loops();
    return evaluate_loop_parallel_balance(outer_loops);
}

std::vector<std::pair<float, cost_eval>> create_default_evaluator() {
    std::vector<std::pair<float, cost_eval>> inits;
    inits.emplace_back(std::make_pair(0.1f, op_num));
    inits.emplace_back(std::make_pair(1.0f, loop_parallelism));
    return inits;
}

fusion_cost_model_base_t::fusion_cost_model_base_t(mixed_parti_t *parti)
    : binded_mxp_(parti), enable_(parti->ctx_->flags_.use_cost_model_) {}

static_fusion_cost_model_t::static_fusion_cost_model_t(mixed_parti_t *parti)
    : fusion_cost_model_base_t(parti)
    , max_scores_(0)
    , evaluators_(create_default_evaluator()) {}

float static_fusion_cost_model_t::evaluate() {
    float new_scores = 0;
    if (!enable_) return new_scores;
    for (auto &eval : evaluators_) {
        new_scores += eval.first * eval.second(binded_mxp_);
    }
    if (new_scores > max_scores_) max_scores_ = new_scores;
    return new_scores;
}

void static_fusion_cost_model_t::append_evaluator(
        float weight, const cost_eval &eval) {
    evaluators_.emplace_back(std::make_pair(weight, eval));
}

bool static_fusion_cost_model_t::make_decision_for_parti(
        const mixed_parti_t *parti, size_t merged_loop_size,
        parti_merge_kind merge_kind) {
    // query if turn on
    if (!enable_) return true;
    /* loop_parallelism */
    auto ths_outer_loop = binded_mxp_->get_outer_loops();
    auto other_outer_loop = parti->get_outer_loops();
    COMPILE_ASSERT(!ths_outer_loop.empty() && !other_outer_loop.empty(),
            "Could not merge empty loop")
    COMPILE_ASSERT((merged_loop_size <= ths_outer_loop.size())
                    && (merged_loop_size <= other_outer_loop.size()),
            "merge loop size should less than both loop")
    if (merge_kind == parti_merge_kind::horizontal) {
        return evaluate_loop_parallel_balance(ths_outer_loop) != 1.0f
                && evaluate_loop_parallel_balance(other_outer_loop) != 1.0f;
    }
    /* for verticall merge*/
    COMPILE_ASSERT(merge_kind == parti_merge_kind::vertical,
            "No cost metric found for parallel merge")
    // in avoid of loss for loop optimize opportunity
    if (binded_mxp_->can_optimize_outer_loop(true)
            ^ parti->can_optimize_outer_loop(true)) {
        return false;
    }

    // check loop parallelism
    auto ths_loop_parallelism = evaluate_loop_parallel_balance(ths_outer_loop);
    auto other_loop_parallelism
            = evaluate_loop_parallel_balance(other_outer_loop);
    auto merged_outer_loop = std::vector<for_loop> {
            ths_outer_loop.begin(), ths_outer_loop.begin() + merged_loop_size};
    auto merged_loop_parallelism
            = evaluate_loop_parallel_balance(merged_outer_loop);
    if (merged_loop_parallelism < ths_loop_parallelism
            || merged_loop_parallelism < other_loop_parallelism) {
        // evalute workload size
        if (!evaluate_loop_parallel_balance(merged_outer_loop, true)
                || ((merged_loop_parallelism < ths_loop_parallelism)
                        && !binded_mxp_->is_small_workload())
                || ((merged_loop_parallelism < other_loop_parallelism)
                        && !parti->is_small_workload())) {
            SC_MODULE_INFO << "rejects to merge two "
                              "partition: "
                           << binded_mxp_->func_->name_ << " and "
                           << parti->func_->name_
                           << " from perspective of loop parallelism";
            return false;
        }
    }

    /* cache efficiency */
    // skip standalone parti
    if (binded_mxp_->ops.size() == 1) return true;
    // get real merged trace and merged inplace map
    auto merged_mem_info
            = merge_real_mem_info(binded_mxp_->buf_alloc_, parti->buf_alloc_);
    // get real buffer usage
    auto buffer_usage = get_buffer_usage(
            binded_mxp_->ctx_, merged_mem_info.first, merged_mem_info.second);
    // get threshold
    auto threshold = binded_mxp_->ctx_->machine_.cpu_flags_.getDCacheSize(2);

    // check cache efficiency
    if (buffer_usage > threshold) {
        SC_MODULE_INFO << "rejects to merge two "
                          "partition: "
                       << binded_mxp_->func_->name_ << " and "
                       << parti->func_->name_
                       << " from perspective of cache efficiency";
        return false;
    } else {
        return true;
    }
}

bool static_fusion_cost_model_t::make_decision_for_op(
        const sc_op *op, const fuse_anchor_map_ptr &fanchor) {
    // query if turn on
    if (!enable_) return true;
    /** Auto Skip List:
     * 1. empty partition
     * 2. nested parallel template
     * 3. singel op lowering
     * */
    if (binded_mxp_->empty() || binded_mxp_->contain_nested_parallel_for()
            || op->get_owner_graph().attrs_.get_or_else(
                    mixed_partition_hint::single_op_graph, false))
        return true;

    auto orig_loop_parallelism
            = evaluate_loop_parallel_balance(binded_mxp_->get_outer_loops());
    auto fanchor_loop_parallelism = evaluate_loop_parallel_balance(
            binded_mxp_->get_outer_loops(fanchor));

    bool ret = (!binded_mxp_->contain_tunable_op() && !op->isa<tunable_op_t>())
            || (fanchor_loop_parallelism >= orig_loop_parallelism);

    bool double_check_standalone_parallel = (op->isa<tunable_op_t>()
            || (is_broadcast_op(op) && !binded_mxp_->contain_tunable_op()
                    && !binded_mxp_->can_optimize_outer_loop(true)));

    // double check parallelism of standalone op
    if (double_check_standalone_parallel) {
        mixed_parti_t op_parti(binded_mxp_->ctx_,
                std::const_pointer_cast<sc_op>(op->shared_from_this()),
                nullptr);
        float standalone_parallel
                = evaluate_loop_parallel_balance(op_parti.get_outer_loops());
        // if original result of partition can not meet requirement
        if (ret
                && !evaluate_loop_parallel_balance(
                        binded_mxp_->get_outer_loops(), true)) {
            // if new parti created by op owning more loop parallelism, reject
            // to fuse it
            if (standalone_parallel > fanchor_loop_parallelism
                    && !fanchor->is_small_op_workload(op)) {
                ret = false;
            }
        } else if (!ret
                && evaluate_loop_parallel_balance(
                        binded_mxp_->get_outer_loops(fanchor), true)) {
            // if new parti created by op can not meet loop parallelism
            // requirement, suggest to fuse it anyway.
            if (standalone_parallel != 1.f) { ret = true; }
        }
    }
    if (!ret) {
        SC_MODULE_INFO << "rejects to commit op: " << op->op_name_
                       << op->logical_op_id_
                       << " into current fusion anchor from "
                          "perspective of parallellism";
    }
    return ret;
}

dynamic_fusion_cost_model_t::dynamic_fusion_cost_model_t(
        mixed_parti_t *parti, dynamic_fusion_policy_t policy)
    : fusion_cost_model_base_t(parti), cond_(false), policy_(policy) {}

bool dynamic_fusion_cost_model_t::make_decision_for_parti(
        const mixed_parti_t *parti, size_t merged_loop_size,
        parti_merge_kind merge_kind) {
    // query if turn on
    if (!enable_ || policy_ == dynamic_fusion_policy_t::max_fusion) return true;
    /* loop_parallelism */
    auto ths_outer_loops = binded_mxp_->get_outer_loops();
    auto other_outer_loops = parti->get_outer_loops();
    COMPILE_ASSERT(!ths_outer_loops.empty() && !other_outer_loops.empty(),
            "Could not merge empty loop")
    COMPILE_ASSERT((merged_loop_size <= ths_outer_loops.size())
                    && (merged_loop_size <= other_outer_loops.size()),
            "merge loop size should less than both loop");
    expr res_cond, dummy_cond;
    if (merge_kind == parti_merge_kind::horizontal) {
        float ths_parallelism = evaluate_loop_parallel_balance(
                {ths_outer_loops[0]}, res_cond);
        float other_parallelism = evaluate_loop_parallel_balance(
                {other_outer_loops[0]}, dummy_cond);
        bool ret = ths_parallelism != 1.0f && other_parallelism != 1.0f;
        if (!ret) { cond_ = cond_ || !(res_cond && dummy_cond); }
        return ret;
    }
    /* for verticall merge*/
    COMPILE_ASSERT(merge_kind == parti_merge_kind::vertical,
            "No cost metric found for parallel merge")
    // in avoid of loss for loop optimize opportunity
    if (binded_mxp_->can_optimize_outer_loop(true)
            ^ parti->can_optimize_outer_loop(true)) {
        return false;
    }

    // check loop parallelism
    float ths_parallelism
            = evaluate_loop_parallel_balance(ths_outer_loops, dummy_cond);
    float other_parallelism
            = evaluate_loop_parallel_balance(other_outer_loops, dummy_cond);
    auto merged_outer_loop = std::vector<for_loop> {ths_outer_loops.begin(),
            ths_outer_loops.begin() + merged_loop_size};
    float merged_parallelism
            = evaluate_loop_parallel_balance(merged_outer_loop, res_cond);
    if (merged_parallelism < ths_parallelism
            || merged_parallelism < other_parallelism) {
        SC_MODULE_INFO << "rejects to merge two "
                          "partition: "
                       << binded_mxp_->func_->name_ << " and "
                       << parti->func_->name_
                       << " from perspective of loop parallelism";
        cond_ = cond_ || res_cond;
        return false;
    }
    // don't set cond_ if accept the parti.
    /* find how to describe cache efficiency */
    return true;
}

bool dynamic_fusion_cost_model_t::make_decision_for_op(
        const sc_op *op, const fuse_anchor_map_ptr &fanchor) {
    // query if turn on
    if (!enable_ || policy_ == dynamic_fusion_policy_t::max_fusion) return true;
    // auto skip
    if (!binded_mxp_->contain_tunable_op() && !op->isa<tunable_op_t>())
        return true;
    expr res_cond, thr_cond, dummy_cond;
    auto ths_outer_loops = binded_mxp_->get_outer_loops();
    auto other_outer_loops = binded_mxp_->get_outer_loops(fanchor);

    auto ths_parallelism
            = evaluate_loop_parallel_balance(ths_outer_loops, dummy_cond);
    auto other_parallelism
            = evaluate_loop_parallel_balance(other_outer_loops, res_cond);
    bool ret = ths_parallelism <= other_parallelism;
    if (op->isa<tunable_op_t>()
            && evaluate_loop_parallel_balance(
                       binded_mxp_->get_outer_loops(), thr_cond, true)
                    == 0.f) {
        mixed_parti_t tunable_parti(binded_mxp_->ctx_,
                std::const_pointer_cast<sc_op>(op->shared_from_this()),
                nullptr);
        if (evaluate_loop_parallel_balance(
                    tunable_parti.get_outer_loops(), dummy_cond)
                > other_parallelism) {
            ret = false;
            cond_ = cond_ || thr_cond;
        }
    }
    if (!ret) {
        SC_MODULE_INFO << "rejects to commit op: " << op->op_name_
                       << op->logical_op_id_
                       << " into current fusion anchor from "
                          "perspective of parallellism";
        cond_ = cond_ || res_cond;
        return false;
    }
    // don't set cond_ when accept the op
    return true;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
