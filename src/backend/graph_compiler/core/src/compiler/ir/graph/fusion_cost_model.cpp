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
#include "fusion_data.hpp"
#include "mixed_partition.hpp"

SC_MODULE(graph.fusion_cost_model);

namespace sc {

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

fusion_cost_model::fusion_cost_model(mixed_parti_t *parti) {
    binded_mxp_ = parti;
    enable_ = parti->ctx_->flags_.use_cost_model_;
    max_scores_ = 0;
    evaluators_ = create_default_evaluator();
}

float fusion_cost_model::evaluate() {
    float new_scores = 0;
    if (!enable_) return new_scores;
    for (auto &eval : evaluators_) {
        new_scores += eval.first * eval.second(binded_mxp_);
    }
    if (new_scores > max_scores_) max_scores_ = new_scores;
    return new_scores;
}

void fusion_cost_model::append_evaluator(float weight, const cost_eval &eval) {
    evaluators_.emplace_back(std::make_pair(weight, eval));
}

bool fusion_cost_model::make_decision_for_parti(const mixed_parti_t *parti,
        size_t merged_loop_size, parti_merge_kind merge_kind) {
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
    if (need_optimize_loop_order_for_parti(binded_mxp_, true)
            ^ need_optimize_loop_order_for_parti(parti, true)) {
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
        SC_MODULE_INFO << "rejects to merge two "
                          "partition: "
                       << binded_mxp_->func_->name_ << " and "
                       << parti->func_->name_
                       << " from perspective of loop parallelism";
        return false;
    }

    /* cache efficiency */
    // skip standalone parti
    if (binded_mxp_->ops.size() == 1) return true;
    // get real merged trace
    auto merged_real_mem_trace
            = merge_real_mem_trace(binded_mxp_->buf_alloc_, parti->buf_alloc_);
    // get real buffer usage
    auto buffer_usage = get_buffer_usage_from_trace(
            merged_real_mem_trace, binded_mxp_->ctx_);
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

bool fusion_cost_model::make_decision_for_op(
        const sc_op *op, const fuse_anchor_map_ptr &fanchor) {
    // query if turn on
    if (!enable_) return true;
    // auto skip
    if (!binded_mxp_->contain_tunable_op() && !op->isa<tunable_op_t>())
        return true;
    // skip nested parallel for
    if (binded_mxp_->contain_nested_parallel_for()) return true;
    auto orig_loop_parallelism
            = evaluate_loop_parallel_balance(binded_mxp_->get_outer_loops());

    bool ret = evaluate_loop_parallel_balance(
                       binded_mxp_->get_outer_loops(fanchor))
            >= orig_loop_parallelism;
    if (!ret) {
        SC_MODULE_INFO << "rejects current inferring result "
                          "for op: "
                       << op->op_name_ << op->logical_op_id_
                       << " under current fusion anchor from "
                          "perspective of parallellism";
    }
    return ret;
}

} // namespace sc
