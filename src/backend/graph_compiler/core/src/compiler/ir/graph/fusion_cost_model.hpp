/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_COST_MODEL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_COST_MODEL_HPP
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <unordered_set>

namespace sc {

class sc_op;
struct mixed_parti_t;
struct fuse_anchor_map_t;
enum class parti_merge_kind;

using cost_eval = std::function<int(mixed_parti_t *)>;

/**
 * fusion cost model has two main function:
 * 1. make final decision for fusing or not in consider of loop parallelism
 * or cache locality.
 * 2. evaluate fusion partition and throw warning when prediction is under
 * expectation.
 * */
struct fusion_cost_model {
private:
    float max_scores_; // cache the top scores
    std::vector<std::pair<float, cost_eval>> evaluators_;
    mixed_parti_t *binded_mxp_;
    bool enable_;

public:
    fusion_cost_model(mixed_parti_t *parti);
    // evaluate current mixed partition by several evaluator
    float evaluate();
    // append new defined evaluator
    void append_evaluator(float weight, const cost_eval &eval);
    // disable cost model
    void disable() { enable_ = false; }
    // make decision for partition merge
    bool make_decision_for_parti(const mixed_parti_t *parti,
            size_t merge_loop_size, parti_merge_kind merge_kind);
    // make decision for op and fusion anchor
    bool make_decision_for_op(
            const sc_op *op, const std::shared_ptr<fuse_anchor_map_t> &fanchor);
};

} // namespace sc
#endif
