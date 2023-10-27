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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_COST_MODEL_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_FUSION_COST_MODEL_HPP
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include <compiler/ir/sc_expr.hpp>
#include <unordered_set>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

class sc_op;
struct mixed_parti_t;
struct fusion_anchor_t;
enum class parti_merge_kind;
enum class dynamic_fusion_policy_t;

using cost_eval = std::function<int(mixed_parti_t *)>;

/**
 * fusion cost model has two main function:
 * 1. make final decision for fusing or not in consider of loop parallelism
 * or cache locality.
 * 2. evaluate fusion partition and throw warning when prediction is under
 * expectation.
 * */
struct fusion_cost_model_base_t {
protected:
    mixed_parti_t *binded_mxp_;
    bool enable_;

public:
    fusion_cost_model_base_t(mixed_parti_t *parti);
    virtual ~fusion_cost_model_base_t() {}
    // disable cost model
    void disable() { enable_ = false; }
    // judge if it is enabled
    bool is_enabled() const { return enable_; }
    // evaluate current mixed partition by several evaluator
    virtual float evaluate() = 0;
    // make decision for partition merge
    virtual bool make_decision_for_parti(const mixed_parti_t *parti,
            size_t merge_loop_size, parti_merge_kind merge_kind)
            = 0;
    // make decision for op and fusion anchor
    virtual bool make_decision_for_op(
            const sc_op *op, const std::shared_ptr<fusion_anchor_t> &fanchor)
            = 0;
    virtual expr get_fusion_policy_condition() const { return false; }
};

struct static_fusion_cost_model_t : public fusion_cost_model_base_t {
private:
    float max_scores_; // cache the top scores
    std::vector<std::pair<float, cost_eval>> evaluators_;

public:
    static_fusion_cost_model_t(mixed_parti_t *parti);
    // evaluate current mixed partition by several evaluator
    float evaluate() override;
    // append new defined evaluator
    void append_evaluator(float weight, const cost_eval &eval);
    // make decision for partition merge
    bool make_decision_for_parti(const mixed_parti_t *parti,
            size_t merge_loop_size, parti_merge_kind merge_kind) override;
    // make decision for op and fusion anchor
    bool make_decision_for_op(const sc_op *op,
            const std::shared_ptr<fusion_anchor_t> &fanchor) override;
};

struct dynamic_fusion_cost_model_t : public fusion_cost_model_base_t {
private:
    expr cond_;
    dynamic_fusion_policy_t policy_;

public:
    dynamic_fusion_cost_model_t(
            mixed_parti_t *parti, dynamic_fusion_policy_t policy);
    // evaluate current mixed partition by several evaluator
    float evaluate() override { return 0.f; }
    expr get_fusion_policy_condition() const override { return cond_; }
    // make decision for partition merge
    bool make_decision_for_parti(const mixed_parti_t *parti,
            size_t merge_loop_size, parti_merge_kind merge_kind) override;
    // make decision for op and fusion anchor
    bool make_decision_for_op(const sc_op *op,
            const std::shared_ptr<fusion_anchor_t> &fanchor) override;
};

using fusion_cost_model_ptr = std::shared_ptr<fusion_cost_model_base_t>;

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
