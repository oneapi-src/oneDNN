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

#include "cost_model.hpp"
#include <utility>
#include "mixed_partition.hpp"
#include <runtime/config.hpp>
SC_MODULE(graph.cost_model);

namespace sc {

static int op_num(mixed_parti_t *parti) {
    int num = parti->ops.size();
    return num <= 50 ? num : 0;
}

static int loop_parallelism(mixed_parti_t *parti) {
    auto outer_loops = parti->get_outer_loops();
    auto get_loop_range_prod
            = [](const std::vector<for_loop> &loops) -> int64_t {
        int64_t prod = 1;
        for (auto &loop : loops) {
            if (!(loop->iter_begin_.isa<constant_c>()
                        && loop->iter_end_.isa<constant_c>())) {
                prod *= 0;
            } else {
                auto begin = get_expr_as_int(loop->iter_begin_),
                     end = get_expr_as_int(loop->iter_end_);
                prod *= (end - begin);
            }
        }
        return prod;
    };
    sc_dim prod = get_loop_range_prod(outer_loops);
    if (prod == 1) return 0;
    const int run_threads = runtime_config_t::get().get_num_threads();
    bool parallelism = (prod / run_threads > 8
            || (prod % run_threads == 0 && prod >= run_threads));
    return parallelism ? 1 : 0;
}

std::vector<std::pair<float, cost_eval>> create_default_evaluator() {
    std::vector<std::pair<float, cost_eval>> inits;
    inits.emplace_back(std::make_pair(0.1f, op_num));
    inits.emplace_back(std::make_pair(1.0f, loop_parallelism));
    return inits;
}

cost_model::cost_model() {
    max_scores_ = 0;
    evaluators_ = create_default_evaluator();
}

float cost_model::evaluate(mixed_parti_t *parti) {
    float new_scores = 0;
    for (auto &eval : evaluators_) {
        new_scores += eval.first * eval.second(parti);
    }
    if (new_scores > max_scores_) max_scores_ = new_scores;
    return new_scores;
}

void cost_model::append_evaluator(float weight, const cost_eval &eval) {
    evaluators_.emplace_back(std::make_pair(weight, eval));
}

} // namespace sc
