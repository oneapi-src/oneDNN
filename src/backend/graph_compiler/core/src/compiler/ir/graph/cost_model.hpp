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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_COST_MODEL_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_COST_MODEL_HPP
#include <functional>
#include <utility>
#include <vector>
#include <unordered_set>

namespace sc {

struct mixed_parti_t;

using cost_eval = std::function<int(mixed_parti_t *)>;

struct cost_model {
private:
    float max_scores_; // cache the top scores
    std::vector<std::pair<float, cost_eval>> evaluators_;

public:
    cost_model();
    // evaluate current mixed partition by several evaluator
    float evaluate(mixed_parti_t *parti);
    // append new defined evaluator
    void append_evaluator(float weight, const cost_eval &eval);
};

} // namespace sc
#endif
