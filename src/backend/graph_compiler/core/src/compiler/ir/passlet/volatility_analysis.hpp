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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_VOLATILITY_ANALYSIS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_VOLATILITY_ANALYSIS_HPP

#include <vector>
#include "passlet.hpp"

namespace sc {
namespace passlet {
struct volatility_result_t {
    enum state_t {
        UNDEF,
        YES,
        NO,
    };
    state_t is_volatile_ = UNDEF;
};

/**
 * The passlet to analyze whether an SSA value is related to any "side-effect".
 * If an SSA value produces/is affected by side-effects (it usually means
 * reads/writes memory, I/O, etc.), it will be marked "YES".
 * @param check_loop_invarient whether to consider loops as "side-effect"
 * @param stmt_result_func the addresser for stmt->volatility_result_t
 * */
struct volatility_analysis_t : public typed_passlet<volatility_result_t> {
    bool check_loop_invarient_;
    using typed_addresser_t
            = typed_passlet<volatility_result_t>::typed_addresser_t;
    // if there is a dependency loop, ususally it means the var depends on a
    // loop. Need to revisit again
    std::vector<const define_node_t *> to_revisit_;
    volatility_analysis_t(bool check_loop_invarient,
            const typed_addresser_t &stmt_result_func);
    void view(const define_c &v, pass_phase phase);
    void view(const func_c &v, pass_phase phase);
};
} // namespace passlet
} // namespace sc
#endif
