/*******************************************************************************
 * Copyright 2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_USE_COUNT_ANALYSIS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_PASSLET_USE_COUNT_ANALYSIS_HPP

#include "passlet.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {

/**
 * The passlet to analyze the usage of each expr
 * */

struct use_count_analysis_t : public typed_passlet<size_t> {
    using typed_addresser_t = typed_passlet<size_t>::typed_addresser_t;

    use_count_analysis_t(const typed_addresser_t &expr_result_func)
        : typed_passlet<size_t>(expr_result_func, nullptr) {}
    void view(const expr_c &v, pass_phase phase);
    void view(const func_c &v, pass_phase phase);
    void view(const define_c &v, pass_phase phase);
    void view(const for_loop_c &v, pass_phase phase);
};
} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
#endif
