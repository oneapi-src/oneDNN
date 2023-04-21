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

#include <functional>
#include <utility>

#include <compiler/ir/ir_utils.hpp>
#include <util/array_ref.hpp>

#include "use_count_analysis.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace passlet {

void use_count_analysis_t::view(const func_c &v, pass_phase phase) {
    if (phase != pass_phase::PRE_VISIT) { return; }
    //
    for (auto &e : v->params_) {
        *get_result(e.get()) = 0;
    }
}

void use_count_analysis_t::view(const expr_c &v, pass_phase phase) {
    if (phase != pass_phase::PRE_VISIT) { return; }
    //
    auto count_use = [this](array_ref<expr> vec) {
        for (auto &e : vec) {
            if (e.isa<var>() || e.isa<tensor>()) { *get_result(e.get()) += 1; }
        }
    };
    if (v.isa<var>() || v.isa<tensor>()) {
        *get_result(v.get()) += 1;
    } else {
        get_direct_dependency_of_expr(v.remove_const(), count_use);
    }
}

void use_count_analysis_t::view(const define_c &v, pass_phase phase) {
    if (phase != pass_phase::PRE_VISIT) { return; }
    *get_result(v->var_.get()) = 0;
}

void use_count_analysis_t::view(const for_loop_c &v, pass_phase phase) {
    if (phase != pass_phase::PRE_VISIT) { return; }
    *get_result(v->var_.get()) = 0;
}

} // namespace passlet
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
