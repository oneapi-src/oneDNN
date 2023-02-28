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

#include <memory>
#include <stdint.h>
#include <vector>
#include "function_pass.hpp"
#include "module_pass.hpp"
#include "pass_id.hpp"
#include "pass_manager.hpp"
#include "util_module_passes.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

const char *get_pass_name(module_pass_t *pass) {
    if (auto ret = pass->get_name()) { return ret; }
    auto &tyid = typeid(*pass);
    if (tyid == typeid(module_function_pass_t)) {
        auto &ptr = *static_cast<module_function_pass_t *>(pass)->impl_;
        return typeid(ptr).name();
    }
    return tyid.name();
}

#ifndef NDEBUG

void module_pass_t::get_dependency_info(tir_pass_dependency_t &out) const {
    out = tir_pass_dependency_t {tir_pass::undef};
}

void function_pass_t::get_dependency_info(tir_pass_dependency_t &out) const {
    out = tir_pass_dependency_t {tir_pass::undef};
}

static size_t cast_to_int(tir_pass::pass_id v) {
    return static_cast<size_t>(v);
}

static const char *tir_state_names[]
        = {"CONST_FOLDED", "IR_SIMPLIFIED", "FUNC_LININED", "SSA_STAGE"};
static_assert(sizeof(tir_state_names) / sizeof(tir_state_names[0])
                == tir_pass::state::NUM_STATES,
        "Bad number of tir_state_names");

void validate_pass_order(const context_ptr &ctx,
        const std::vector<std::unique_ptr<module_pass_t>> &passes,
        bool gen_wrapper) {
    uint64_t state = 0;
    std::vector<bool> visited(
            cast_to_int(tir_pass::pass_id::MAX_ID_PLUS_1), false);
    for (size_t i = 0; i < passes.size(); i++) {
        auto pass = passes[i].get();
        tir_pass_dependency_t dep;
        pass->get_dependency_info(dep);
        visited[cast_to_int(dep.id_)] = true;
        for (auto prev : dep.depending_) {
            if (prev == tir_pass::trace_inserter && !ctx->flags_.trace_) {
                continue;
            }
            if ((prev == tir_pass::tensor_inplace
                        || prev == tir_pass::buffer_scheduler)
                    && ctx->flags_.buffer_schedule_ <= 0) {
                continue;
            }
            if (prev == tir_pass::interface_generalizer && !gen_wrapper) {
                continue;
            }
            if (prev == tir_pass::tensor2var && !ctx->flags_.tensor2var_) {
                continue;
            }
            if (prev == tir_pass::index2var && !ctx->flags_.index2var_) {
                continue;
            }
            if (prev == tir_pass::dead_write_eliminator
                    && !ctx->flags_.dead_write_elimination_) {
                continue;
            }
            COMPILE_ASSERT(visited[cast_to_int(prev)],
                    "The pass " << get_pass_name(pass) << " at index " << i
                                << " depends on pass_id " << cast_to_int(prev)
                                << ", but this requirement is not satisfied");
        }
        for (int sid = 0; sid < tir_pass::state::NUM_STATES; sid++) {
            uint64_t mask = uint64_t(1) << sid;
            if ((dep.required_state_ & mask) && ((state & mask) == 0)) {
                COMPILE_ASSERT(false,
                        "The pass "
                                << get_pass_name(pass) << " at index " << i
                                << " requires the state "
                                << tir_state_names[sid]
                                << ", but this requirement is not satisfied");
            }

            if ((dep.required_not_state_ & mask) && ((state & mask) != 0)) {
                COMPILE_ASSERT(false,
                        "The pass "
                                << get_pass_name(pass) << " at index " << i
                                << " rejects the state " << tir_state_names[sid]
                                << ", but this requirement is not satisfied");
            }
        }
        state |= dep.set_state_;
        state &= ~dep.unset_state_;
    }
}

#endif

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
