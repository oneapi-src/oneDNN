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

#include "gpu/jit/ir/message_patterns.hpp"

#include "gpu/jit/ir/message.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

const std::vector<uniform_blocked_pattern_t> &get_uniform_blocked_patterns(
        compute::gpu_arch_t arch) {
    switch (arch) {
        case compute::gpu_arch_t::xe_hpc: {
            static const std::vector<uniform_blocked_pattern_t> xe_hpc_loads
                    = {uniform_blocked_pattern_t(512, 16),
                            uniform_blocked_pattern_t(256, 16),
                            uniform_blocked_pattern_t(128, 16)};
            return xe_hpc_loads;
        }
        case compute::gpu_arch_t::xe_hp: {
            static const std::vector<uniform_blocked_pattern_t> xe_hp_loads
                    = {uniform_blocked_pattern_t(256, 16),
                            uniform_blocked_pattern_t(128, 16)};
            return xe_hp_loads;
        }
        default: {
            static const std::vector<uniform_blocked_pattern_t> default_list {
                    uniform_blocked_pattern_t(128, 16)};
            return default_list;
        }
    }
}

class uniform_blocked_matcher_t : public ir_visitor_t {
public:
    static bool is_match(const send_pattern_t &pattern, const stmt_t &stmt) {
        if (!pattern.is_uniform_blocked()) return false;

        uniform_blocked_matcher_t matcher(pattern.as_uniform_blocked());
        matcher.visit(stmt);
        return matcher.is_match_;
    };

    void _visit(const func_impl_t &obj) override {
        if (!obj.is<send_t>()) return;

        auto &s = obj.as<send_t>();

        // Larger blocked or 2D messages are a strict improvement
        if ((s.is_block() || s.is_2d()) && s.access_size() >= pattern.size)
            return;

        is_match_ = false;
    }

private:
    uniform_blocked_matcher_t(const uniform_blocked_pattern_t &pattern)
        : pattern(pattern), is_match_(true) {}
    uniform_blocked_pattern_t pattern;
    bool is_match_;
};

bool send_pattern_t::matches(const stmt_t &stmt) const {
    switch (type_id_) {
        case empty: return true;
        case uniform_blocked: {
            return uniform_blocked_matcher_t::is_match(*this, stmt);
        }
        default: return false;
    }
}
} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
