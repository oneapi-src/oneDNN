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

#include "gpu/jit/pass/barrier.hpp"

#include "gpu/jit/ir/message.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class barrier_optimizer_t : public ir_mutator_t {
public:
    object_t _mutate(const for_t &obj) override {
        loop_level_++;
        auto new_obj = ir_mutator_t::_mutate(obj);
        loop_level_--;
        return new_obj;
    }

    object_t _mutate(const func_call_t &obj) override {
        if (is_func_call<send_t>(obj)) {
            auto &send = obj.func.as<send_t>();
            if (send.is_slm()) can_remove_barrier_ = false;
        } else if (obj.func.is_same(funcs::barrier_func())) {
            bool can_remove = can_remove_barrier_;
            can_remove_barrier_ = false;

            // If not in a loop and this is the first barrier -> can be removed.
            if (loop_level_ == 0 && can_remove) return stmt_t();
            return obj;
        }

        return obj;
    }

    // Store doesn't contain nested statements, return as is.
    object_t _mutate(const store_t &obj) override { return obj; }

private:
    int loop_level_ = 0;
    bool can_remove_barrier_ = true;
};

stmt_t optimize_barrier(const stmt_t &s, ir_context_t &ir_ctx) {
    trace_start();
    auto ret = barrier_optimizer_t().mutate(s);
    trace_pass("optimize_barrier", ret, ir_ctx);
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
