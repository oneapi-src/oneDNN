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

#include "gpu/jit/conv/builder_utils.hpp"

#include "gpu/jit/conv/grf_usage.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

#ifdef GEN_CONV_PROFILE
ir_utils::debug_profiler_t &get_trace_profiler() {
    static thread_local ir_utils::debug_profiler_t profiler("Trace Profile");
    return profiler;
}
#endif

#if defined(GEN_CONV_PROFILE) || defined(GEN_CONV_DEBUG)
void trace_pass(
        const char *pass_name, const stmt_t &stmt, ir_context_t &ir_ctx) {
    trace_stop(pass_name);
    ir_trace() << "=== After " << pass_name << std::endl;
    ir_trace() << stmt << std::endl;
#ifdef GEN_CONV_DEBUG
    auto grf_usage = get_grf_usage(stmt, ir_ctx.hw_cfg().grf_size());
    if (!grf_usage.is_empty()) ir_trace() << grf_usage << std::endl;
#endif
}
#endif

// Performs the following operation:
//     buf = alpha * buf + beta
stmt_t create_mul_add_stmt(ir_context_t &ir_ctx, const expr_t &buf, int size,
        const type_t &type, float alpha, float beta) {
    if (alpha == 1 && beta == 0) return stmt_t();

    stmt_t ret;
    int step_bytes = 2 * ir_ctx.hw_cfg().grf_size();
    for (int i = 0; i < size; i += step_bytes) {
        auto elems = std::min(step_bytes, size - i) / type.size();
        auto e_alpha = shuffle_t::make_broadcast(alpha, elems);
        auto e_beta = shuffle_t::make_broadcast(beta, elems);
        auto e = load_t::make(type.with_elems(elems), buf, i);
        // Avoid extra IR expressions when not needed.
        if (alpha == 0)
            e = shuffle_t::make_broadcast(expr_t(0.0f), elems);
        else if (alpha != 1)
            e *= e_alpha;
        if (beta != 0) e += e_beta;
        ir_assert(e.type().scalar() == type);
        ret = ret.append(store_t::make(buf, i, e));
    }
    return ret;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
