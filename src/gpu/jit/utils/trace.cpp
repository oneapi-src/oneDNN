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

#include "gpu/jit/utils/trace.hpp"

#include "gpu/jit/conv/grf_usage.hpp"
#include "gpu/jit/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

#ifdef DNNL_DEV_MODE
ir_utils::debug_profiler_t &get_trace_profiler() {
    static thread_local ir_utils::debug_profiler_t profiler("Trace Profile");
    return profiler;
}
#endif

#if defined(DNNL_DEV_MODE)
void trace_pass(
        const char *pass_name, const stmt_t &stmt, ir_context_t &ir_ctx) {
    trace_stop(pass_name);
    ir_trace() << "=== After " << pass_name << std::endl;
    ir_trace() << stmt << std::endl;
    auto grf_usage = get_grf_usage(stmt, ir_ctx.hw_cfg().grf_size());
    if (!grf_usage.is_empty()) ir_trace() << grf_usage << std::endl;
}
#endif

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl
