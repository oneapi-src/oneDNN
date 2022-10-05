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

#ifndef GPU_JIT_PASS_SLM_HPP
#define GPU_JIT_PASS_SLM_HPP

#include "gpu/jit/ir/ir.hpp"
#include "gpu/jit/ir/tensor.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

// Merges all SLM buffers into a single one.
stmt_t merge_slm_buffers(const stmt_t &_stmt, ir_context_t &ir_ctx);

// Replaces some heavy GRF reorders by reorder through SLM (store and load).
stmt_t inject_slm_reorder(const stmt_t &s, ir_context_t &ir_ctx,
        const grid_info_t &tg_grid, bool has_slm_usage);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
