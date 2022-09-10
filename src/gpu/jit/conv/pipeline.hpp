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

#ifndef GPU_JIT_CONV_PIPELINE_HPP
#define GPU_JIT_CONV_PIPELINE_HPP

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/utils/trace.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

stmt_t inject_prefetch_pipeline(
        const stmt_t &s, ir_context_t &ir_ctx, const conv_config_t &cfg);

// Injects SLM buffering without unrolling based on the config.
stmt_t inject_simple_slm_buffering(const stmt_t &s, ir_context_t &ir_ctx,
        const conv_config_t &cfg, int ab_slm_size);

// Injects loop unrolling based on the config. Possible options:
// - Without preload (no SLM buffering, no prefetch)
// - With SLM buffering
// - With prefetch
stmt_t inject_unrolling(const stmt_t &s, ir_context_t &ir_ctx,
        const conv_config_t &cfg, int ab_slm_size);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
