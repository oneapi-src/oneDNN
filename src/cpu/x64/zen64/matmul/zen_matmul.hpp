/*******************************************************************************
* Copyright 2026 Advanced Micro Devices, Inc.
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

#ifndef CPU_X64_ZEN64_MATMUL_ZEN_MATMUL_HPP
#define CPU_X64_ZEN64_MATMUL_ZEN_MATMUL_HPP

#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

#if DNNL_X64_USE_ZEN
#include "lowoha_operators/matmul/lowoha_common.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace zen {
namespace matmul {

#if DNNL_X64_USE_ZEN
namespace zen_matmul = zendnnl::lowoha::matmul;
#endif

struct zen_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("zen:matmul:f32|bf16|f16:amd", zen_matmul_t);

        status_t init(const engine_t *engine);
    };

    zen_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    // Build Zen post-op chain once per primitive (mirrors brgemm_matmul_t
    // convention; keeps pd_t cheaply-copyable for the primitive cache).
    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_body(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_body(const exec_ctx_t &ctx) const;

#if DNNL_X64_USE_ZEN
    // Pre-built Zen post-op chain. Owned by the primitive (never copied
    // by the framework). Binary buffer pointers are patched at execute time
    // from the execution context.
    std::vector<zen_matmul::matmul_post_op> zen_postop_;
    // oneDNN post-op index per entry (for binary buffer patching).
    std::vector<int> postop_indices_;
    float beta_ = 0.f;
#endif
};

} // namespace matmul
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
