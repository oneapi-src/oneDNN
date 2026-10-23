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

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/primitive_attr.hpp"
#include "common/type_helpers.hpp"

#include "cpu/gemm_inner_product_utils.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/x64/zen64/common/zen_post_ops_fallback.hpp"

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

        DECLARE_COMMON_PD_T(impl_name(), zen_matmul_t);

        status_t init(const engine_t *engine);

        // Post-ops [0, zen_post_ops_len()) are fused into the Zen kernel. The
        // rest (prepacked weights only) run on the kernel output, in one pass
        // of pp_kernel_t when use_pp_kernel() (with pp_attr() carrying them),
        // or else through post_ops_fallback(). With use_acc(), the kernel
        // writes a dense f32 accumulator instead of dst.
        int zen_post_ops_len() const { return zen_post_ops_len_; }
        bool use_acc() const { return use_acc_; }
        bool use_pp_kernel() const { return use_pp_kernel_; }
        const primitive_attr_t &pp_attr() const { return pp_attr_; }
        const zen::post_ops_fallback_t &post_ops_fallback() const {
            return post_ops_fallback_;
        }

    private:
        status_t init_post_ops(const engine_t *engine, bool wei_already_packed);
        bool pp_kernel_ok(const post_ops_t &post_ops) const;

        // The suffix shows in verbose output that post-ops run after the Zen
        // kernel and how.
        const char *impl_name() const {
            if (use_pp_kernel_) return "zen:matmul:f32|bf16|f16:amd+pp_kernel";
            if (!post_ops_fallback_.empty())
                return "zen:matmul:f32|bf16|f16:amd+post_ops_fallback";
            return "zen:matmul:f32|bf16|f16:amd";
        }

        int zen_post_ops_len_ = 0;
        bool use_acc_ = false;
        bool use_pp_kernel_ = false;
        primitive_attr_t pp_attr_;
        zen::post_ops_fallback_t post_ops_fallback_;
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

    std::unique_ptr<::dnnl::impl::cpu::inner_product_utils::pp_kernel_t>
            pp_kernel_;
    // Copy of the pd's fallback that owns the nested post-op primitives.
    zen::post_ops_fallback_t post_ops_fallback_;

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
