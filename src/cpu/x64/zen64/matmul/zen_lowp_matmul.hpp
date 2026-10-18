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

#ifndef CPU_X64_ZEN64_MATMUL_ZEN_LOWP_MATMUL_HPP
#define CPU_X64_ZEN64_MATMUL_ZEN_LOWP_MATMUL_HPP

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
namespace zen_lowp = zendnnl::lowoha::matmul;

// Per-argument (src/wei/dst) static-quantization scale metadata resolved at
// primitive init() and consumed at execute() (the scale buffer pointers live
// in the execution context and are patched in there). dims follow the ZenDNN
// convention: {1} for per-tensor, {1, N} for per-channel along N.
struct zen_lowp_scale_info_t {
    bool present = false;
    zendnnl::common::data_type_t dt = zendnnl::common::data_type_t::none;
    std::vector<int64_t> dims;
};
#endif

// Low-precision Zen matmul, separate from the f32/bf16 zen_matmul_t so the
// floating-point path is unaffected. Supports 2D matmul and 3D BMM with one
// leading batch dimension, including src/weights batch broadcasting. All
// quantization scales and zero-points are shared across batches.
// Two configurations:
//   int8 static quant: u8/s8 src, s8 wei, u8/s8/s32/f32/bf16 dst; ungrouped
//     f32/bf16 scales (wei also per-channel-N); per-tensor u8 src/dst zp only.
//   WOQ: bf16 src, s4/u4 wei, bf16/f32 dst; needs fpmath_mode bf16|any with
//     apply_to_int and prepacked weights; wei scale per-channel-N or grouped
//     along K; s8 wei zp required for u4, rejected for s4.
//   bias (both): f32/bf16, 1xN.
// Weights must be passed in a plain ab/ba layout (abc/acb with a batch):
// format_any and prepacked (zen_packed) weights are declined, so WOQ declines
// to the next implementation.
struct zen_lowp_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("zen:matmul:lowp:amd", zen_lowp_matmul_t);

        status_t init(const engine_t *engine);

    private:
        // Weight-only-quantization (WOQ / weight decompression) validation:
        // bf16 source with s4/u4 weights dequantized via weight scales
        // (per-channel or per-group along K) and a u4 weight zero-point,
        // which is required for u4 and rejected for s4. Called from init()
        // on the WOQ branch.
        status_t init_woq(const engine_t *engine);
    };

    zen_lowp_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    // Build Zen post-op chain + scale metadata once per primitive (mirrors
    // brgemm_matmul_t convention; keeps pd_t cheaply-copyable for the cache).
    status_t init(engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_body(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_body(const exec_ctx_t &ctx) const;

#if DNNL_X64_USE_ZEN
    // Pre-built Zen post-op chain. Owned by the primitive (never copied by the
    // framework). Binary buffer pointers are patched at execute time.
    std::vector<zen_lowp::matmul_post_op> zen_postop_;
    std::vector<int> postop_indices_;
    float beta_ = 0.f;

    // Static-quantization scale metadata (dtype + granularity dims); buffers
    // are patched from the execution context at execute().
    zen_lowp_scale_info_t src_scale_;
    zen_lowp_scale_info_t wei_scale_;
    zen_lowp_scale_info_t dst_scale_;
    // Zero-point metadata: source (u8 source only) and destination (u8
    // destination only) for int8 static quant, both per-tensor; weight
    // zero-point (u4 weights only) for WOQ, matching the weight-scale
    // granularity.
    zen_lowp_scale_info_t src_zp_;
    zen_lowp_scale_info_t dst_zp_;
    zen_lowp_scale_info_t wei_zp_;
#endif
};

} // namespace matmul
} // namespace zen
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
