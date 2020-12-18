/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <algorithm>
#include <cstdlib>
#include <memory>

#include "common/math_utils.hpp"

#include "cpu/primitive_attr_postops.hpp"
#include "cpu/simple_q10n.hpp"

#if DNNL_X64
#include "cpu/x64/jit_gemm_x8s8s32x_convolution_utils.hpp"
#endif

#include "cpu/gemm_x8s8s32x_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace gemm_x8s8s32x_convolution_utils {

template <typename dst_data_t>
struct ref_pp_ker_t : pp_ker_t {
    ref_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
        : pp_ker_t(pd, jcp) {
        if (jcp.with_eltwise || jcp.with_binary) {
            ref_post_ops_.reset(new ref_post_ops_t(jcp.post_ops));
        }
    }

    using acc_data_t = pp_ker_t::acc_data_t;

    void operator()(void *dst, const acc_data_t *acc, const char *bias,
            const float *scales, float sum_scale, float signed_scale, int g,
            size_t start, size_t end, const int32_t *zp_src,
            const int32_t *zp_dst, const int32_t *zp_src_comp,
            const int32_t *zp_src_pad_comp,
            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
            const exec_ctx_t &ctx, const memory_desc_t &dst_md,
            const single_gemm_conv_chunk_desc_t &chunk_desc) const override;

private:
    std::unique_ptr<ref_post_ops_t> ref_post_ops_;
};

template <typename dst_data_t>
void ref_pp_ker_t<dst_data_t>::operator()(void *void_dst, const acc_data_t *acc,
        const char *bias, const float *scales, float sum_scale,
        float signed_scale, int g, size_t start, size_t end,
        const int32_t *zp_src, const int32_t *zp_dst,
        const int32_t *zp_src_comp, const int32_t *zp_src_pad_comp,
        const void * /* post_ops_binary_rhs_arg_vec */,
        const void * /* dst_orig */, const exec_ctx_t &ctx,
        const memory_desc_t &dst_md,
        const single_gemm_conv_chunk_desc_t &chunk_desc) const {

    if (end <= start) return;

    assert(data_traits<dst_data_t>::data_type == jcp_.dst_data_type);
    dst_data_t *dst = (dst_data_t *)void_dst;

    const auto dv_start = std::div(start, jcp_.oc);
    const auto dv_end = std::div((end - 1), jcp_.oc);
    const size_t first_oc = dv_start.rem;
    const size_t last_oc = dv_end.rem;
    const size_t first_os = dv_start.quot;
    const size_t last_os = dv_end.quot;
    const int32_t zp_dst_val = jcp_.zp.dst_exists ? *zp_dst : 0;

    ref_post_ops_t::args_t args;
    args.ctx = &ctx;
    args.dst_md = &dst_md;

    for (size_t os = first_os; os <= last_os; os++) {
        const size_t start_oc = (os == first_os) ? first_oc : 0;
        const size_t end_oc = (os == last_os) ? last_oc : jcp_.oc - 1;
        for (size_t oc = start_oc; oc <= end_oc; oc++) {
            const size_t acc_off = os * jcp_.oc + oc;
            const size_t dst_off = os * jcp_.dst_os_stride + oc;

            float data = static_cast<float>(acc[acc_off]);
            if (jcp_.zp.src_exists) {
                const auto oc_offset = g * jcp_.oc + oc;
                const int32_t &zp_src_val
                        = jcp_.zp.src_is_common ? *zp_src : zp_src[oc_offset];
                const int32_t &zp_src_comp_val = zp_src_comp[oc_offset];
                data += static_cast<float>(zp_src_val * zp_src_comp_val);
            }

            if (jcp_.signed_input) data *= signed_scale;

            if (jcp_.with_bias)
                data += math::get_bias(
                        bias, g * jcp_.oc + oc, jcp_.bias_data_type);

            data *= scales[(g * jcp_.oc + oc) * jcp_.scale_idx_mult];
            if (jcp_.with_sum) data += sum_scale * dst[dst_off];
            if (jcp_.with_eltwise || jcp_.with_binary) {
                args.l_offset = (g * jcp_.oc + oc) * jcp_.os;
                ref_post_ops_->execute(data, args);
            }

            if (jcp_.zp.dst_exists) data += zp_dst_val;

            dst[dst_off] = qz_a1b0<float, dst_data_t>()(data);
        }
    }
}

// Interface section

pp_ker_t::pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
    : jcp_(jcp) {}

pp_ker_t *pp_ker_t::create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
#if DNNL_X64
    auto *res
            = x64::gemm_x8s8s32x_convolution_utils::jit_pp_ker_create(pd, jcp);
    if (res) return res;
#endif
    switch (pd->dst_md()->data_type) {
        case data_type::f32: return new ref_pp_ker_t<float>(pd, jcp);
        case data_type::s32: return new ref_pp_ker_t<int32_t>(pd, jcp);
        case data_type::s8: return new ref_pp_ker_t<int8_t>(pd, jcp);
        case data_type::u8: return new ref_pp_ker_t<uint8_t>(pd, jcp);
        default: assert(!"unexpected data type");
    }
    return nullptr;
}

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_wrapper *dst_d) {
#if DNNL_X64
    return x64::gemm_x8s8s32x_convolution_utils::post_ops_ok(post_ops, dst_d);
#endif
    return std::all_of(post_ops.entry_.cbegin(), post_ops.entry_.cend(),
            [](const dnnl_post_ops::entry_t &post_op) {
                return post_op.is_eltwise() || post_op.is_sum()
                        || post_op.is_binary();
            });
}

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_t *dst_d) {
    const auto dst_md = memory_desc_wrapper(dst_d);
    return post_ops_ok(post_ops, &dst_md);
}

bool mayiuse_jit_pp_kernel() noexcept {
#if DNNL_X64
    return x64::gemm_x8s8s32x_convolution_utils::mayiuse_jit_pp_kernel();
#else
    return false;
#endif
}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace cpu
} // namespace impl
} // namespace dnnl
