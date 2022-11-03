/*******************************************************************************
* Copyright 2018-2022 Intel Corporation
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

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/gemm/gemm.hpp"
#include "cpu/gemm_x8s8s32x_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace math;
using namespace format_tag;
using namespace memory_tracking::names;

status_t gemm_x8s8s32x_inner_product_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector_utils::prepare_binary_args(
                    this->pd()->attr()->post_ops_, ctx);

    const dim_t MB = pd()->MB();
    const dim_t OC = pd()->OC();
    const dim_t IC = pd()->IC();

    const auto &wmd = *pd()->weights_md();
    const auto &smd = *pd()->src_md();
    bool wei_tr = wmd.format_desc.blocking.strides[0] != 1;
    // check if MB is the leading dimension
    bool src_tr = smd.format_desc.blocking.strides[0] == 1 && IC > 1;

    const dim_t M = OC;
    const dim_t N = MB;
    const dim_t K = pd()->IC_total_padded();
    const int8_t off_a = 0;
    const int32_t off_c = 0;

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();
    const float *scales = precompute_scales(
            scratchpad, src_scales, wei_scales, OC, pd()->attr());

    int32_t *acc = pd()->dst_is_acc_
            ? (int32_t *)dst
            : ctx.get_scratchpad_grantor().template get<int32_t>(
                    key_iprod_int_dat_in_acc_dt);

    const float onef = 1.0, zerof = 0.0;

    if (smd.data_type == data_type::s8) {
        const int8_t off_b = 0;
        const int8_t *src_ = reinterpret_cast<const int8_t *>(src);
        CHECK(gemm_s8x8s32(wei_tr ? "T" : "N", src_tr ? "T" : "N", "F", &M, &N,
                &K, &onef, weights, wei_tr ? &K : &M, &off_a, src_,
                src_tr ? &N : &K, &off_b, &zerof, acc, &M, &off_c));
    } else if (smd.data_type == data_type::u8) {
        const uint8_t off_b = 0;
        const uint8_t *src_ = reinterpret_cast<const uint8_t *>(src);
        CHECK(gemm_s8x8s32(wei_tr ? "T" : "N", src_tr ? "T" : "N", "F", &M, &N,
                &K, &onef, weights, wei_tr ? &K : &M, &off_a, src_,
                src_tr ? &N : &K, &off_b, &zerof, acc, &M, &off_c));
    } else {
        assert(!"unsupported data type!");
    }

    if (!pd()->attr()->has_default_values()
            || pd()->dst_md()->data_type != data_type::s32
            || pd()->with_bias()) {
        const bool force_sequential
                = pp_kernel_->sequential_kernel() || MB * OC < 2000;
        parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)(OC * MB), nthr, ithr, start, end);
            const size_t dst_logical_off = start;
            const size_t dim1_off = start % OC;
            (*pp_kernel_)(dst, acc, bias, scales, dst_scales[0], start,
                    dst_logical_off, dim1_off, end, 0, 0, nullptr,
                    post_ops_binary_rhs_arg_vec.data(), dst, 0, ctx,
                    *pd()->dst_md());
        });
    }

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
