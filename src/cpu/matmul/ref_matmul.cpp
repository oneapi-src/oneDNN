/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <assert.h>
#include <float.h>
#include <math.h>

#include "c_types_map.hpp"
#include "dnnl_thread.hpp"
#include "type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"

#include "ref_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

template <data_type_t src_type, data_type_t weights_type, data_type_t dst_type,
        data_type_t acc_type>
status_t ref_matmul_t<src_type, weights_type, dst_type, acc_type>::execute_ref(
        const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const weights_data_t *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    DEFINE_SCALES_BUFFER(scales);
    DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(weights_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, DNNL_ARG_DST);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    const bool batched = pd()->batched();
    const bool non_default_attrs = !pd()->attr()->has_default_values();
    const bool do_sum = pd()->attr()->post_ops_.contain(primitive_kind::sum, 0)
            && pd()->attr()->post_ops_.entry_[0].sum.scale != 0.f;
    const float sum_scale
            = do_sum ? pd()->attr()->post_ops_.entry_[0].sum.scale : 0.f;

    const dim_t MB = batched ? dst_d.dims()[0] : 1;
    const dim_t M = dst_d.dims()[batched + 0];
    const dim_t N = dst_d.dims()[batched + 1];
    const dim_t K = src_d.dims()[batched + 1];

    // mm kernel
    auto ker = [&](dim_t mb, dim_t m, dim_t n) {
        acc_data_t acc = 0;
        if (batched)
            for (dim_t k = 0; k < K; ++k)
                acc += (src[src_d.off(mb, m, k)] - src_zero_point)
                        * (weights[weights_d.off(mb, k, n)]
                                - weights_zero_point);
        else
            for (dim_t k = 0; k < K; ++k)
                acc += (src[src_d.off(m, k)] - src_zero_point)
                        * (weights[weights_d.off(k, n)] - weights_zero_point);
        return acc;
    };

    // bias section
    const data_type_t bia_dt = pd()->desc()->bias_desc.data_type;
    dim_t bia_stride_mb {}, bia_stride_m {}, bia_stride_n {};
    if (bia_dt != data_type::undef) {
        const auto &bia_strides = bia_d.blocking_desc().strides;
        bia_stride_mb = batched && bia_d.dims()[0] > 1 ? bia_strides[0] : 0;
        bia_stride_m
                = bia_d.dims()[batched + 0] > 1 ? bia_strides[batched + 0] : 0;
        bia_stride_n
                = bia_d.dims()[batched + 1] > 1 ? bia_strides[batched + 1] : 0;
    }
    auto get_bias = [&](dim_t mb, dim_t m, dim_t n) -> float {
        dim_t off = mb * bia_stride_mb + m * bia_stride_m + n * bia_stride_n;
        return math::get_bias(bias, off, bia_dt);
    };

    // output scale section
    const dim_t scale_stride = pd()->attr()->output_scales_.mask_ == 0 ? 0 : 1;

    // computations
    parallel_nd(MB, M, N, [&](dim_t mb, dim_t m, dim_t n) {
        using math::saturate;
        using math::out_round;

        auto &dst_value = dst[batched ? dst_d.off(mb, m, n) : dst_d.off(m, n)];

        acc_data_t acc = ker(mb, m, n);
        if (bias || non_default_attrs) {
            float res = acc;
            if (bias) res += get_bias(mb, m, n);
            res *= scales[scale_stride * n];
            if (do_sum) res = sum_scale * dst_value + res;
            if (eltwise_ker_) res = eltwise_ker_->compute_scalar(res);
            res += (float)dst_zero_point;
            if (utils::one_of(dst_type, data_type::f32, data_type::bf16))
                dst_value = res;
            else
                dst_value = saturate<dst_data_t>(out_round<int32_t>(res));
        } else {
            if (utils::one_of(dst_type, data_type::f32, data_type::bf16))
                dst_value = (dst_data_t)acc;
            else
                dst_value = saturate<dst_data_t>(acc);
        }
    });

    return status::success;
}

using namespace data_type;
template struct ref_matmul_t<f32, f32, f32, f32>;
template struct ref_matmul_t<bf16, bf16, f32, f32>;
template struct ref_matmul_t<bf16, bf16, bf16, f32>;
template struct ref_matmul_t<s8, s8, f32, s32>;
template struct ref_matmul_t<s8, s8, s32, s32>;
template struct ref_matmul_t<s8, s8, s8, s32>;
template struct ref_matmul_t<s8, s8, u8, s32>;
template struct ref_matmul_t<u8, s8, f32, s32>;
template struct ref_matmul_t<u8, s8, s32, s32>;
template struct ref_matmul_t<u8, s8, s8, s32>;
template struct ref_matmul_t<u8, s8, u8, s32>;

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
