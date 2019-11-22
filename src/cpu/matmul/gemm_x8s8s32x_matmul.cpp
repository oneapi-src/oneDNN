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
#include "memory_tracking.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "gemm_x8s8s32x_matmul.hpp"

#include "gemm/gemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace data_type;

template <data_type_t src_type, data_type_t weights_type, data_type_t dst_type>
status_t
gemm_x8s8s32x_matmul_t<src_type, weights_type, dst_type>::pd_t::init() {
    using namespace utils;

    auto check_bias = [&]() -> bool {
        if (!with_bias()) return true;

        const auto &bia_md = *weights_md(1);
        bool ok = one_of(weights_md(1)->data_type, f32, s32, s8, u8)
                && bia_md.dims[0] == 1
                && IMPLICATION(batched(), bia_md.dims[1] == 1)
                && bia_md.dims[batched() + 1] == dst_md()->dims[batched() + 1];

        return ok;
    };

    auto check_attr_oscale = [&]() -> bool {
        const auto &oscale = attr()->output_scales_;
        return oscale.mask_ == 0
                || (oscale.mask_ == (1 << 1) && batched() == false);
    };

    auto check_attr_post_ops = [&]() -> bool {
        using namespace primitive_kind;
        const auto &p = attr()->post_ops_;
        switch (p.len_) {
            case 0: return true;
            case 1: return p.contain(sum, 0) || p.contain(eltwise, 0);
            case 2: return p.contain(sum, 0) && p.contain(eltwise, 1);
            default: return false;
        }
    };

    bool ok = src_md()->data_type == src_type
            && weights_md()->data_type == weights_type
            && desc()->accum_data_type == acc_type
            && dst_md()->data_type == dst_type && batch() == 1 && check_bias()
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::zero_points_runtime
                    | primitive_attr_t::skip_mask_t::post_ops)
            && check_attr_oscale() && check_attr_post_ops()
            && set_default_formats();
    if (!ok) return status::unimplemented;

    bool do_sum = attr()->post_ops_.find(primitive_kind::sum) >= 0;
    dst_is_acc_ = utils::one_of(dst_type, s32, f32) && !do_sum;

    if (!dst_is_acc_ && !one_of(DNNL_RUNTIME_DIM_VAL, batch(), M(), N())) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
                sizeof(acc_data_t) * batch() * M() * N());
    }

    return status::success;
}

template <data_type_t src_type, data_type_t weights_type, data_type_t dst_type>
status_t gemm_x8s8s32x_matmul_t<src_type, weights_type, dst_type>::execute_ref(
        const exec_ctx_t &ctx) const {
    using math::get_bias;
    using math::saturate;

    const auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const weights_data_t *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    DEFINE_SCALES_BUFFER(scales);
    DEFINE_ZERO_POINT_VALUE(src_zero_point, DNNL_ARG_SRC);
    DEFINE_ZERO_POINT_VALUE(weights_zero_point, DNNL_ARG_WEIGHTS);
    DEFINE_ZERO_POINT_VALUE(dst_zero_point, DNNL_ARG_DST);

    src_data_t gemm_off_a = (src_data_t)src_zero_point;
    weights_data_t gemm_off_b = (weights_data_t)weights_zero_point;
    bool post_process_src_and_weights_zero_points = false;
    if (gemm_off_a != src_zero_point || gemm_off_b != weights_zero_point) {
        post_process_src_and_weights_zero_points = true;
        gemm_off_a = gemm_off_b = 0;
    }
    const float dst_zero_point_f32 = (float)dst_zero_point;

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC);
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS);
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST);

    acc_data_t *acc = pd()->dst_is_acc_
            ? (acc_data_t *)dst
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    memory_tracking::names::key_matmul_dst_in_acc_dt);

    const auto &dst_bd = dst_d.blocking_desc();

    const bool batched = pd()->batched();

    const dim_t batch = batched ? dst_d.dims()[0] : 1;
    const dim_t M = dst_d.dims()[batched + 0];
    const dim_t N = dst_d.dims()[batched + 1];
    const dim_t K = src_d.dims()[batched + 1];

    // case: dynamic sizes
    bool need_free_acc = false;
    if (acc == nullptr) {
        acc = (acc_data_t *)malloc(sizeof(acc_data_t) * batch * M * N, 64);
        need_free_acc = true;
    }

    // gemm section
    {
        const auto &src_strides = &src_d.blocking_desc().strides[batched];
        const auto &weights_strides
                = &weights_d.blocking_desc().strides[batched];

        const char *transA
                = src_strides[1] == 1 && src_d.dims()[batched + 0] > 1 ? "N"
                                                                       : "T";
        const char *transB
                = weights_strides[1] == 1 && weights_d.dims()[batched + 0] > 1
                ? "N"
                : "T";

        const int M_s32 = (int)M;
        const int N_s32 = (int)N;
        const int K_s32 = (int)K;

        const int lda = (int)src_strides[*transA == 'N' ? 0 : 1];
        const int ldb = (int)weights_strides[*transB == 'N' ? 0 : 1];
        const int ldc = (int)dst_bd.strides[batched + 0];

        const float onef = 1.0, zerof = 0.0;
        const int32_t gemm_off_c = 0;

        status_t status = gemm_s8x8s32(transB, transA, "F", &N_s32, &M_s32,
                &K_s32, &onef, weights, &ldb, &gemm_off_b, src, &lda,
                &gemm_off_a, &zerof, acc, &ldc, &gemm_off_c);
        assert(status == status::success);
        if (status != status::success) {
            if (need_free_acc) free(acc);
            return status;
        }

        // if igemm cannot handle src and weights zero points
        if (post_process_src_and_weights_zero_points) {
            std::vector<acc_data_t> src_compensation(M, 0);
            std::vector<acc_data_t> weights_compensation(N, 0);

            if (weights_zero_point) {
                for_(dim_t m = 0; m < M; ++m)
                for (dim_t k = 0; k < K; ++k)
                    src_compensation[m]
                            += src[src_strides[0] * m + src_strides[1] * k];
            }

            if (src_zero_point) {
                for_(dim_t k = 0; k < K; ++k)
                for (dim_t n = 0; n < N; ++n)
                    weights_compensation[n] += weights[weights_strides[0] * k
                            + weights_strides[1] * n];
            }

            for_(dim_t m = 0; m < M; ++m)
            for (dim_t n = 0; n < N; ++n)
                acc[m * ldc + n] += 0 - src_zero_point * weights_compensation[n]
                        - weights_zero_point * src_compensation[m]
                        + src_zero_point * weights_zero_point * (int)K;
        }
    }

    const bool postops_in_matmul = pd()->with_bias()
            || !pd()->attr()->has_default_values() || !pd()->dst_is_acc_
            || dst_type != s32 || dst_zero_point_f32 != 0.f;

    if (postops_in_matmul) {
        const bool force_sequential = pp_kernel_->sequential_kernel();
        parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
            size_t start {}, end {};
            balance211((size_t)(M * N), nthr, ithr, start, end);
            (*pp_kernel_)(dst, acc, bias, scales, start, end, (size_t)N,
                    &dst_zero_point_f32);
        });
    }

    if (need_free_acc) free(acc);

    return status::success;
}

template struct gemm_x8s8s32x_matmul_t<s8, s8, f32>;
template struct gemm_x8s8s32x_matmul_t<s8, s8, s32>;
template struct gemm_x8s8s32x_matmul_t<s8, s8, s8>;
template struct gemm_x8s8s32x_matmul_t<s8, s8, u8>;
template struct gemm_x8s8s32x_matmul_t<u8, s8, f32>;
template struct gemm_x8s8s32x_matmul_t<u8, s8, s32>;
template struct gemm_x8s8s32x_matmul_t<u8, s8, s8>;
template struct gemm_x8s8s32x_matmul_t<u8, s8, u8>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
