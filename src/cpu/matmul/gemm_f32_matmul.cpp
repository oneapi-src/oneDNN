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
#include "utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "gemm_f32_matmul.hpp"

#include "gemm/gemm.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace data_type;

status_t gemm_f32_matmul_t::pd_t::init() {
    auto check_bias = [&]() -> bool {
        if (!with_bias()) return true;

        const auto &bia_md = *weights_md(1);
        bool ok = bia_md.data_type == f32 && bia_md.dims[0] == 1
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
        auto check_sum = [&](int idx) -> bool {
            // Implementation scales after applying beta * dst, hence this only
            // works if output scale = 1.
            return p.contain(sum, idx) && attr()->output_scales_.mask_ == 0
                    && attr()->output_scales_.scales_[0] == 1.f;
        };
        switch (p.len_) {
            case 0: return true;
            case 1: return check_sum(0) || p.contain(eltwise, 0);
            case 2: return check_sum(0) && p.contain(eltwise, 1);
            default: return false;
        }
    };

    auto can_use_gemm = [&]() -> bool {
        return IMPLICATION(src_type == f32, mayiuse(sse41))
                && IMPLICATION(utils::one_of(src_type, s8, bf16),
                        mayiuse(avx512_core));
    };

    bool ok = src_md()->data_type == src_type
            && weights_md()->data_type == weights_type
            && desc()->accum_data_type == acc_type
            && dst_md()->data_type == dst_type && can_use_gemm() && batch() == 1
            && check_bias()
            && attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::oscale_runtime
                    | primitive_attr_t::skip_mask_t::post_ops)
            && check_attr_oscale() && check_attr_post_ops()
            && set_default_formats();
    return ok ? status::success : status::unimplemented;
}

status_t gemm_f32_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const weights_data_t *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    DEFINE_SCALES_BUFFER(scales);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    const auto &dst_bd = dst_d.blocking_desc();

    const bool batched = pd()->batched();

    const dim_t M = dst_d.dims()[batched + 0];
    const dim_t N = dst_d.dims()[batched + 1];
    const dim_t K = src_d.dims()[batched + 1];

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

        int sum_idx = pd()->attr()->post_ops_.find(primitive_kind::sum);
        const float alpha = 1;
        const float beta = sum_idx >= 0
                ? pd()->attr()->post_ops_.entry_[sum_idx].sum.scale
                : 0;

        status_t status = extended_sgemm(transB, transA, &N_s32, &M_s32, &K_s32,
                &alpha, weights, &ldb, src, &lda, &beta, dst, &ldc, nullptr,
                false);
        assert(status == status::success);
        if (status != status::success) return status;
    }

    const bool postops_in_matmul = pd()->with_bias()
            || pd()->attr()->post_ops_.find(primitive_kind::eltwise) >= 0
            || !pd()->attr()->output_scales_.has_default_values();

    if (postops_in_matmul) {
        const bool force_sequential = pp_kernel_->sequential_kernel();
        parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
            size_t start {}, end {};
            balance211((size_t)(M * N), nthr, ithr, start, end);
            (*pp_kernel_)(dst, dst, bias, scales, start, end, (size_t)N);
        });
    }

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
