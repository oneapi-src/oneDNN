/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#include <atomic>

#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/gemm/gemm.hpp"

#include "cpu/binary_injector_utils.hpp"
#include "cpu/matmul/gemm_f32_matmul.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/scale_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

using namespace data_type;

status_t gemm_f32_matmul_t::pd_t::init(engine_t *engine) {
    auto check_bias = [&]() -> bool {
        return !with_bias()
                || (weights_md(1)->data_type == f32 && is_bias_1xN());
    };

    auto check_attr_scales = [&]() -> bool {
        bool ok = attr_scales_ok();
        if (!attr()->scales_.get(DNNL_ARG_SRC).has_default_values()
                && !attr()->scales_.get(DNNL_ARG_WEIGHTS).has_default_values()
                && attr()->scales_.get(DNNL_ARG_WEIGHTS).mask_ != 0) {
            // This case requires scratchpad with unknown size
            if (N() == DNNL_RUNTIME_DIM_VAL) ok = false;
        }
        return ok;
    };

    auto check_attr_post_ops = [&]() -> bool {
        using namespace primitive_kind;
        const auto &post_ops = attr()->post_ops_;
        static const bcast_set_t enabled_bcast_strategy {
                broadcasting_strategy_t::scalar,
                broadcasting_strategy_t::per_oc,
                broadcasting_strategy_t::per_oc_spatial,
                broadcasting_strategy_t::per_mb_spatial,
                broadcasting_strategy_t::per_mb_w,
                broadcasting_strategy_t::per_w,
                broadcasting_strategy_t::no_broadcast};
        const bool is_binary_po_per_oc
                = binary_injector_utils::bcast_strategy_present(
                        binary_injector_utils::extract_bcast_strategies(
                                post_ops.entry_, dst_md()),
                        broadcasting_strategy_t::per_oc);
        return cpu::inner_product_utils::post_ops_ok(
                       post_ops, dst_md(), enabled_bcast_strategy)
                && IMPLICATION(is_binary_po_per_oc,
                        gemm_based::check_gemm_binary_per_oc_compatible_formats(
                                *this));
    };

    const bool problem_dt_correct = src_md()->data_type == src_type
            && weights_md()->data_type == weights_type
            && desc()->accum_data_type == acc_type
            && dst_md()->data_type == dst_type;

    VDISPATCH_MATMUL(is_dense_data(), VERBOSE_NONTRIVIAL_STRIDE);
    VDISPATCH_MATMUL(problem_dt_correct, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_MATMUL(attr()->has_default_values(
                             primitive_attr_t::skip_mask_t::scales_runtime
                                     | primitive_attr_t::skip_mask_t::post_ops
                                     | primitive_attr_t::skip_mask_t::sum_dt,
                             dst_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_MATMUL(attr()->post_ops_.check_sum_consistency(dst_type,
                             /* is_int8 */ false),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_MATMUL(check_attr_scales(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_MATMUL(check_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    // Should be followed by `set_default_formats`.
    VDISPATCH_MATMUL(check_attr_post_ops(), VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_MATMUL(gemm_based::check_gemm_compatible_formats(*this),
            "Incompatible format");

    bool po_format_ok = attr_.set_default_formats(dst_md(0)) == status::success;
    VDISPATCH_MATMUL(po_format_ok, VERBOSE_UNSUPPORTED_TAG);

    CHECK(configure_attributes());

    nthr_ = dnnl_get_max_threads();
    gemm_based::book_acc_scratchpad(*this, params_, sizeof(acc_data_t), nthr_);
    auto scratchpad = scratchpad_registry().registrar();
    book_precomputed_scales(scratchpad, attr()->scales_, N());

    return status::success;
}

status_t gemm_f32_matmul_t::pd_t::configure_attributes() {
    matmul_helper_t helper(src_md(), weights_md(), dst_md());

    if (!has_runtime_dims_or_strides())
        params_.use_single_gemm_call_optimization_
                = helper.use_single_gemm_call_optimization(attr()->post_ops_);

    CHECK(params_.pp_attr_.copy_from(*attr()));
    params_.gemm_applies_output_scales_
            = attr()->scales_.get(DNNL_ARG_WEIGHTS).mask_ == 0 && !with_bias();
    if (params_.gemm_applies_output_scales_) {
        params_.pp_attr_.scales_.reset(DNNL_ARG_SRC);
        params_.pp_attr_.scales_.reset(DNNL_ARG_WEIGHTS);
    }

    const auto &po = params_.pp_attr_.post_ops_;
    static constexpr int sum_idx = 0;

    const bool sum_po_via_gemm_beta = po.len() > 0
            && po.contain(primitive_kind::sum, sum_idx)
            && params_.gemm_applies_output_scales_
            && po.entry_[sum_idx].sum.zero_point == 0
            && utils::one_of(po.entry_[sum_idx].sum.dt, dst_md()->data_type,
                    data_type::undef);

    // `C_is_abx` limitation comes from `extended_sgemm`.
    const bool C_is_abx = helper.ldc() >= helper.N()
            && helper.ldc() != DNNL_RUNTIME_DIM_VAL;
    params_.dst_is_acc_ = C_is_abx
            && IMPLICATION(attr()->post_ops_.find(primitive_kind::sum) != -1,
                    sum_po_via_gemm_beta);

    if (sum_po_via_gemm_beta) {
        params_.skip_sum_ = params_.dst_is_acc_;
        params_.gemm_beta_
                = params_.skip_sum_ ? po.entry_[sum_idx].sum.scale : 0.f;
    }

    params_.has_pp_kernel_ = !params_.dst_is_acc_ || with_bias()
            || !params_.pp_attr_.has_default_values();

    return status::success;
}

status_t gemm_f32_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    using namespace binary_injector_utils;
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const weights_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    const auto &po = this->pd()->attr()->post_ops_;
    const auto post_ops_binary_rhs_arg_vec = prepare_binary_args(po, ctx);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());

    const int ndims = pd()->ndims();

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    auto scratchpad = ctx.get_scratchpad_grantor();
    const float *scales = precompute_scales(scratchpad, src_scales, wei_scales,
            dst_d.dims()[ndims - 1], pd()->attr());

    if (src_d.has_zero_dim() || weights_d.has_zero_dim()
            || dst_d.has_zero_dim())
        return status::success;

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int batch_ndims = ndims - 2;
    dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();
    const dim_t batch_without_dim0
            = helper.ndims() > 3 ? batch / dst_d.dims()[0] : 0;
    const dim_t batch_without_dim01
            = helper.ndims() > 4 ? batch_without_dim0 / dst_d.dims()[1] : 1;
    const char transA = helper.transA();
    const char transB = helper.transB();
    const dim_t lda = helper.lda();
    const dim_t ldb = helper.ldb();
    const dim_t ldc = helper.ldc();
    const int nthr = pd()->nthr_;

    const gemm_based::params_t &params = pd()->params();
    const float alpha = params.get_gemm_alpha(scales);
    const float beta = params.gemm_beta_;
    const bool use_single_gemm_call = pd()->has_runtime_dims_or_strides()
            ? helper.use_single_gemm_call_optimization(po)
            : params.use_single_gemm_call_optimization_;
    bool dst_is_acc = params.dst_is_acc_;
    acc_data_t *acc = dst_is_acc
            ? (acc_data_t *)dst
            : ctx.get_scratchpad_grantor().template get<acc_data_t>(
                    memory_tracking::names::key_matmul_dst_in_acc_dt);
    // case: dynamic sizes
    bool need_free_acc = false;
    if (acc == nullptr) {
        const size_t buf_elements = gemm_based::get_scratchpad_num_elements(
                batch, M, N, use_single_gemm_call, nthr);
        acc = (acc_data_t *)malloc(sizeof(acc_data_t) * buf_elements, 64);

        if (acc == nullptr) return status::out_of_memory;
        need_free_acc = true;
    }

    const dim_t acc_ldc = dst_is_acc ? ldc : N;
    const int scale_idx_mult
            = this->pd()->attr()->scales_.get(DNNL_ARG_WEIGHTS).mask_
            == (1 << (ndims - 1));

    std::atomic<status_t> st(status::success);
    if (!use_single_gemm_call) {
        const int src_mask
                = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
        const int wei_mask
                = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);
        const size_t bia_dt_size = !pd()->with_bias()
                ? 0
                : types::data_type_size(pd()->weights_md(1)->data_type);
        const size_t work_amount = (size_t)batch * M * N;
        const size_t work_per_batch = (size_t)M * N;
        const dim_t acc_stride = gemm_based::get_scratchpad_block_elements(
                batch, M, N, use_single_gemm_call, nthr);

        parallel(nthr, [&](int ithr, int nthr) {
            size_t t_work_start {0}, t_work_end {0};
            balance211(work_amount, nthr, ithr, t_work_start, t_work_end);

            dim_t cur_b {0}, cur_m {0}, cur_n {0};
            dims_t s_dims_idx, w_dims_idx, d_dims_idx;
            size_t i_work = t_work_start;
            const bool reuse_acc = acc != (acc_data_t *)dst;
            acc_data_t *curr_acc
                    = reuse_acc ? acc + ithr * acc_stride : nullptr;

            while (i_work < t_work_end) {
                utils::nd_iterator_init(
                        i_work, cur_b, batch, cur_m, M, cur_n, N);

                utils::l_dims_by_l_offset(
                        d_dims_idx, i_work, dst_d.dims(), ndims);

                utils::copy_dims_with_mask(
                        s_dims_idx, d_dims_idx, batch_ndims, src_mask);
                s_dims_idx[ndims - 2] = cur_m;
                s_dims_idx[ndims - 1] = 0; // k idx is always 0

                utils::copy_dims_with_mask(
                        w_dims_idx, d_dims_idx, batch_ndims, wei_mask);
                w_dims_idx[ndims - 2] = 0; // k idx is always 0
                w_dims_idx[ndims - 1] = cur_n;

                const src_data_t *curr_src = src + src_d.off_v(s_dims_idx);
                const weights_data_t *curr_weights
                        = weights + weights_d.off_v(w_dims_idx);
                const dim_t dst_off = dst_d.off_v(d_dims_idx);
                dst_data_t *curr_dst = dst + dst_off;
                if (!reuse_acc) curr_acc = acc + dst_off;
                dim_t gemm_M {0}, gemm_N {0};

                size_t matrix_offset;
                const size_t rem_work = t_work_end - i_work;
                if (rem_work >= work_per_batch && cur_m == 0 && cur_n == 0) {
                    // parallel over batch
                    gemm_M = M;
                    gemm_N = N;
                    matrix_offset = 0;
                } else if (rem_work >= (size_t)N && cur_n == 0) {
                    // parallel over M
                    gemm_M = nstl::min(
                            (size_t)(M - cur_m), (size_t)(rem_work / N));
                    gemm_N = N;
                    matrix_offset = cur_n + cur_m * N;
                } else {
                    // parallel over N
                    gemm_M = 1;
                    gemm_N = nstl::min((size_t)(N - cur_n), rem_work);
                    matrix_offset = cur_n + cur_m * N;
                }

                status_t st_thr = extended_sgemm(&transB, &transA, &gemm_N,
                        &gemm_M, &K, &alpha, curr_weights, &ldb, curr_src, &lda,
                        &beta, curr_acc, &acc_ldc, nullptr, false);
                if (st_thr != status::success) {
                    st = st_thr;
                    return;
                }

                if (params.has_pp_kernel_) {
                    const float *pp_scales
                            = params.get_post_processing_scales(scales);
                    const size_t dst_logical_off = i_work;
                    const size_t dim1_off = helper.ndims() > 3
                            ? ((cur_b % batch_without_dim0)
                                    / batch_without_dim01)
                            : cur_m;

                    // offset for case with post-op broadcast_channel
                    const size_t matrix_per_first_batch_off = helper.ndims() > 3
                            ? M * N * (cur_b / batch_without_dim0)
                                    + matrix_offset
                            : 0;
                    const ptrdiff_t oc_off = i_work % N;
                    (*pp_kernel_)(curr_dst, curr_acc,
                            bias + oc_off * bia_dt_size,
                            pp_scales + oc_off * scale_idx_mult, dst_scales[0],
                            0, dst_logical_off, dim1_off, gemm_M * gemm_N,
                            static_cast<size_t>(N), ldc, nullptr,
                            post_ops_binary_rhs_arg_vec.data(), dst,
                            matrix_per_first_batch_off, ctx, *pd()->dst_md());
                }
                i_work += gemm_M * gemm_N;
            }
        });
    } else {
        // collapse batch into M, if weights batch dimensions are broadcasted.
        M = batch * M;

        st = extended_sgemm(&transB, &transA, &N, &M, &K, &alpha, weights, &ldb,
                src, &lda, &beta, acc, &acc_ldc, nullptr, false);

        if (st == status::success && params.has_pp_kernel_) {
            const bool force_sequential = pp_kernel_->sequential_kernel();
            const float *pp_scales = params.get_post_processing_scales(scales);
            parallel(force_sequential ? 1 : nthr, [&](int ithr, int nthr) {
                size_t start {}, end {};
                balance211((size_t)(M * N), nthr, ithr, start, end);
                const size_t dst_logical_off = start;
                const size_t dst_start_row_idx = start % N;
                (*pp_kernel_)(dst, acc, bias, pp_scales, dst_scales[0], start,
                        dst_logical_off, dst_start_row_idx, end, (size_t)N, ldc,
                        nullptr, post_ops_binary_rhs_arg_vec.data(), dst, 0,
                        ctx, *pd()->dst_md());
            });
        }
    }

    if (need_free_acc) free(acc);

    return st;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
