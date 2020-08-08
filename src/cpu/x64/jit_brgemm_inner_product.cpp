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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_brgemm_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::data_type;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace nstl;

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void brgemm_inner_product_fwd_t<src_type, wei_type, dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_ = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights_ = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst_ = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);

    auto src = const_cast<src_data_t *>(src_);
    auto weights = const_cast<wei_data_t *>(weights_);
    auto dst = const_cast<dst_data_t *>(dst_);

    memory_tracking::grantor_t scratchpad = ctx.get_scratchpad_grantor();
    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const float *oscales = pd()->attr()->output_scales_.scales_;

    const auto &jbgp = pd()->jbgp_;

    src_data_t **addr_A_global = scratchpad.template get<src_data_t *>(
            key_brgemm_primitive_addr_a);
    wei_data_t **addr_B_global = scratchpad.template get<wei_data_t *>(
            key_brgemm_primitive_addr_b);
    char *c_buffer_global = (jbgp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;

    int ic_chunks = jbgp.nb_ic / jbgp.nb_ic_blocking;
    bool are_post_ops_applicable = one_of(true, jbgp.with_sum, jbgp.with_bias,
            jbgp.with_scales, jbgp.with_eltwise, jbgp.acc_dt != jbgp.dst_dt);

    const auto ker = [&](const int ithr, int n, int ocb, int icc) {
        src_data_t **addr_A = addr_A_global + ithr * 16 * jbgp.gemm_batch_size;
        wei_data_t **addr_B = addr_B_global + ithr * 16 * jbgp.gemm_batch_size;

        char *c_buffer = (jbgp.use_buffer) ? c_buffer_global
                        + ithr * types::data_type_size(jbgp.acc_dt) * jbgp.LDC
                                * jbgp.M
                                           : nullptr;

        int oc = ocb * jbgp.oc_block;
        int icb = icc * jbgp.nb_ic_blocking;
        int ic = icb * jbgp.ic_block;

        bool kernel_init = (icc == 0);

        bool is_os_tail = (jbgp.mb - n < jbgp.os_block);
        bool is_oc_tail = (jbgp.oc - oc < jbgp.oc_block);
        bool is_ic_tail = (jbgp.ic - ic < jbgp.ic_block * jbgp.nb_ic_blocking);
        auto nb_ic_b = is_ic_tail ? (jbgp.ic - ic) / jbgp.ic_block
                                  : jbgp.nb_ic_blocking;

        auto brg_kernel = brg_kernels[pd()->get_brg_kernel_idx(kernel_init,
                                              is_os_tail, is_oc_tail, false)]
                                  .get();

        if (nb_ic_b > 0 && brg_kernel != nullptr) {
            for (int ic_block = 0; ic_block < nb_ic_b; ic_block++) {
                addr_A[ic_block]
                        = src + src_d.blk_off(n, ic + ic_block * jbgp.ic_block);
                addr_B[ic_block]
                        = weights + weights_d.blk_off(ocb, icb + ic_block);
            }

            if (are_post_ops_applicable && icc == ic_chunks - 1
                    && !is_ic_tail) {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)dst
                                + sizeof(dst_data_t) * dst_d.blk_off(n, oc);
                auto ptr_D = dst + dst_d.blk_off(n, oc);
                auto bias_w
                        = jbgp.with_bias ? bias + bia_dt_size * oc : nullptr;
                brgemm_kernel_execute_postops(brg_kernel, nb_ic_b,
                        (void **)addr_A, (void **)addr_B, (void *)ptr_C,
                        (void *)ptr_D, (void *)bias_w,
                        &oscales[jbgp.is_oc_scale * oc]);
            } else {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)dst
                                + sizeof(dst_data_t) * dst_d.blk_off(n, oc);
                brgemm_kernel_execute(brg_kernel, nb_ic_b, (void **)addr_A,
                        (void **)addr_B, (void *)ptr_C);
            }
        }

        if (is_ic_tail) {
            int ic_block = jbgp.nb_ic_blocking - 1;
            addr_A[0] = src + src_d.blk_off(n, ic + ic_block * jbgp.ic_block);
            addr_B[0] = weights + weights_d.blk_off(ocb, icb + ic_block);

            auto use_init_ker = (kernel_init && nb_ic_b == 0);
            auto brg_kernel_ic_tail
                    = brg_kernels[pd()->get_brg_kernel_idx(use_init_ker,
                                          is_os_tail, is_oc_tail, true)]
                              .get();

            if (are_post_ops_applicable && icc == ic_chunks - 1) {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)dst
                                + sizeof(dst_data_t) * dst_d.blk_off(n, oc);
                auto ptr_D = dst + dst_d.blk_off(n, oc);
                auto bias_w
                        = jbgp.with_bias ? bias + bia_dt_size * oc : nullptr;
                brgemm_kernel_execute_postops(brg_kernel_ic_tail, 1,
                        (void **)addr_A, (void **)addr_B, (void *)ptr_C,
                        (void *)ptr_D, (void *)bias_w,
                        &oscales[jbgp.is_oc_scale * oc]);
            } else {
                char *ptr_C = (jbgp.use_buffer) ? c_buffer
                                                : (char *)dst
                                + sizeof(dst_data_t) * dst_d.blk_off(n, oc);
                brgemm_kernel_execute(brg_kernel_ic_tail, 1, (void **)addr_A,
                        (void **)addr_B, (void *)ptr_C);
            }
        }
    };

    int os_chunks = jbgp.nb_os / jbgp.nb_os_blocking;
    int work_amount = jbgp.nb_oc * os_chunks;

    parallel(0, [&](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int ocb {0}, oss {0};

        nd_iterator_init(start, oss, os_chunks, ocb, jbgp.nb_oc);
        while (start < end) {
            for_(int osb = 0; osb < jbgp.nb_os_blocking; osb++)
            for (int icc = 0; icc < ic_chunks; icc++) {
                int n = (oss * jbgp.nb_os_blocking + osb) * jbgp.os_block;
                ker(ithr, n, ocb, icc);
            }
            ++start;
            nd_iterator_step(oss, os_chunks, ocb, jbgp.nb_oc);
        }
    });
}

template struct brgemm_inner_product_fwd_t<f32>;
template struct brgemm_inner_product_fwd_t<bf16>;
template struct brgemm_inner_product_fwd_t<bf16, bf16, f32>;
template struct brgemm_inner_product_fwd_t<u8, s8, f32>;
template struct brgemm_inner_product_fwd_t<u8, s8, s32>;
template struct brgemm_inner_product_fwd_t<u8, s8, u8>;
template struct brgemm_inner_product_fwd_t<u8, s8, s8>;
template struct brgemm_inner_product_fwd_t<s8, s8, f32>;
template struct brgemm_inner_product_fwd_t<s8, s8, s32>;
template struct brgemm_inner_product_fwd_t<s8, s8, u8>;
template struct brgemm_inner_product_fwd_t<s8, s8, s8>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
