/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_1X1_CONV_HPP
#define CPU_X64_JIT_BRGEMM_1X1_CONV_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_containers.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_brgemm_conv_trans_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"
#include "cpu/x64/jit_brgemm_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct brgemm_1x1_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(adesc, attr, hint_fwd_pd)
            , with_sum(false)
            , sum_scale(0) {}

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("brgconv_1x1:", isa, ""),
                brgemm_1x1_convolution_fwd_t);

        status_t init(engine_t *engine);

        std::shared_ptr<brgemm_containers::brgemm_desc_container_t> brgs_;
        bool with_sum;
        float sum_scale;

        bool need_postwork;
        int ic_chunks;

        jit_brgemm_conv_conf_t jcp_;

    protected:
        bool arg_scales_ok() const {
            std::vector<int> supported_args
                    = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
            return attr_scales_ok(supported_args);
        }
        bool zero_points_ok() const {
            // Only common zero points are supported -> mask should only be 0
            int mask_src = 0, mask_dst = 0;
            attr()->zero_points_.get(DNNL_ARG_SRC, &mask_src);
            attr()->zero_points_.get(DNNL_ARG_DST, &mask_dst);
            return attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS)
                    && mask_src == 0 && mask_dst == 0;
        }
    };

    brgemm_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), bias_d(pd()->weights_md(1)) {}

    ~brgemm_1x1_convolution_fwd_t() {}

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward_all(ctx);

        if (pd()->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad(ctx);

        return status::success;
    }

protected:
    status_t init(engine_t *engine) override;

private:
    //  brgemm convolution execution context
    struct brgemm_exec_ctx_t {
        brgemm_exec_ctx_t(const exec_ctx_t &ctx, const pd_t *pd)
            : src(CTX_IN_MEM(const char *, DNNL_ARG_SRC))
            , weights(CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS))
            , bias(CTX_IN_MEM(const char *, DNNL_ARG_BIAS))
            , dst(CTX_OUT_MEM(char *, DNNL_ARG_DST))
            , post_ops_binary_rhs_arg_vec(binary_injector::prepare_binary_args(
                      pd->attr()->post_ops_, ctx))
            , wsp_tile(ctx.get_scratchpad_grantor().template get<char>(
                      memory_tracking::names::key_conv_amx_tile_buffer)) {}
        const char *const __restrict src;
        const char *const __restrict weights;
        const char *const __restrict bias;
        char *const __restrict dst;
        const std::vector<const void *> post_ops_binary_rhs_arg_vec;
        char *const wsp_tile;
    };

    void maybe_rtus(int ithr, const char *__restrict src,
            char *__restrict inp_buffer, uint8_t *__restrict inp_buffer_mask,
            int g, int n, int icc, int od, int oh, int ow) const;
    void exec_ker(const brgemm_exec_ctx_t &brgemm_ctx, int ithr,
            brgemm_batch_element_t *const __restrict brg_batch,
            char *const c_buffer, const char *inp_buffer, int g, int n, int ocb,
            int od, int oh, int ow, int icc, int *last_brg_idx,
            const float *oscales, int32_t src_zp_vals, int32_t *src_zp_comp,
            int32_t *dst_zp_vals, int32_t *s8s8_compensation,
            const float *dst_scales) const;
    status_t execute_forward_all(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    static int get_brg_idx(bool do_initialization, int is_M_tail,
            bool is_N_tail, bool is_K_tail) {
        return (((int)do_initialization * 2 + (int)is_M_tail) * 2
                       + (int)is_N_tail)
                * 2
                + (int)is_K_tail;
    }

    static int get_ker_po_idx(int is_M_tail, bool is_N_tail) {
        return (int)is_M_tail * 2 + (int)is_N_tail;
    }

    brgemm_containers::brgemm_kernel_container_t brg_kernels_ {16};
    brgemm_containers::brgemm_palette_container_t brgemm_palettes_ {16};

    std::unique_ptr<jit_avx512_core_brgemm_conv_trans_kernel::
                    jit_avx512_core_brgemm_conv_rtus_kernel_t>
            rtus_kernel_;

    const memory_desc_wrapper bias_d;

    int ID, IH, IW, OD, OH, OW, SD, SH, SW;
    size_t bia_dsz, acc_dsz, src_dsz, wei_dsz;
    // const variables used for address calculations
    dim_t src_w_sz, src_h_sz, src_d_sz, dst_w_sz, dst_h_sz, dst_d_sz;
    dim_t wei_g_stride, wei_ic_stride, wei_ocb_stride;
    dim_t wei_kw_stride, wei_kh_stride, wei_kd_stride;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
