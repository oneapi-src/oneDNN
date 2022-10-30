/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef CPU_X64_JIT_BRGEMM_CONV_BWD_STRIDED_HPP
#define CPU_X64_JIT_BRGEMM_CONV_BWD_STRIDED_HPP

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_trans_kernel.hpp"
#include "cpu/x64/jit_brgemm_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa, bool enable_postops = false>
struct brgemm_convolution_bwd_strided_t : public primitive_t {

    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::hint_class *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}

        ~pd_t() = default;

        // ------- DECLARE_COMMON_PD_t -------
        pd_t *clone() const override {
            auto new_pd = utils::make_unique<pd_t>(*this);
            if (!new_pd->is_initialized()) return nullptr;
            new_pd->brgs_.resize(brgs_sz_);
            for (int i = 0; i < brgs_sz_; i++) {
                new_pd->brgs_[i] = brgs_[i];
                new_pd->bd_masks[i] = bd_masks[i];
            }
            return new_pd.release();
        }

        status_t create_primitive(
                std::pair<std::shared_ptr<primitive_t>, bool> &primitive,
                engine_t *engine,
                const cache_blob_t &cache_blob) const override {
            return primitive_t::create_primitive_common<
                    brgemm_convolution_bwd_strided_t, pd_t>(
                    primitive, this, engine, false, cache_blob);
        }

        const char *name() const override {
            return JIT_IMPL_NAME_HELPER("brgconv_strided:", isa, "");
        }
        // ---------------------------------

        status_t init(engine_t *engine);

        int brgs_sz_;
        std::vector<std::shared_ptr<brgemm_t>> brgs_;
        std::vector<std::shared_ptr<std::vector<char>>> bd_masks;
        jit_brgemm_conv_conf_t jcp_;
        // batch sizes info for unrolled kernels
        int bs_c, first_bs;
        std::vector<int> batchsizes;
        int get_brg_idx(int bs, int m, bool do_initialization, bool is_N_tail,
                bool is_K_tail) const {
            auto bs_idx = 0;
            return (((m * bs_c + bs_idx) * 2
                            + static_cast<int>(do_initialization))
                                   * 2
                           + static_cast<int>(is_N_tail))
                    * 2
                    + static_cast<int>(is_K_tail);
        }
    };

    brgemm_convolution_bwd_strided_t(const pd_t *apd)
        : primitive_t(apd), bias_d(pd()->weights_md(1)) {}

    ~brgemm_convolution_bwd_strided_t() = default;

    status_t execute(const exec_ctx_t &ctx) const override;

protected:
    status_t init(engine_t *engine) override;

private:
    struct S_t {
        char a[AMX_PALETTE_SIZE];
    };

    //  brgemm convolution execution context
    struct brgemm_bwd_exec_ctx_t {
        brgemm_bwd_exec_ctx_t(const exec_ctx_t &ctx, const pd_t *pd)
            : diff_dst(CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST))
            , weights(CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS))
            , bias(CTX_IN_MEM(const char *, DNNL_ARG_BIAS))
            , dst(CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC))
            , post_ops_binary_rhs_arg_vec(binary_injector::prepare_binary_args(
                      pd->attr()->post_ops_, ctx)) {}
        const char *const __restrict diff_dst;
        const char *const __restrict weights;
        const char *const __restrict bias;
        char *const __restrict dst;
        const std::vector<const void *> post_ops_binary_rhs_arg_vec;
    };

    struct brgemm_bwd_thread_ctx_t {
        brgemm_bwd_thread_ctx_t(brgemm_bwd_exec_ctx_t &brgemm_ctx_, int ithr_,
                brgemm_batch_element_t *__restrict brg_batch_, char *c_buffer_,
                char *wsp_tile_)
            : brgemm_ctx(brgemm_ctx_)
            , ithr(ithr_)
            , brg_batch(brg_batch_)
            , c_buffer(c_buffer_)
            , wsp_tile(wsp_tile_) {}

        brgemm_bwd_exec_ctx_t &brgemm_ctx;
        int ithr;
        brgemm_batch_element_t *__restrict brg_batch;
        char *c_buffer;
        char *wsp_tile;
        S_t cur_palette;
        int g, n, icb;
        int id, idb, ih, ihb, iwb;
        int occ;
        int sw;
        const float *oscales {nullptr};
    };

    void ker_trans(brgemm_bwd_thread_ctx_t &btc, char *inp_buffer) const;

    void call_brgemm_kernel(brgemm_bwd_thread_ctx_t &btc, int brg_idx,
            int batch_size, char *ptr_C, char *ptr_D, const char *bias_w,
            int g_ic, bool do_postops, const void *binary_post_ops_rhs,
            int32_t src_zp_vals, int32_t *src_zp_ptr, int32_t *dst_zp_ptr,
            int32_t *s8s8_comp, bool do_only_comp,
            bool is_first_call_postops) const;

    void maybe_trans_inp(int ithr, const char *__restrict input,
            char *__restrict inp_buffer, uint8_t *__restrict inp_buffer_mask,
            int g, int n, int icc, int odb, int ohb, int owb, int last_g,
            int last_n, int last_icc, int last_odb, int last_ohb,
            int last_owb) const;

    status_t add_brg_kernel(int bs, int M, int i_N, int i_K, int i_init);
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    std::vector<std::unique_ptr<brgemm_kernel_t>> brg_kernels_;
    std::unique_ptr<jit_avx512_core_brgemm_conv_bwd_trans_kernel::
                    jit_avx512_core_brgemm_conv_bwd_trans_kernel_t>
            copy_to_pbuffer_;
    std::vector<S_t> brg_kernel_palettes_;

    size_t acc_dsz, bia_dsz, src_dsz, wei_dsz, dst_dsz;

    const memory_desc_wrapper bias_d;

    int KD, KH, KW, EXT_KD, EXT_KH, EXT_KW, KS, KD_BLOCK, KH_BLOCK, KW_BLOCK,
            KD_BLOCK_PAD, KH_BLOCK_PAD, ID, IH, IW, ODP, OHP, OWP, OD, OH, OW,
            SD, SH, SW, FP, TP, LP, DD, DH, DW;
    dim_t src_w_sz, src_h_sz, src_d_sz, dst_w_sz, dst_h_sz, dst_d_sz, wei_oc_sz,
            wei_kw_sz, wei_kh_sz, wei_kd_sz, wei_icb_sz;
    dim_t pbuf_w_sz, pbuf_h_sz, pbuf_d_sz;

    int oc_chunks;
    bool need_postwork;
    bool is_amx;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
