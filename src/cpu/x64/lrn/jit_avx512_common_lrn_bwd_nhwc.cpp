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
#include "cpu/x64/lrn/jit_avx512_common_lrn_bwd_nhwc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

using acc_data_t = float;

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_bwd_nhwc_t<
        d_type>::jit_avx512_common_lrn_kernel_bwd_nhwc_t(unsigned C,
        float alpha, float beta, void *code_ptr, size_t code_size)
    : jit_avx512_common_lrn_kernel_bwd_t<d_type>(
            alpha, beta, code_ptr, code_size) {

    this->preamble();
    this->set_up_ker_params();
    this->execute_compute_loop(C);
    this->postamble();
    this->ker = reinterpret_cast<decltype(this->ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::set_up_ker_params() {
#define GET_OFF(field) \
    offsetof(typename jit_avx512_common_lrn_kernel_bwd_t< \
                     d_type>::jit_args_bwd_t, \
            field)
    this->mov(this->src_, ptr[this->param_ + GET_OFF(src)]);
    this->mov(this->diffdst_, ptr[this->param_ + GET_OFF(diff_dst)]);
    this->mov(this->workspace0_, ptr[this->param_ + GET_OFF(ws0)]);
    this->mov(this->workspace1_, ptr[this->param_ + GET_OFF(ws1)]);
    this->mov(this->diffsrc_, ptr[this->param_ + GET_OFF(diff_src)]);

    this->mov(this->mask_, ptr[this->param_ + GET_OFF(mask_ptr)]);
#undef GET_OFF

    // W = J.W;
    // HW = J.H * J.W;
    // int LSB = this->use_h_parallelism ? W : HW;

    //sub(t_, reg_block * buffer_block);
    this->mov(this->imm_addr64_, float2int(this->nalphabeta_));
    this->movq(this->xnalphabeta_, this->imm_addr64_);
    this->vbroadcastss(this->znalphabeta_, this->xnalphabeta_);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::execute_compute_loop(
        unsigned C) {

    const unsigned num_16c_blocks = std::ceil(C / 16);

    if (num_16c_blocks == 1u)
        compute_loop(across_version::Single);
    else {
        const auto middle_16_c_blocks = num_16c_blocks - 2;
        const int LSREST = middle_16_c_blocks % this->reg_block_;
        const int LS = middle_16_c_blocks - LSREST;

        if (LS > 0) this->mov(this->blockC_, LS);
        compute_loop(across_version::First);
        increment_loop_params(this->vlen_);

        Label lrn_loop;

        if (LS > 0) {
            this->L(lrn_loop);
            {
                compute_loop(across_version::Middle, this->reg_block_);
                increment_loop_params(this->reg_block_ * this->vlen_);
                this->sub(this->blockC_, this->reg_block_);
                this->cmp(this->blockC_, 0);
                this->jne(lrn_loop, this->T_NEAR);
            }
        }

        if (LSREST > 0) {
            compute_loop(across_version::Middle, LSREST);
            increment_loop_params(LSREST * this->vlen_);
        }

        compute_loop(across_version::Last);
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::compute_loop(
        across_version version, int loop_size_param) {
    load_compute_data(version, loop_size_param);
    compute(loop_size_param);
    store_compute_data(loop_size_param);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::compute(
        int loop_size_param) {
    const auto loop_size = loop_size_param;

    IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffsrc_), this->zreg(irb, this->za_)));
    assert(this->zsrc_ == this->za_);
    IRB_LOOP(this->load_data(this->zreg(irb, this->zsrc_),
            this->EVEX_compress_addr(this->src_, irb * this->vlen_)));
    IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffsrc_), this->zreg(irb, this->zb_)));
    IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffsrc_), this->zreg(irb, this->zd_)));
    IRB_LOOP(this->vaddps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffsrc_), this->zreg(irb, this->ze_)));
    IRB_LOOP(this->vmulps(this->zreg(irb, this->zsrc_),
            this->zreg(irb, this->zsrc_), this->znalphabeta_));

    IRB_LOOP(this->load_data(this->zreg(irb, this->zws0_),
            this->EVEX_compress_addr(this->workspace0_, irb * this->vlen_)));
    IRB_LOOP(this->vdivps(this->zreg(irb, this->zdiffdst_),
            this->zreg(irb, this->zdiffdst_), this->zreg(irb, this->zws0_)));
    IRB_LOOP(this->vfmadd213ps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zsrc_), this->zreg(irb, this->zdiffdst_)));
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::increment_loop_params(
        std::size_t offset) {
    this->add(this->src_, offset);
    this->add(this->diffsrc_, offset);
    this->add(this->diffdst_, offset);
    this->add(this->workspace0_, offset);
    this->add(this->workspace1_, offset);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::load_compute_data(
        across_version version, int loop_size_param) {

    const int loop_size = loop_size_param;
    static constexpr int mask_shift = sizeof(int32_t);
    static constexpr int acc_size
            = d_type == bf16 ? 2 : 4; //sizeof(acc_data_t);
    const auto load_shifted_padded_with_zeros
            = [this](int dstIdx, int srcIdx, int maskTmpIdx, int offset) {
                  this->vxorps(this->zreg(0, dstIdx), this->zreg(0, dstIdx),
                          this->zreg(0, dstIdx));
                  this->vmovups(this->zreg(0, maskTmpIdx),
                          this->EVEX_compress_addr(this->mask_, offset));
                  this->vpermt2ps(this->zreg(0, dstIdx),
                          this->zreg(0, maskTmpIdx), this->zreg(0, srcIdx));
              };

    const auto load_ws_diffdst = [&, this](int dstIdx, int offset) {
        IRB_LOOP(this->load_data(this->zreg(irb, dstIdx),
                this->EVEX_compress_addr(
                        this->workspace1_, (irb * this->vlen_) + offset)));
        if (d_type == bf16) {
            IRB_LOOP(this->load_data(this->zreg(irb, this->z_tmp_),
                    this->EVEX_compress_addr(
                            this->diffdst_, (irb * this->vlen_) + offset)));
            IRB_LOOP(this->vmulps(this->zreg(irb, dstIdx),
                    this->zreg(irb, this->z_tmp_), this->zreg(irb, dstIdx)));
        } else {
            IRB_LOOP(this->vmulps(this->zreg(irb, dstIdx),
                    this->zreg(irb, dstIdx),
                    this->EVEX_compress_addr(
                            this->diffdst_, (irb * this->vlen_) + offset)));
        }
    };

    IRB_LOOP(this->load_data(this->zreg(irb, this->zdiffsrc_),
            this->EVEX_compress_addr(this->workspace1_, irb * this->vlen_)));
    IRB_LOOP(this->load_data(this->zreg(irb, this->zdiffdst_),
            this->EVEX_compress_addr(this->diffdst_, irb * this->vlen_)));
    IRB_LOOP(this->vmulps(this->zreg(irb, this->zdiffsrc_),
            this->zreg(irb, this->zdiffdst_),
            this->zreg(irb, this->zdiffsrc_)));

    if (version == across_version::First || version == across_version::Single) {
        load_shifted_padded_with_zeros(this->za_, this->zdiffsrc_,
                this->tmp_mask_za_idx_, -2 * mask_shift);
        load_shifted_padded_with_zeros(this->zb_, this->zdiffsrc_,
                this->tmp_mask_zb_idx_, -1 * mask_shift);
    } else {
        load_ws_diffdst(this->za_, -2 * acc_size);
        load_ws_diffdst(this->zb_, -1 * acc_size);
    }

    if (version == across_version::Last || version == across_version::Single) {
        load_shifted_padded_with_zeros(
                this->zd_, this->zdiffsrc_, this->tmp_mask_zd_idx_, mask_shift);
        load_shifted_padded_with_zeros(this->ze_, this->zdiffsrc_,
                this->tmp_mask_ze_idx_, 2 * mask_shift);
    } else {
        load_ws_diffdst(this->zd_, 1 * acc_size);
        load_ws_diffdst(this->ze_, 2 * acc_size);
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_bwd_nhwc_t<d_type>::store_compute_data(
        int loop_size_param) {
    const int loop_size = loop_size_param;

    Label unaligned_store, end_store;
    this->test(this->diffsrc_, this->vlen_ - 1);
    this->jnz(unaligned_store, this->T_NEAR);
    IRB_LOOP(this->store_data(true,
            this->EVEX_compress_addr(this->diffsrc_, irb * this->vlen_),
            this->zreg(irb, this->zdiffsrc_)));
    this->jmp(end_store, this->T_NEAR);
    this->L(unaligned_store);
    {
        IRB_LOOP(this->store_data(false,
                this->EVEX_compress_addr(this->diffsrc_, irb * this->vlen_),
                this->zreg(irb, this->zdiffsrc_)));
    }
    this->L(end_store);
}

template class jit_avx512_common_lrn_kernel_bwd_nhwc_t<f32>;
template class jit_avx512_common_lrn_kernel_bwd_nhwc_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
