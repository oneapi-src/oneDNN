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
#include "cpu/x64/lrn/jit_avx512_common_lrn_fwd_nhwc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace lrn {

template <data_type_t d_type>
jit_avx512_common_lrn_kernel_fwd_nhwc_t<
        d_type>::jit_avx512_common_lrn_kernel_fwd_nhwc_t(unsigned C,
        prop_kind_t prop_kind, float alpha, float k, void *code_ptr,
        size_t code_size)
    : jit_avx512_common_lrn_kernel_fwd_t<d_type>(
            prop_kind, alpha, k, code_ptr, code_size) {

    const auto res = std::div(C, 16);
    const auto &C_tail = res.rem;
    const auto &num_full_16c_blocks = res.quot;
    static const auto stack_space = zmm_size * 3;

    this->preamble();
    if (C_tail) reserve_stack_space(stack_space);
    this->set_up_ker_params();
    this->execute_compute_loop(num_full_16c_blocks, C_tail);
    if (C_tail) unreserve_stack_space(stack_space);
    this->postamble();
    this->ker = reinterpret_cast<decltype(this->ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::reserve_stack_space(
        std::size_t space) {
    this->sub(rsp, space);
    this->vxorps(zmm4, zmm4, zmm4);
    for (unsigned i = 0; i < 2u; ++i)
        this->vmovups(ptr[rsp + i * zmm_size], zmm4);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::unreserve_stack_space(
        std::size_t space) {
    this->add(rsp, space);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::set_up_ker_params() {

#define GET_OFF(field) \
    offsetof(typename jit_avx512_common_lrn_kernel_fwd_t< \
                     d_type>::jit_args_fwd_t, \
            field)
    this->mov(this->src_, ptr[this->param_ + GET_OFF(src)]);
    this->mov(this->dst_, ptr[this->param_ + GET_OFF(dst)]);
    if (this->pk_ != prop_kind::forward_inference) {
        this->mov(this->ws0_, ptr[this->param_ + GET_OFF(ws0)]);
        this->mov(this->ws1_, ptr[this->param_ + GET_OFF(ws1)]);
    }
    this->mov(this->mask_, ptr[this->param_ + GET_OFF(mask_ptr)]);
#undef GET_OFF

    this->mov(this->imm_addr64_, float2int(this->alpha_));
    this->movq(this->xalpha_, this->imm_addr64_);
    this->vbroadcastss(this->zalpha_, this->xalpha_);

    this->mov(this->imm_addr64_, float2int(this->k_));
    this->movq(this->xk_, this->imm_addr64_);
    this->vbroadcastss(this->zk_, this->xk_);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::execute_compute_loop(
        unsigned num_full_16c_blocks, unsigned C_tail) {

    if ((num_full_16c_blocks == 1u && !C_tail)
            || (num_full_16c_blocks == 0u && C_tail)) {
        const auto tail_proc
                = C_tail ? tail_mode::CurrentTail : tail_mode::NoTail;
        compute_loop(across_version::Single, tail_proc, C_tail);
    } else {
        const int begin_end = C_tail ? 1 : 2;
        int middle_16_c_blocks = num_full_16c_blocks == 1
                ? 0
                : num_full_16c_blocks - begin_end;
        int LTAIL = 0;
        if (C_tail && middle_16_c_blocks) {
            middle_16_c_blocks -= 1;
            LTAIL = 1;
        }

        const int LSREST = middle_16_c_blocks % this->reg_block_;
        const int LS = middle_16_c_blocks - LSREST;

        if (LS > 0) this->mov(this->blockC_, LS);
        const auto first_tail_proc = num_full_16c_blocks == 1
                ? tail_mode::NextTail
                : tail_mode::NoTail;
        compute_loop(across_version::First, first_tail_proc, C_tail);
        increment_loop_params(this->vlen_);

        Label lrn_loop;

        if (LS > 0) {

            this->L(lrn_loop);
            {
                compute_loop(across_version::Middle, tail_mode::NoTail, C_tail,
                        this->reg_block_);
                increment_loop_params(this->reg_block_ * this->vlen_);
                this->sub(this->blockC_, this->reg_block_);
                this->cmp(this->blockC_, 0);
                this->jne(lrn_loop, this->T_NEAR);
            }
        }

        if (LSREST > 0) {
            compute_loop(
                    across_version::Middle, tail_mode::NoTail, C_tail, LSREST);
            increment_loop_params(LSREST * this->vlen_);
        }

        if (LTAIL) {
            compute_loop(
                    across_version::Middle, tail_mode::NextTail, C_tail, LTAIL);
            increment_loop_params(LTAIL * this->vlen_);
        }

        const auto last_tail_proc
                = C_tail ? tail_mode::CurrentTail : tail_mode::NoTail;
        compute_loop(across_version::Last, last_tail_proc, C_tail);
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::increment_loop_params(
        std::size_t offset) {

    this->add(this->src_, offset);
    this->add(this->dst_, offset);
    if (this->pk_ != prop_kind::forward_inference) {
        this->add(this->ws0_, offset);
        this->add(this->ws1_, offset);
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::compute_loop(
        across_version version, tail_mode tail_proc, unsigned C_tail,
        int loop_size_param) {

    if (tail_proc != tail_mode::NoTail)
        load_data_to_stack(C_tail, version, tail_proc);
    load_compute_data(version, tail_proc, loop_size_param);
    compute(loop_size_param);
    store_compute_data(loop_size_param, tail_proc, C_tail);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::load_data_to_stack(
        unsigned C_tail, across_version version, tail_mode tail_proc) {

    if (version != across_version::Single) {
        const int previousChunkOffset
                = tail_proc == tail_mode::NextTail ? 0 : -1 * this->vlen_;
        this->load_data(this->zreg(0, tmp_load_to_stack_idx_prev_),
                this->EVEX_compress_addr(this->src_, previousChunkOffset));
        this->vmovups(this->EVEX_compress_addr(rsp, 0),
                this->zreg(0, tmp_load_to_stack_idx_prev_));
    }

    const int tail_src_mem_offset
            = tail_proc == tail_mode::NextTail ? this->vlen_ : 0;
    static constexpr int tail_dst_stack_offset = zmm_size;
    this->load_tail(C_tail, this->src_, tail_src_mem_offset,
            tail_dst_stack_offset, this->tmp_load_to_stack_idx_tail_);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::load_compute_data(
        across_version version, tail_mode tail_proc, int loop_size_param) {

    static constexpr int acc_bf_16_size = sizeof(acc_data_bf16_t);
    static constexpr int acc_size
            = d_type == bf16 ? acc_bf_16_size : sizeof(acc_data_t);

    const int loop_size = loop_size_param;
    static constexpr int mask_shift = sizeof(int32_t);
    const auto load_shifted_padded_with_zeros
            = [&](int dstIdx, int srcIdx, int maskTmpIdx, int offset) {
                  this->vxorps(this->zreg(0, dstIdx), this->zreg(0, dstIdx),
                          this->zreg(0, dstIdx));
                  this->load_data(this->zreg(0, maskTmpIdx),
                          this->EVEX_compress_addr(this->mask_, offset), true);
                  this->vpermt2ps(this->zreg(0, dstIdx),
                          this->zreg(0, maskTmpIdx), this->zreg(0, srcIdx));
              };

    if (tail_proc == tail_mode::CurrentTail) {
        this->load_data(this->zreg(0, this->zc_),
                this->EVEX_compress_addr(rsp, zmm_size), true);
    } else {
        IRB_LOOP(this->load_data(this->zreg(irb, this->zc_),
                this->EVEX_compress_addr(this->src_, irb * this->vlen_)));
    }

    if (version == across_version::First || version == across_version::Single) {
        load_shifted_padded_with_zeros(
                this->za_, this->zc_, this->tmp_mask_za_idx_, -2 * mask_shift);
        load_shifted_padded_with_zeros(
                this->zb_, this->zc_, this->tmp_mask_zb_idx_, -1 * mask_shift);
    } else {
        if (tail_proc == tail_mode::CurrentTail) {
            this->load_data(this->zreg(0, this->za_),
                    this->EVEX_compress_addr(rsp, zmm_size - 2 * acc_size),
                    true);
            this->load_data(this->zreg(0, this->zb_),
                    this->EVEX_compress_addr(rsp, zmm_size - acc_size), true);
        } else {
            IRB_LOOP(this->load_data(this->zreg(irb, this->za_),
                    this->EVEX_compress_addr(
                            this->src_, (irb * this->vlen_) - 2 * acc_size)));
            IRB_LOOP(this->load_data(this->zreg(irb, this->zb_),
                    this->EVEX_compress_addr(
                            this->src_, (irb * this->vlen_) - acc_size)));
        }
    }

    if (version == across_version::Last || version == across_version::Single) {
        load_shifted_padded_with_zeros(
                this->zd_, this->zc_, this->tmp_mask_zd_idx_, mask_shift);
        load_shifted_padded_with_zeros(
                this->ze_, this->zc_, this->tmp_mask_ze_idx_, 2 * mask_shift);
    } else {
        if (tail_proc == tail_mode::NextTail) {
            this->load_data(this->zreg(0, this->zd_),
                    this->EVEX_compress_addr(rsp, acc_size), true);
            this->load_data(this->zreg(0, this->ze_),
                    this->EVEX_compress_addr(rsp, 2 * acc_size), true);
        } else {
            IRB_LOOP(this->load_data(this->zreg(irb, this->zd_),
                    this->EVEX_compress_addr(
                            this->src_, (irb * this->vlen_) + acc_size)));
            IRB_LOOP(this->load_data(this->zreg(irb, this->ze_),
                    this->EVEX_compress_addr(
                            this->src_, (irb * this->vlen_) + 2 * acc_size)));
        }
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::compute(
        int loop_size_param) {

    const int loop_size = loop_size_param;

    IRB_LOOP(this->vmulps(this->zreg(irb, this->zsum_),
            this->zreg(irb, this->zc_), this->zreg(irb, this->zc_)));

    for (const auto &regIdx : {this->za_, this->zb_, this->zd_, this->ze_})
        IRB_LOOP(this->vfmadd231ps(this->zreg(irb, this->zsum_),
                this->zreg(irb, regIdx), this->zreg(irb, regIdx)));

    IRB_LOOP(this->vfmadd132ps(
            this->zreg(irb, this->zsum_), this->zk_, this->zalpha_));
    IRB_LOOP(this->vmovaps(
            this->zreg(irb, this->zbase_), this->zreg(irb, this->zsum_)));
    IRB_LOOP(this->vmulps(this->zreg(irb, this->zsum2_),
            this->zreg(irb, this->zsum_), this->zreg(irb, this->zsum_)));
    IRB_LOOP(this->vmulps(this->zreg(irb, this->zsum_),
            this->zreg(irb, this->zsum_), this->zreg(irb, this->zsum2_)));

    for (unsigned i = 0; i < 2; ++i)
        IRB_LOOP(this->vsqrtps(
                this->zreg(irb, this->zsum_), this->zreg(irb, this->zsum_)));
}

template <data_type_t d_type>
void jit_avx512_common_lrn_kernel_fwd_nhwc_t<d_type>::store_compute_data(
        int loop_size_param, tail_mode tail_proc, unsigned C_tail) {

    const int loop_size = loop_size_param;
    static const int ytmp = this->zsum2_;

    if (this->pk_ != prop_kind::forward_inference) {
        // save intermediate results for lrn backward
        if (tail_proc == tail_mode::CurrentTail)
            this->store_tail(C_tail, this->zreg(0, this->zsum_), this->ws0_, 0,
                    2 * zmm_size, tmp_store_from_stack_idx_tail_);
        else
            IRB_LOOP(this->store_data(
                    this->EVEX_compress_addr(this->ws0_, irb * this->vlen_),
                    this->zreg(irb, this->zsum_), this->yreg(irb, ytmp)));
    }
    IRB_LOOP(this->vdivps(this->zreg(irb, this->zdst_),
            this->zreg(irb, this->zsrc_), this->zreg(irb, this->zsum_)));
    // storing to dst
    if (tail_proc == tail_mode::CurrentTail)
        this->store_tail(C_tail, this->zreg(0, this->zdst_), this->dst_, 0,
                2 * zmm_size, tmp_store_from_stack_idx_tail_);
    else
        IRB_LOOP(this->store_data(
                this->EVEX_compress_addr(this->dst_, irb * this->vlen_),
                this->zreg(irb, this->zdst_), this->yreg(irb, ytmp)));

    if (this->pk_ != prop_kind::forward_inference) {
        // calculate and save more intermediate results for lrn backward
        /* ws1 = zdst / zbase = zsrc / (zbase^1.75) */

        IRB_LOOP(this->vdivps(this->zreg(irb, this->zsum_),
                this->zreg(irb, this->zdst_), this->zreg(irb, this->zbase_)));

        if (tail_proc == tail_mode::CurrentTail)
            this->store_tail(C_tail, this->zreg(0, this->zsum_), this->ws1_, 0,
                    2 * zmm_size, tmp_store_from_stack_idx_tail_);
        else
            IRB_LOOP(this->store_data(
                    this->EVEX_compress_addr(this->ws1_, irb * this->vlen_),
                    this->zreg(irb, this->zsum_), this->yreg(irb, ytmp)));
    }
}

template class jit_avx512_common_lrn_kernel_fwd_nhwc_t<f32>;
template class jit_avx512_common_lrn_kernel_fwd_nhwc_t<bf16>;

} // namespace lrn
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
