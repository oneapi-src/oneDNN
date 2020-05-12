/*******************************************************************************
* Copyright 2017-2020 Intel Corporation
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

#include "cpu/x64/jit_avx512_common_lrn.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

using acc_data_t = float;

#define IRB_LOOP(statement) \
    for (int irb = 0; irb < loop_size; irb++) { \
        statement; \
    }

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;
using namespace data_type;
using namespace Xbyak;
using namespace Xbyak::util;

/*  version:
*  First: channels 0..15,
*  Middle: channels C-16 .. C-1,
*  Last: other channels
*  Single: channels only for this kernel(without prev and next)
*/
enum class fwd_across_version : char { First, Middle, Last, Single };

struct nChw16c_across {
    int H, W;
    fwd_across_version version;
    nChw16c_across(int h, int w, fwd_across_version version)
        : H(h), W(w), version(version) {}
};

template <data_type_t d_type>
class jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_fwd_f
    : public jit_generator {
public:
    jit_avx512_common_lrn_kernel_fwd_f(prop_kind_t prop_kind, float alpha,
            float k, void *code_ptr, size_t code_size);

    struct jit_args_fwd_t {
        const data_t *src;
        data_t *dst, *ws0, *ws1;
        static constexpr int32_t mask[20] = {0, 0, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 0};
        const int32_t *mask_ptr = &mask[2];
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_fwd);
    void (*ker)(jit_args_fwd_t *);
    void operator()(jit_args_fwd_t *arg) { ker(arg); }

protected:
    static inline Zmm zreg(int irb, int i) { return Zmm(irb * 7 + i); };

    prop_kind_t pk_;
    float alpha_, k_;
    static constexpr int xmm_size_ = 4 * sizeof(acc_data_t);
    static constexpr int zmm_size_ = 64;
    const Reg64 imm_addr64_ = rbx;
    const Xmm xalpha_ = xmm0;
    const Zmm zalpha_ = zmm0;
    const Zmm zk_ = zmm1;
    const Xmm xk_ = xmm1;
    const Reg64 src_ = rax;
    const Reg64 dst_ = r8;
    const Reg64 ws0_ = rdx;
    const Reg64 ws1_ = rsi;
    const Reg64 param_ = abi_param1;
    static constexpr int zc_ = 7;
    static constexpr int za_ = 2;
    static constexpr int zb_ = 3;
    static constexpr int zd_ = 5;
    static constexpr int ze_ = 6;
    static constexpr int zsum_ = 4;
    static constexpr int zsum2_ = 5;
    static constexpr int zbase_ = 3;
    static constexpr int zsrc_ = 7;
    static constexpr int zdst_ = 2;
};

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_fwd_f::
        jit_avx512_common_lrn_kernel_fwd_f(prop_kind_t prop_kind, float alpha,
                float k, void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size)
    , pk_(prop_kind)
    , alpha_(alpha)
    , k_(k) {}

template <data_type_t d_type>
constexpr int jit_avx512_common_lrn_fwd_t<
        d_type>::jit_avx512_common_lrn_kernel_fwd_f::jit_args_fwd_t::mask[20];

template <data_type_t d_type>
class jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_nhwc_f
    : public jit_avx512_common_lrn_kernel_fwd_f {
public:
    jit_avx512_common_lrn_kernel_nhwc_f(unsigned C, prop_kind_t prop_kind,
            float alpha, float k, void *code_ptr = nullptr,
            size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_nhwc_f);

private:
    void set_up_ker_parmas();
    void execute_compute_loop(unsigned C);
    void compute_loop(fwd_across_version version, int loop_size_param = 1);
    void compute(int loop_size_param);
    void increment_loop_params(std::size_t offset);
    void load_compute_data(fwd_across_version version, int loop_size_param);
    void store_compute_data(int loop_size_param);

    static constexpr int tmp_mask_za_idx_ = 8;
    static constexpr int tmp_mask_zb_idx_ = 9;
    static constexpr int tmp_mask_zd_idx_ = 10;
    static constexpr int tmp_mask_ze_idx_ = 11;
    static constexpr int reg_block_ = 4;
    static constexpr int vlen_ = 64;
    const Reg64 mask_ = r10;
    const Reg64 blockC_ = r9;
};

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_nhwc_f::
        jit_avx512_common_lrn_kernel_nhwc_f(unsigned C, prop_kind_t prop_kind,
                float alpha, float k, void *code_ptr, size_t code_size)
    : jit_avx512_common_lrn_kernel_fwd_f(
            prop_kind, alpha, k, code_ptr, code_size) {

    this->preamble();
    this->set_up_ker_parmas();
    this->execute_compute_loop(C);
    this->postamble();
    this->ker = reinterpret_cast<decltype(this->ker)>(
            const_cast<uint8_t *>(this->getCode()));
}

template <data_type_t d_type>
void jit_avx512_common_lrn_fwd_t<
        d_type>::jit_avx512_common_lrn_kernel_nhwc_f::set_up_ker_parmas() {

#define GET_OFF(field) \
    offsetof(typename jit_avx512_common_lrn_kernel_fwd_f::jit_args_fwd_t, field)
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
void jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_nhwc_f::
        execute_compute_loop(unsigned C) {

    const unsigned num_16c_blocks = std::ceil(C / 16);

    if (num_16c_blocks == 1u)
        compute_loop(fwd_across_version::Single);
    else {
        const auto middle_16_c_blocks = num_16c_blocks - 2;
        const int LSREST = middle_16_c_blocks % this->reg_block_;
        const int LS = middle_16_c_blocks - LSREST;

        if (LS > 0) this->mov(this->blockC_, LS);
        compute_loop(fwd_across_version::First);
        increment_loop_params(this->vlen_);

        Label lrn_loop;

        if (LS > 0) {

            this->L(lrn_loop);
            {
                compute_loop(fwd_across_version::Middle, this->reg_block_);
                increment_loop_params(this->reg_block_ * this->vlen_);
                this->sub(this->blockC_, this->reg_block_);
                this->cmp(this->blockC_, 0);
                this->jne(lrn_loop, this->T_NEAR);
            }
        }

        if (LSREST > 0) {
            compute_loop(fwd_across_version::Middle, LSREST);
            increment_loop_params(LSREST * this->vlen_);
        }

        compute_loop(fwd_across_version::Last);
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_nhwc_f::
        increment_loop_params(std::size_t offset) {

    this->add(this->src_, offset);
    this->add(this->dst_, offset);
    if (this->pk_ != prop_kind::forward_inference) {
        this->add(this->ws0_, offset);
        this->add(this->ws1_, offset);
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_nhwc_f::
        compute_loop(fwd_across_version version, int loop_size_param) {

    load_compute_data(version, loop_size_param);
    compute(loop_size_param);
    store_compute_data(loop_size_param);
}

template <data_type_t d_type>
void jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_nhwc_f::
        load_compute_data(fwd_across_version version, int loop_size_param) {

    const int loop_size = loop_size_param;
    static constexpr int mask_shift = sizeof(int32_t);
    static constexpr int acc_size = sizeof(acc_data_t);
    const auto load_shifted_padded_with_zeros
            = [&](int dstIdx, int srcIdx, int maskTmpIdx, int offset) {
                  this->vpxorq(this->zreg(0, dstIdx), this->zreg(0, dstIdx),
                          this->zreg(0, dstIdx));
                  this->vmovups(this->zreg(0, maskTmpIdx),
                          this->EVEX_compress_addr(this->mask_, offset));
                  this->vpermt2ps(this->zreg(0, dstIdx),
                          this->zreg(0, maskTmpIdx), this->zreg(0, srcIdx));
              };

    IRB_LOOP(this->vmovups(this->zreg(irb, this->zc_),
            this->EVEX_compress_addr(this->src_, irb * this->vlen_)));

    if (version == fwd_across_version::First
            || version == fwd_across_version::Single) {
        load_shifted_padded_with_zeros(
                this->za_, this->zc_, this->tmp_mask_za_idx_, -2 * mask_shift);
        load_shifted_padded_with_zeros(
                this->zb_, this->zc_, this->tmp_mask_zb_idx_, -1 * mask_shift);
    } else {
        IRB_LOOP(this->vmovups(this->zreg(irb, this->za_),
                this->EVEX_compress_addr(
                        this->src_, (irb * this->vlen_) - 2 * acc_size)));
        IRB_LOOP(this->vmovups(this->zreg(irb, this->zb_),
                this->EVEX_compress_addr(
                        this->src_, (irb * this->vlen_) - acc_size)));
    }

    if (version == fwd_across_version::Last
            || version == fwd_across_version::Single) {
        load_shifted_padded_with_zeros(
                this->zd_, this->zc_, this->tmp_mask_zd_idx_, mask_shift);
        load_shifted_padded_with_zeros(
                this->ze_, this->zc_, this->tmp_mask_ze_idx_, 2 * mask_shift);
    } else {
        IRB_LOOP(this->vmovups(this->zreg(irb, this->zd_),
                this->EVEX_compress_addr(
                        this->src_, (irb * this->vlen_) + acc_size)));
        IRB_LOOP(this->vmovups(this->zreg(irb, this->ze_),
                this->EVEX_compress_addr(
                        this->src_, (irb * this->vlen_) + 2 * acc_size)));
    }
}

template <data_type_t d_type>
void jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_nhwc_f::
        compute(int loop_size_param) {

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
void jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_kernel_nhwc_f::
        store_compute_data(int loop_size_param) {

    const int loop_size = loop_size_param;

    auto store_data
            = [=](const Address addr, Zmm zr) { this->vmovups(addr, zr); };

    if (this->pk_ != prop_kind::forward_inference) {
        // save intermediate results for lrn backward
        IRB_LOOP(store_data(
                this->EVEX_compress_addr(this->ws0_, irb * this->vlen_),
                this->zreg(irb, this->zsum_)));
    }
    IRB_LOOP(this->vdivps(this->zreg(irb, this->zdst_),
            this->zreg(irb, this->zsrc_), this->zreg(irb, this->zsum_)));
    // storing to dst
    IRB_LOOP(store_data(this->EVEX_compress_addr(this->dst_, irb * this->vlen_),
            this->zreg(irb, this->zdst_)));
    if (this->pk_ != prop_kind::forward_inference) {
        // calculate and save more intermediate results for lrn backward
        /* ws1 = zdst / zbase = zsrc / (zbase^1.75) */

        IRB_LOOP(this->vdivps(this->zreg(irb, this->zsum_),
                this->zreg(irb, this->zdst_), this->zreg(irb, this->zbase_)));
        IRB_LOOP(store_data(
                this->EVEX_compress_addr(this->ws1_, irb * this->vlen_),
                this->zreg(irb, this->zsum_)));
    }
}

template <data_type_t d_type>
struct jit_avx512_common_lrn_fwd_t<
        d_type>::jit_avx512_common_lrn_kernel_nChw16c_f
    : public jit_avx512_common_lrn_kernel_fwd_f {

    int xmm_size, zmm_size, buffer_block, buffer_nest_offset, src_prev_offset,
            vlen, reg_block;

    int HW, W;
    fwd_across_version version;
    Reg64 t = rsp;
    Reg64 hw = r9;
    Zmm bf16_emu_reserv_1 = Zmm(28);
    Zmm bf16_emu_reserv_2 = Zmm(29);
    Reg64 bf16_emu_scratch = rax;
    Zmm bf16_emu_reserv_3 = Zmm(30);
    Zmm bf16_emu_reserv_4 = Zmm(31);

    static constexpr int xsrc_prev = 2;
    static constexpr int xsrc_next = 3;
    int use_h_parallelism;

    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_fwd_f::
                    jit_avx512_common_lrn_kernel_nChw16c_f)

    inline void compute_loop(int loop_size_param) {
        // loop_size - param for IRB_LOOP macro
        const int prf0_offt = 1 * reg_block;
        const int prf2_offt = 8 * reg_block;

        int loop_size = reg_block;

        static const auto xreg
                = [=](int irb, int i) { return Xmm(irb * 3 + i); };
        static const auto yreg
                = [=](int irb, int i) { return Ymm(irb * 7 + i); };
        static const auto zreg
                = [=](int irb, int i) { return Zmm(irb * 7 + i); };
        const auto load_data = [=](Xmm reg, const Address p) {
            if (d_type == bf16) {
                this->vpmovzxwd(reg, p);
                this->vpslld(reg, reg, 0x10);
            } else
                this->vmovups(reg, p);
        };

        const auto store_data = [=](const Address addr, Zmm zr, Ymm yr) {
            if (d_type == bf16) {
                if (mayiuse(avx512_core_bf16))
                    this->vcvtneps2bf16(yr, zr);
                else
                    bf16_emu_->vcvtneps2bf16(yr, zr);
                this->vmovdqu16(addr, yr);
            } else
                this->vmovups(addr, zr);
        };

        if (version != fwd_across_version::First
                && version != fwd_across_version::Single) {
            IRB_LOOP(this->mic_prefetcht0(
                    ptr[this->src_ + (irb + prf0_offt - HW) * vlen]));
            IRB_LOOP(this->mic_prefetcht2(
                    ptr[this->src_ + (irb + prf2_offt - HW) * vlen]));
        }
        IRB_LOOP(this->mic_prefetcht0(this->EVEX_compress_addr(
                this->src_, (irb + prf0_offt) * vlen)));
        IRB_LOOP(this->mic_prefetcht2(this->EVEX_compress_addr(
                this->src_, (irb + prf2_offt) * vlen)));
        if (version != fwd_across_version::Last
                && version != fwd_across_version::Single) {
            IRB_LOOP(this->mic_prefetcht0(
                    ptr[this->src_ + (irb + prf0_offt + HW) * vlen]));
            IRB_LOOP(this->mic_prefetcht2(
                    ptr[this->src_ + (irb + prf2_offt + HW) * vlen]));
        }
        if (this->pk_ != prop_kind::forward_inference) {
            IRB_LOOP(this->mic_prefetcht0(this->EVEX_compress_addr(
                    this->ws0_, (irb + prf0_offt) * vlen)));
            IRB_LOOP(this->mic_prefetcht2(this->EVEX_compress_addr(
                    this->ws0_, (irb + prf2_offt) * vlen)));
        }
        IRB_LOOP(this->mic_prefetcht0(this->EVEX_compress_addr(
                this->dst_, (irb + prf0_offt) * vlen)));
        IRB_LOOP(this->mic_prefetcht2(this->EVEX_compress_addr(
                this->dst_, (irb + prf2_offt) * vlen)));
        if (this->pk_ != prop_kind::forward_inference) {
            IRB_LOOP(this->mic_prefetcht0(this->EVEX_compress_addr(
                    this->ws1_, (irb + prf0_offt) * vlen)));
            IRB_LOOP(this->mic_prefetcht2(this->EVEX_compress_addr(
                    this->ws1_, (irb + prf2_offt) * vlen)));
        }

        loop_size = loop_size_param;
        if (loop_size == 0) return;

        // --- loading source data to special buffer to form convenient data layout
        // for ACROSS lrn ---

        if (version != fwd_across_version::First
                && version != fwd_across_version::Single) {
            IRB_LOOP(load_data(xreg(irb, xsrc_prev),
                    ptr[this->src_ + (irb - HW) * vlen + src_prev_offset]));
        }
        IRB_LOOP(load_data(zreg(irb, this->zsrc_),
                this->EVEX_compress_addr(this->src_, irb * vlen)));
        if (version != fwd_across_version::Last
                && version != fwd_across_version::Single) {
            IRB_LOOP(load_data(
                    xreg(irb, xsrc_next), ptr[this->src_ + (irb + HW) * vlen]));
        }

        if (version != fwd_across_version::First
                && version != fwd_across_version::Single) {
            IRB_LOOP(this->vmovups(
                    ptr[t + irb * buffer_block], xreg(irb, xsrc_prev)));
        }
        IRB_LOOP(this->vmovups(
                this->EVEX_compress_addr(t, irb * buffer_block + xmm_size),
                zreg(irb, this->zsrc_)));
        if (version != fwd_across_version::Last
                && version != fwd_across_version::Single) {
            IRB_LOOP(this->vmovups(
                    ptr[t + irb * buffer_block + buffer_nest_offset],
                    xreg(irb, xsrc_next)));
        }

        // --- perform ACROSS lrn ---
        const size_t acc_size = sizeof(acc_data_t);
        IRB_LOOP(this->vmovups(zreg(irb, this->za_),
                this->EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size - 2 * acc_size)));
        IRB_LOOP(this->vmovups(zreg(irb, this->zb_),
                this->EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size - acc_size)));
        IRB_LOOP(this->vmovups(zreg(irb, this->zd_),
                this->EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size + acc_size)));
        IRB_LOOP(this->vmovups(zreg(irb, this->ze_),
                this->EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size + 2 * acc_size)));

        assert(this->zc_ == this->zsrc_);
        IRB_LOOP(this->vmulps(zreg(irb, this->zsum_), zreg(irb, this->zc_),
                zreg(irb, this->zc_)));

        IRB_LOOP(this->vfmadd231ps(zreg(irb, this->zsum_), zreg(irb, this->za_),
                zreg(irb, this->za_)));
        IRB_LOOP(this->vfmadd231ps(zreg(irb, this->zsum_), zreg(irb, this->zb_),
                zreg(irb, this->zb_)));
        IRB_LOOP(this->vfmadd231ps(zreg(irb, this->zsum_), zreg(irb, this->zd_),
                zreg(irb, this->zd_)));
        IRB_LOOP(this->vfmadd231ps(zreg(irb, this->zsum_), zreg(irb, this->ze_),
                zreg(irb, this->ze_)));

        IRB_LOOP(this->vfmadd132ps(
                zreg(irb, this->zsum_), this->zk_, this->zalpha_));

        IRB_LOOP(
                this->vmovaps(zreg(irb, this->zbase_), zreg(irb, this->zsum_)));

        IRB_LOOP(this->vmulps(zreg(irb, this->zsum2_), zreg(irb, this->zsum_),
                zreg(irb, this->zsum_)));
        IRB_LOOP(this->vmulps(zreg(irb, this->zsum_), zreg(irb, this->zsum_),
                zreg(irb, this->zsum2_)));

        IRB_LOOP(this->vsqrtps(zreg(irb, this->zsum_), zreg(irb, this->zsum_)));
        IRB_LOOP(this->vsqrtps(zreg(irb, this->zsum_), zreg(irb, this->zsum_)));

        const int ytmp = this->zsum2_; // temporary ymm for f32->bf16 conversion
        if (this->pk_ != prop_kind::forward_inference) {
            // save intermediate results for lrn backward
            IRB_LOOP(
                    store_data(this->EVEX_compress_addr(this->ws0_, irb * vlen),
                            zreg(irb, this->zsum_), yreg(irb, ytmp)));
        }
        IRB_LOOP(this->vdivps(zreg(irb, this->zdst_), zreg(irb, this->zsrc_),
                zreg(irb, this->zsum_)));
        // storing to dst
        IRB_LOOP(store_data(this->EVEX_compress_addr(this->dst_, irb * vlen),
                zreg(irb, this->zdst_), yreg(irb, ytmp)));
        if (this->pk_ != prop_kind::forward_inference) {
            // calculate and save more intermediate results for lrn backward
            /* ws1 = zdst / zbase = zsrc / (zbase^1.75) */
            IRB_LOOP(this->vdivps(zreg(irb, this->zsum_),
                    zreg(irb, this->zdst_), zreg(irb, this->zbase_)));
            IRB_LOOP(
                    store_data(this->EVEX_compress_addr(this->ws1_, irb * vlen),
                            zreg(irb, this->zsum_), yreg(irb, ytmp)));
        }
    }

    jit_avx512_common_lrn_kernel_nChw16c_f(const struct nChw16c_across &J,
            prop_kind_t prop_kind, int use_h_parallel, float A, float K,
            void *code_ptr = nullptr,
            size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_avx512_common_lrn_kernel_fwd_f(
                prop_kind, A, K, code_ptr, code_size)
        , use_h_parallelism(use_h_parallel)
        , bf16_emu_(nullptr) {
        version = J.version;
        vlen = d_type == bf16 ? 32 : 64;
        // some registers needed for conversion from bf16 to f32
        reg_block = (d_type == bf16 && !mayiuse(avx512_core_bf16)) ? 3 : 4;
        src_prev_offset = vlen - 4 * sizeof(data_t);

        xmm_size = 4 * sizeof(acc_data_t);
        zmm_size = 64;
        buffer_block = xmm_size + zmm_size + xmm_size;
        buffer_nest_offset = xmm_size + zmm_size;

        if (d_type == bf16 && !mayiuse(avx512_core_bf16)) {
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                    bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                    bf16_emu_scratch, bf16_emu_reserv_4);
            bf16_emu_->init_vcvtneps2bf16();
        }

        this->preamble();

#define GET_OFF(field) \
    offsetof(typename jit_avx512_common_lrn_kernel_fwd_f::jit_args_fwd_t, field)
        this->mov(this->src_, ptr[this->param_ + GET_OFF(src)]);
        this->mov(this->dst_, ptr[this->param_ + GET_OFF(dst)]);
        if (this->pk_ != prop_kind::forward_inference) {
            this->mov(this->ws0_, ptr[this->param_ + GET_OFF(ws0)]);
            this->mov(this->ws1_, ptr[this->param_ + GET_OFF(ws1)]);
        }
#undef GET_OFF

        W = J.W;
        HW = J.W * J.H;
        int LSB = use_h_parallelism ? W : HW;

        this->sub(t, reg_block * buffer_block);
        this->mov(this->imm_addr64_, float2int(this->alpha_));
        this->movq(this->xalpha_, this->imm_addr64_);
        this->vbroadcastss(this->zalpha_, this->xalpha_);

        this->mov(this->imm_addr64_, float2int(this->k_));
        this->movq(this->xk_, this->imm_addr64_);
        this->vbroadcastss(this->zk_, this->xk_);

        if (version == fwd_across_version::First
                || version == fwd_across_version::Single) {
            this->vpxorq(xmm2, xmm2, xmm2);
            for (int irb = 0; irb < reg_block; irb++) {
                this->vmovups(ptr[t + irb * buffer_block], xmm2);
            }
        }
        if (version == fwd_across_version::Last
                || version == fwd_across_version::Single) {
            this->vpxorq(xmm2, xmm2, xmm2);
            for (int irb = 0; irb < reg_block; irb++) {
                this->vmovups(
                        ptr[t + irb * buffer_block + buffer_nest_offset], xmm2);
            }
        }

        const int LSREST = LSB % reg_block;
        const int LS = LSB - LSREST;

        Label lrn_loop;

        if (LS > 0) {
            this->mov(hw, LS);

            this->L(lrn_loop);
            {
                compute_loop(reg_block);

                this->add(this->src_, reg_block * vlen);
                this->add(this->dst_, reg_block * vlen);
                if (this->pk_ != prop_kind::forward_inference) {
                    this->add(this->ws0_, reg_block * vlen);
                    this->add(this->ws1_, reg_block * vlen);
                }

                for (int irb = 0; irb < reg_block; irb++)
                    this->dec(hw);
                this->cmp(hw, 0);
                this->jne(lrn_loop, this->T_NEAR);
            }
        }

        compute_loop(LSREST);

        this->add(t, reg_block * buffer_block);
        this->postamble();

        this->ker = reinterpret_cast<decltype(this->ker)>(
                const_cast<uint8_t *>(this->getCode()));
    }
    ~jit_avx512_common_lrn_kernel_nChw16c_f() = default;
};

template <data_type_t d_type>
status_t jit_avx512_common_lrn_fwd_t<d_type>::pd_t::init(engine_t *engine) {
    using namespace prop_kind;
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());
    const bool ok = true && mayiuse(avx512_common)
            && IMPLICATION(d_type == bf16, mayiuse(avx512_core)) && is_fwd()
            && !has_zero_dim_memory() && everyone_is(d_type, data_d.data_type())
            && data_d.ndims() == 4 && data_d.dims()[1] % vsize == 0
            && attr()->has_default_values();
    if (!ok) return unimplemented;

    const auto fmt_tag
            = data_d.matches_one_of_tag(format_tag::nhwc, format_tag::nChw16c);

    const bool args_ok_across = true && desc()->alg_kind == lrn_across_channels
            && desc()->local_size == 5 && desc()->lrn_beta == 0.75
            && data_d.matches_tag(fmt_tag)
            && IMPLICATION(fmt_tag == format_tag::nhwc, d_type != bf16);

    if (!args_ok_across) return unimplemented;

    if (desc()->prop_kind == forward_training) {
        dims_t ws_dims = {MB(), C(), H(), 2 * W()};
        dnnl_memory_desc_init_by_tag(&ws_md_, 4, ws_dims, d_type, fmt_tag);
    }

    return success;
}

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::jit_avx512_common_lrn_fwd_t(
        const pd_t *apd)
    : primitive_t(apd)
    , use_h_parallelism_(0)
    , ker_(nullptr)
    , ker_first_(nullptr)
    , ker_last_(nullptr) {
    using namespace alg_kind;
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int ls = pd()->desc()->local_size;
    const float alpha = pd()->desc()->lrn_alpha / ls;
    const float k = pd()->desc()->lrn_k;

    const auto pk = pd()->desc()->prop_kind;
    const memory_desc_wrapper data_d(pd()->src_md());

    use_h_parallelism_ = H > 28 ? 1 : 0;

    if (data_d.matches_tag(format_tag::nChw16c)) {
        if (C / vsize == 1) {
            ker_ = utils::make_unique<jit_avx512_common_lrn_kernel_nChw16c_f>(
                    nChw16c_across(H, W, fwd_across_version::Single), pk,
                    use_h_parallelism_, alpha, k);
        } else {
            ker_ = utils::make_unique<jit_avx512_common_lrn_kernel_nChw16c_f>(
                    nChw16c_across(H, W, fwd_across_version::Middle), pk,
                    use_h_parallelism_, alpha, k);
            ker_first_ = utils::make_unique<
                    jit_avx512_common_lrn_kernel_nChw16c_f>(
                    nChw16c_across(H, W, fwd_across_version::First), pk,
                    use_h_parallelism_, alpha, k);
            ker_last_ = utils::make_unique<
                    jit_avx512_common_lrn_kernel_nChw16c_f>(
                    nChw16c_across(H, W, fwd_across_version::Last), pk,
                    use_h_parallelism_, alpha, k);
        }
    } else if (data_d.matches_tag(format_tag::nhwc)) {
        ker_ = utils::make_unique<jit_avx512_common_lrn_kernel_nhwc_f>(
                C, pk, alpha, k);
    }
}

template <data_type_t d_type>
jit_avx512_common_lrn_fwd_t<d_type>::~jit_avx512_common_lrn_fwd_t() = default;

template <data_type_t d_type>
void jit_avx512_common_lrn_fwd_t<d_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(data_t *, DNNL_ARG_WORKSPACE);

    const int N = pd()->MB();
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();

    const memory_desc_wrapper src_d = pd()->src_md();

    if (src_d.matches_tag(format_tag::nChw16c)) {

        parallel(0, [&](const int ithr, const int nthr) {
            size_t start {0}, end {0};
            const int C16 = C / vsize;
            const size_t work_amount
                    = use_h_parallelism_ ? N * C16 * H : N * C16;

            balance211(work_amount, nthr, ithr, start, end);
            if (use_h_parallelism_) {
                int n {0}, c16 {0}, h {0};
                nd_iterator_init(start, n, N, c16, C16, h, H);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto offset = n * C * H * W + c16 * H * W * vsize
                            + h * W * vsize;
                    auto ws_offset0 = n * C * H * 2 * W
                            + c16 * H * 2 * W * vsize + h * 2 * W * vsize;
                    auto ws_offset1 = ws_offset0 + W * vsize;

                    typename jit_avx512_common_lrn_kernel_fwd_f::jit_args_fwd_t
                            args;
                    args.src = &src[offset];
                    args.dst = &dst[offset];
                    args.ws0 = &ws[ws_offset0];
                    args.ws1 = &ws[ws_offset1];

                    if (C16 == 1)
                        (*ker_)(&args);
                    else if (c16 == 0)
                        (*ker_first_)(&args);
                    else if (c16 == C16 - 1)
                        (*ker_last_)(&args);
                    else
                        (*ker_)(&args);
                    nd_iterator_step(n, N, c16, C16, h, H);
                }
            } else {
                int n {0}, c16 {0};
                nd_iterator_init(start, n, N, c16, C16);
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto offset = n * C * H * W + c16 * H * W * vsize;
                    auto ws_offset0
                            = n * C * H * 2 * W + c16 * H * 2 * W * vsize;
                    auto ws_offset1 = ws_offset0 + H * W * vsize;

                    typename jit_avx512_common_lrn_kernel_fwd_f::jit_args_fwd_t
                            args;
                    args.src = &src[offset];
                    args.dst = &dst[offset];
                    args.ws0 = &ws[ws_offset0];
                    args.ws1 = &ws[ws_offset1];

                    if (C16 == 1)
                        (*ker_)(&args);
                    else if (c16 == 0)
                        (*ker_first_)(&args);
                    else if (c16 == C16 - 1)
                        (*ker_last_)(&args);
                    else
                        (*ker_)(&args);

                    nd_iterator_step(n, N, c16, C16);
                }
            }
        });
    } else if (src_d.matches_tag(format_tag::nhwc)) {
        const auto ker = ker_.get();
        parallel_nd(N, H * W, [&](int n, int pixel_id) {
            typename jit_avx512_common_lrn_kernel_fwd_f::jit_args_fwd_t args;
            const auto offset = n * C * H * W + pixel_id * C;
            const auto ws_offset0 = offset * 2;
            const auto ws_offset1 = ws_offset0 + vsize;

            args.src = &src[offset];
            args.dst = &dst[offset];
            args.ws0 = &ws[ws_offset0];
            args.ws1 = &ws[ws_offset1];

            (*ker)(&args);
        });
    }
}

template struct jit_avx512_common_lrn_fwd_t<f32>;
template struct jit_avx512_common_lrn_fwd_t<bf16>;

template <data_type_t d_type>
struct jit_avx512_common_lrn_bwd_t<
        d_type>::jit_avx512_common_lrn_kernel_nChw16c_f : public jit_generator {
    struct jit_args_bwd_t {
        const data_t *src, *diff_dst, *ws0, *ws1;
        data_t *diff_src;
    };

    int xmm_size, zmm_size, buffer_block, buffer_nest_offset, src_prev_offset,
            vlen, reg_block;
    int HW, W;
    fwd_across_version version;

    Reg64 src = rax;
    Reg64 diffsrc = r8;
    Reg64 diffdst = r9;
    Reg64 workspace0 = rdx;
    Reg64 workspace1 = rsi;
    Reg64 imm_addr64 = rbx;
    Reg64 param = abi_param1;
    Zmm znalphabeta = zmm0;
    Xmm xnalphabeta = xmm0;

    Reg64 t = rsp;
    Reg64 hw = r10;
    Zmm bf16_emu_reserv_1 = Zmm(28);
    Zmm bf16_emu_reserv_2 = Zmm(29);
    Reg64 bf16_emu_scratch = rax;
    Zmm bf16_emu_reserv_3 = Zmm(30);
    Zmm bf16_emu_reserv_4 = Zmm(31);

    const int xws1_prev = 1;
    const int xdiffdst_prev = 2;
    const int zws1 = 1;

    const int zsrc = 1;
    const int zdiffdst = 5;
    const int zdiffsrc = 6;

    const int xws1_next = 1;
    const int xdiffdst_next = 3;

    const int za = 1;
    const int zb = 2;
    const int zd = 3;
    const int ze = 4;
    const int zws0 = 2;

    float nalphabeta;

    int use_h_parallelism;
    bf16_emulation_t *bf16_emu_;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_nChw16c_f)

    void (*ker)(jit_args_bwd_t *);
    void operator()(jit_args_bwd_t *arg) { ker(arg); }

    inline void compute_loop(
            int loop_size_param, int prefetchL1, int prefetchL2) {
        // loop_size - param for IRB_LOOP macro
        int loop_size = loop_size_param;
        const int prf0_offt = 1 * reg_block;
        const int prf2_offt = 8 * reg_block;

        auto xreg = [=](int irb, int i) { return Xmm(irb * 6 + i); };

        auto zreg = [=](int irb, int i) { return Zmm(irb * 6 + i); };
        auto load_data = [=](Xmm reg, const Address p) {
            if (d_type == bf16) {
                vpmovzxwd(reg, p);
                vpslld(reg, reg, 0x10);
            } else
                vmovups(reg, p);
        };

        auto store_data = [=](bool nt, const Address addr, Zmm zr) {
            if (d_type == bf16) {
                Ymm yr = Ymm(zr.getIdx());
                if (mayiuse(avx512_core_bf16))
                    vcvtneps2bf16(yr, zr);
                else
                    bf16_emu_->vcvtneps2bf16(yr, zr);
                vmovdqu16(addr, yr);
            } else if (nt)
                uni_vmovntps(addr, zr);
            else
                uni_vmovups(addr, zr);
        };

        // ---- prefetching -------------------------------------------
        if (version != fwd_across_version::First
                && version != fwd_across_version::Single) {
            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(
                        ptr[workspace1 + (irb + prf0_offt - 2 * HW) * vlen]));
            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(
                        ptr[diffdst + (irb + prf0_offt - HW) * vlen]));
        }

        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(ptr[src + (irb + prf0_offt) * vlen]));
        if (prefetchL2)
            IRB_LOOP(mic_prefetcht2(ptr[src + (irb + prf2_offt) * vlen]));

        if (prefetchL1)
            IRB_LOOP(
                    mic_prefetcht0(ptr[workspace1 + (irb + prf0_offt) * vlen]));

        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(ptr[diffdst + (irb + prf0_offt) * vlen]));

        if (version != fwd_across_version::Last
                && version != fwd_across_version::Single) {
            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(
                        ptr[workspace1 + (irb + prf0_offt + 2 * HW) * vlen]));
            if (prefetchL2)
                IRB_LOOP(mic_prefetcht2(
                        ptr[workspace1 + (irb + prf2_offt + 2 * HW) * vlen]));

            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(
                        ptr[diffdst + (irb + prf0_offt + HW) * vlen]));
            if (prefetchL2)
                IRB_LOOP(mic_prefetcht2(
                        ptr[diffdst + (irb + prf2_offt + HW) * vlen]));
        }
        if (prefetchL1)
            IRB_LOOP(
                    mic_prefetcht0(ptr[workspace0 + (irb + prf0_offt) * vlen]));
        if (prefetchL2)
            IRB_LOOP(
                    mic_prefetcht2(ptr[workspace0 + (irb + prf2_offt) * vlen]));
        // -----------------------------------------------------------

        if (loop_size_param == 0) return;

        if (version != fwd_across_version::First
                && version != fwd_across_version::Single) {
            IRB_LOOP(load_data(xreg(irb, xws1_prev),
                    ptr[workspace1 + (irb - 2 * HW) * vlen + src_prev_offset]));
            IRB_LOOP(load_data(xreg(irb, xdiffdst_prev),
                    ptr[diffdst + (irb - HW) * vlen + src_prev_offset]));
            IRB_LOOP(vmulps(xreg(irb, xdiffdst_prev), xreg(irb, xdiffdst_prev),
                    xreg(irb, xws1_prev)));
        }

        IRB_LOOP(load_data(
                zreg(irb, zws1), EVEX_compress_addr(workspace1, irb * vlen)));
        IRB_LOOP(load_data(
                zreg(irb, zdiffdst), EVEX_compress_addr(diffdst, irb * vlen)));
        IRB_LOOP(vmulps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffdst), zreg(irb, zws1)));

        if (version != fwd_across_version::Last
                && version != fwd_across_version::Single) {
            IRB_LOOP(load_data(xreg(irb, xws1_next),
                    ptr[workspace1 + (irb + 2 * HW) * vlen]));
            IRB_LOOP(load_data(xreg(irb, xdiffdst_next),
                    ptr[diffdst + (irb + HW) * vlen]));
            IRB_LOOP(vmulps(xreg(irb, xdiffdst_next), xreg(irb, xdiffdst_next),
                    xreg(irb, xws1_next)));
        }

        if (version != fwd_across_version::First
                && version != fwd_across_version::Single) {
            IRB_LOOP(vmovups(
                    ptr[t + irb * buffer_block], xreg(irb, xdiffdst_prev)));
        }
        IRB_LOOP(vmovups(EVEX_compress_addr(t, irb * buffer_block + xmm_size),
                zreg(irb, zdiffsrc)));
        if (version != fwd_across_version::Last
                && version != fwd_across_version::Single) {
            IRB_LOOP(vmovups(ptr[t + irb * buffer_block + buffer_nest_offset],
                    xreg(irb, xdiffdst_next)));
        }
        size_t acc_size = sizeof(acc_data_t);
        IRB_LOOP(vmovups(zreg(irb, za),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size - 2 * acc_size)));
        IRB_LOOP(vmovups(zreg(irb, zb),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size - 1 * acc_size)));
        IRB_LOOP(vmovups(zreg(irb, zd),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size + 1 * acc_size)));
        IRB_LOOP(vmovups(zreg(irb, ze),
                EVEX_compress_addr(
                        t, irb * buffer_block + xmm_size + 2 * acc_size)));
        IRB_LOOP(vaddps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, za)));
        assert(zsrc == za);
        IRB_LOOP(load_data(
                zreg(irb, zsrc), EVEX_compress_addr(src, irb * vlen)));
        IRB_LOOP(vaddps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, zb)));
        IRB_LOOP(vaddps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, zd)));
        IRB_LOOP(vaddps(
                zreg(irb, zdiffsrc), zreg(irb, zdiffsrc), zreg(irb, ze)));
        IRB_LOOP(vmulps(zreg(irb, zsrc), zreg(irb, zsrc), znalphabeta));

        IRB_LOOP(load_data(
                zreg(irb, zws0), EVEX_compress_addr(workspace0, irb * vlen)));
        IRB_LOOP(vdivps(
                zreg(irb, zdiffdst), zreg(irb, zdiffdst), zreg(irb, zws0)));
        IRB_LOOP(vfmadd213ps(
                zreg(irb, zdiffsrc), zreg(irb, zsrc), zreg(irb, zdiffdst)));

        Label unaligned_store, end_store;
        test(diffsrc, vlen - 1);
        jnz(unaligned_store, T_NEAR);
        IRB_LOOP(store_data(true, EVEX_compress_addr(diffsrc, irb * vlen),
                zreg(irb, zdiffsrc)));
        jmp(end_store, T_NEAR);
        L(unaligned_store);
        {
            IRB_LOOP(store_data(false, EVEX_compress_addr(diffsrc, irb * vlen),
                    zreg(irb, zdiffsrc)));
        }
        L(end_store);
    }

    jit_avx512_common_lrn_kernel_nChw16c_f(const struct nChw16c_across &J,
            float A, float B, int use_h_parallel, void *code_ptr = nullptr,
            size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , nalphabeta(-2 * A * B)
        , use_h_parallelism(use_h_parallel)
        , bf16_emu_(nullptr) {

        vlen = d_type == bf16 ? 32 : 64;
        reg_block = 3;
        src_prev_offset = vlen - 4 * sizeof(data_t);

        xmm_size = 4 * sizeof(acc_data_t);
        zmm_size = 64;
        buffer_block = xmm_size + zmm_size + xmm_size;
        buffer_nest_offset = xmm_size + zmm_size;

        if (d_type == bf16 && !mayiuse(avx512_core_bf16)) {
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_scratch,
                    bf16_emu_reserv_4);
            bf16_emu_->init_vcvtneps2bf16();
        }

        this->preamble();

#define GET_OFF(field) offsetof(jit_args_bwd_t, field)
        mov(src, ptr[param + GET_OFF(src)]);
        mov(diffdst, ptr[param + GET_OFF(diff_dst)]);
        mov(workspace0, ptr[param + GET_OFF(ws0)]);
        mov(workspace1, ptr[param + GET_OFF(ws1)]);
        mov(diffsrc, ptr[param + GET_OFF(diff_src)]);
#undef GET_OFF

        W = J.W;
        HW = J.H * J.W;
        int LSB = this->use_h_parallelism ? W : HW;

        sub(t, reg_block * buffer_block);
        mov(imm_addr64, float2int(this->nalphabeta));
        movq(xnalphabeta, imm_addr64);
        vbroadcastss(znalphabeta, xnalphabeta);

        version = J.version;

        if (version == fwd_across_version::First
                || version == fwd_across_version::Single) {
            vpxorq(xmm1, xmm1, xmm1);
            for (int irb = 0; irb < reg_block; irb++) {
                vmovups(ptr[t + irb * buffer_block], xmm1);
            }
        }
        if (version == fwd_across_version::Last
                || version == fwd_across_version::Single) {
            vpxorq(xmm1, xmm1, xmm1);
            for (int irb = 0; irb < reg_block; irb++) {
                vmovups(ptr[t + irb * buffer_block + buffer_nest_offset], xmm1);
            }
        }

        int LSREST = LSB % reg_block;
        int LS = LSB - LSREST;

        Label lrn_loop;

        if (LS > 0) {
            mov(hw, LS);

            L(lrn_loop);
            {
                compute_loop(reg_block, 1, 1);

                add(src, reg_block * vlen);
                add(diffsrc, reg_block * vlen);
                add(diffdst, reg_block * vlen);
                add(workspace0, reg_block * vlen);
                add(workspace1, reg_block * vlen);

                for (int irb = 0; irb < reg_block; irb++)
                    dec(hw);
                cmp(hw, 0);
                jne(lrn_loop, T_NEAR);
            }
        }

        compute_loop(LSREST, 1, this->use_h_parallelism ? 0 : 1);

        add(t, reg_block * buffer_block);
        this->postamble();

        ker = reinterpret_cast<decltype(ker)>(
                const_cast<uint8_t *>(this->getCode()));
    }
    ~jit_avx512_common_lrn_kernel_nChw16c_f() { delete bf16_emu_; }
};

template <data_type_t d_type>
status_t jit_avx512_common_lrn_bwd_t<d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());
    bool ok = true && mayiuse(avx512_common)
            && IMPLICATION(d_type == bf16, mayiuse(avx512_core)) && !is_fwd()
            && utils::everyone_is(d_type, data_d.data_type())
            && set_default_formats_common() && !has_zero_dim_memory()
            && data_d.ndims() == 4 && data_d.dims()[1] % vsize == 0
            && attr()->has_default_values();
    if (!ok) return unimplemented;

    dims_t ws_dims = {MB(), C(), H(), 2 * W()};
    dnnl_memory_desc_init_by_tag(
            &ws_md_, 4, ws_dims, d_type, format_tag::nChw16c);

    if (!compare_ws(hint_fwd_pd_)) return unimplemented;

    bool args_ok_across = true && desc()->alg_kind == lrn_across_channels
            && desc()->local_size == 5 && desc()->lrn_beta == 0.75
            && data_d.matches_tag(format_tag::nChw16c);

    return args_ok_across ? success : unimplemented;
}

template <data_type_t d_type>
jit_avx512_common_lrn_bwd_t<d_type>::jit_avx512_common_lrn_bwd_t(
        const pd_t *apd)
    : primitive_t(apd)
    , use_h_parallelism(0)
    , ker_(nullptr)
    , ker_first_(nullptr)
    , ker_last_(nullptr) {
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int ls = pd()->desc()->local_size;
    const float alpha = pd()->desc()->lrn_alpha / ls;
    const float beta = pd()->desc()->lrn_beta;

    use_h_parallelism = H > 28 ? 1 : 0;

    if (C / vsize == 1) {
        ker_ = new jit_avx512_common_lrn_kernel_nChw16c_f(
                nChw16c_across(H, W, fwd_across_version::Single), alpha, beta,
                use_h_parallelism);
    } else {
        ker_ = new jit_avx512_common_lrn_kernel_nChw16c_f(
                nChw16c_across(H, W, fwd_across_version::Middle), alpha, beta,
                use_h_parallelism);
        ker_first_ = new jit_avx512_common_lrn_kernel_nChw16c_f(
                nChw16c_across(H, W, fwd_across_version::First), alpha, beta,
                use_h_parallelism);
        ker_last_ = new jit_avx512_common_lrn_kernel_nChw16c_f(
                nChw16c_across(H, W, fwd_across_version::Last), alpha, beta,
                use_h_parallelism);
    }
}

template <data_type_t d_type>
jit_avx512_common_lrn_bwd_t<d_type>::~jit_avx512_common_lrn_bwd_t() {
    delete ker_;
    delete ker_first_;
    delete ker_last_;
}

template <data_type_t d_type>
void jit_avx512_common_lrn_bwd_t<d_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const data_t *, DNNL_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const int N = pd()->MB();
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();

    parallel(0, [&](const int ithr, const int nthr) {
        size_t start {0}, end {0};
        const int C16 = C / vsize;
        const size_t work_amount = use_h_parallelism ? N * C16 * H : N * C16;

        balance211(work_amount, nthr, ithr, start, end);
        if (use_h_parallelism) {
            int n {0}, c16 {0}, h {0};
            nd_iterator_init(start, n, N, h, H, c16, C16);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset
                        = n * C * H * W + c16 * H * W * vsize + h * W * vsize;
                auto ws_offset0 = n * C * H * 2 * W + c16 * H * 2 * W * vsize
                        + h * 2 * W * vsize;
                auto ws_offset1 = ws_offset0 + W * vsize;

                typename jit_avx512_common_lrn_kernel_nChw16c_f::jit_args_bwd_t
                        args;
                args.src = &src[offset];
                args.diff_dst = &diff_dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];
                args.diff_src = &diff_src[offset];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);
                nd_iterator_step(n, N, h, H, c16, C16);
            }
        } else {
            int n {0}, c16 {0};
            nd_iterator_init(start, n, N, c16, C16);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset = n * C * H * W + c16 * H * W * vsize;
                auto ws_offset0 = n * C * H * 2 * W + c16 * H * 2 * W * vsize;
                auto ws_offset1 = ws_offset0 + H * W * vsize;

                typename jit_avx512_common_lrn_kernel_nChw16c_f::jit_args_bwd_t
                        args;
                args.src = &src[offset];
                args.diff_dst = &diff_dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];
                args.diff_src = &diff_src[offset];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);

                nd_iterator_step(n, N, c16, C16);
            }
        }
    });
}

template struct jit_avx512_common_lrn_bwd_t<f32>;
template struct jit_avx512_common_lrn_bwd_t<bf16>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
