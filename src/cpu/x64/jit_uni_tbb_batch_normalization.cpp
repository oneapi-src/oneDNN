/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include <cassert>
#include <cmath>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_batch_normalization_utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_uni_tbb_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace memory_tracking::names;
using namespace Xbyak;
using acc_data_t = float;

constexpr int bits_per_byte = 8;

bool normalize_only(const batch_normalization_pd_t *pd) {
    return pd->stats_is_src() && pd->is_fwd();
}

dim_t get_c_padded(const batch_normalization_pd_t *pd) {
    return pd->src_md()->padded_dims[1];
}

template <cpu_isa_t isa>
int get_vlen(jit_memory_tag_kind_t tag_kind) {
    return isa == sse41 && tag_kind == jit_memory_tag_kind_t::blocked
            ? 32
            : cpu_isa_traits<isa>::vlen;
}

template <cpu_isa_t isa>
int get_simd_w(jit_memory_tag_kind_t tag_kind) {
    return get_vlen<isa>(tag_kind) / sizeof(acc_data_t);
}

template <cpu_isa_t isa>
bool is_avx2_ne_xf16(const batch_normalization_pd_t *pd) {
    return isa == avx2 && mayiuse(avx2_vnni_2)
            && utils::one_of(
                    pd->src_md()->data_type, data_type::bf16, data_type::f16);
}

template <cpu_isa_t isa>
std::tuple<dim_t, dim_t, dim_t> get_data_strides(
        const batch_normalization_pd_t *pd, jit_memory_tag_kind_t tag_kind) {
    const int simd_w = get_simd_w<isa>(tag_kind);
    size_t stride_N, stride_S, stride_C;

    if (tag_kind == jit_memory_tag_kind_t::nspc) {
        stride_C = static_cast<size_t>(simd_w);
        stride_S = static_cast<size_t>(pd->C());
        stride_N = static_cast<size_t>(pd->D() * pd->H() * pd->W()) * stride_S;
    } else {
        const size_t C_blks = static_cast<size_t>(get_c_padded(pd) / simd_w);

        stride_C = static_cast<size_t>(pd->D() * pd->H() * pd->W() * simd_w);
        stride_S = static_cast<size_t>(simd_w);
        stride_N = C_blks * stride_C;
    }

    return std::make_tuple(stride_N, stride_S, stride_C);
}

#define PARAM_ADDR(x) (reg_param_ + offsetof(call_params_t, x))
template <cpu_isa_t isa>
struct jit_bnorm_process_tail_t {
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    jit_bnorm_process_tail_t(const batch_normalization_pd_t *pd,
            jit_generator *host, Reg64 reg_tmp, Reg64 reg_blk_has_tail,
            Reg64 reg_C, Vmm vtail_mask, Opmask ktail_mask)
        : h_(host)
        , reg_tmp_(reg_tmp)
        , reg_blk_has_tail_(reg_blk_has_tail)
        , reg_C_(reg_C)
        , vtail_mask_(vtail_mask)
        , ktail_mask_(ktail_mask) {
        const memory_desc_wrapper data_d(pd->src_md());
        c_is_padded_ = pd->C() != data_d.padded_dims()[1];

        const int vlen = isa == sse41 ? 32 : cpu_isa_traits<isa>::vlen;
        tail_ = pd->C() % (int)(vlen / sizeof(float));
    }

    jit_generator *const h_;
    const Reg64 reg_tmp_;
    const Reg64 reg_blk_has_tail_;
    const Reg64 reg_C_;
    const Vmm vtail_mask_;
    const Opmask ktail_mask_;
    bool c_is_padded_;
    int tail_;

    void prepare_tail_mask_avx512_common() {
        if (!c_is_padded_) return;

        const int mask = (1 << tail_) - 1;

        Reg32 regw_tmp = reg_tmp_.cvt32();
        h_->mov(regw_tmp, mask);
        h_->kmovw(ktail_mask_, regw_tmp);
    }

    void prepare_tail_mask_avx2_common() {
        if (!c_is_padded_) return;

        static const uint32_t mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                0, 0, 0, 0, 0, 0, 0};

        h_->mov(reg_tmp_, reinterpret_cast<size_t>(&mask[8 - tail_]));
        h_->vmovups(vtail_mask_, h_->ptr[reg_tmp_]);
    }

    void prepare_tail() {
        if (isa == avx512_core)
            prepare_tail_mask_avx512_common();
        else if (isa == avx2)
            prepare_tail_mask_avx2_common();
    }

    void uni_vmovups_tail_avx2_common(
            const Operand &dst, const Operand &src, Label &l_ret) {
        if (dst.isMEM()) {
            h_->vmaskmovps(dst.getAddress(), vtail_mask_, Vmm(src.getIdx()));
        } else {
            h_->vmaskmovps(Vmm(dst.getIdx()), vtail_mask_, src.getAddress());
        }
        h_->jmp(l_ret);
    }

    void uni_vmovups_tail_avx512_common(
            const Operand &dst, const Operand &src, Label &l_ret) {
        if (dst.isMEM())
            h_->uni_vmovups(dst.getAddress() | ktail_mask_ | h_->T_z,
                    Vmm(src.getIdx()));
        else
            h_->uni_vmovups(Vmm(dst.getIdx()) | ktail_mask_ | h_->T_z,
                    src.getAddress());

        h_->jmp(l_ret);
    }

    void uni_vmovups_maybe_tail(const Operand &dst, const Operand &src) {
        Label l_no_mask, l_ret;
        if (c_is_padded_) {
            h_->cmp(reg_blk_has_tail_, 0);
            h_->jz(l_no_mask);

            h_->cmp(reg_C_, 1);
            h_->jne(l_no_mask);
            assert(isa == avx512_core || isa == avx2);
            if (isa == avx512_core)
                uni_vmovups_tail_avx512_common(dst, src, l_ret);
            else if (isa == avx2)
                uni_vmovups_tail_avx2_common(dst, src, l_ret);
        }
        h_->L(l_no_mask);
        if (dst.isMEM())
            h_->uni_vmovups(dst.getAddress(), Vmm(src.getIdx()));
        else
            h_->uni_vmovups(Vmm(dst.getIdx()), src.getAddress());

        h_->L(l_ret);
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_process_relu_t {
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    jit_bnorm_process_relu_t(const batch_normalization_pd_t *pd,
            jit_generator *host, Reg64 reg_off_dat, Reg64 reg_tmp,
            Reg64 reg_ptr_ws, Vmm vzero, Vmm vstore_mask, Opmask kstore_mask,
            Vmm valpha, Vmm vmask, Reg64 reg_alpha)
        : h_(host)
        , reg_off_dat_(reg_off_dat)
        , reg_tmp_(reg_tmp)
        , reg_ptr_ws_(reg_ptr_ws)
        , reg_alpha(reg_alpha)
        , vzero_(vzero)
        , vstore_mask_(vstore_mask)
        , kstore_mask_(kstore_mask)
        , valpha(valpha)
        , vmask(vmask)
        , with_relu_(pd->with_relu_post_op(pd->is_training())
                  || pd->fuse_norm_relu())
        , with_relu_inf_only_(
                  with_relu_ && !(pd->fuse_norm_relu() && pd->is_training()))
        , bit_shift_(static_cast<int>(log2(bits_per_byte
                  * types::data_type_size(pd->src_md()->data_type))))
        , alpha(with_relu_inf_only_ && pd->with_relu_post_op(pd->is_training())
                          ? pd->alpha()
                          : 0.f) {}

    jit_bnorm_process_relu_t(const batch_normalization_pd_t *pd,
            jit_generator *host, Reg64 reg_off_dat, Reg64 reg_tmp,
            Reg64 reg_ptr_ws, Vmm vzero, Vmm vstore_mask, Opmask kstore_mask)
        : jit_bnorm_process_relu_t(pd, host, reg_off_dat, reg_tmp, reg_ptr_ws,
                vzero, vstore_mask, kstore_mask, Vmm(), Vmm(), Reg64()) {}

    jit_generator *const h_;
    const Reg64 reg_off_dat_;
    const Reg64 reg_tmp_;
    const Reg64 reg_ptr_ws_;
    const Reg64 reg_alpha;
    const Vmm vzero_, vstore_mask_;
    const Opmask kstore_mask_;
    // used for ReLU computation
    const Vmm valpha;
    const Vmm vmask; // used for AVX2 and SSE41
    Label l_relu_mask_avx2_;
    const bool with_relu_, with_relu_inf_only_;
    const int bit_shift_;
    const float alpha;

    bool with_relu() const { return with_relu_; }

    bool with_relu_inf_only() const { return with_relu_inf_only_; }

    void fwd_prepare_relu() {
        if (with_relu_) { h_->uni_vpxor(vzero_, vzero_, vzero_); }
        if (with_relu_inf_only_ && alpha != 0)
            h_->mov(reg_alpha, float2int(alpha));
    }

    void bwd_prepare_relu() {
        if (with_relu_) {
            h_->uni_vpxor(vzero_, vzero_, vzero_);
            if (isa == avx2) prepare_l_relu_mask_avx2();
        }
    }

    void prepare_l_relu_mask_avx2() {
        Label l_mask_after;
        h_->jmp(l_mask_after);
        h_->align(32);
        h_->L(l_relu_mask_avx2_); /* [0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01] */
        for (int i = 0; i < 8; ++i)
            h_->dd(1 << i);
        h_->L(l_mask_after);
    }

    void fwd_process_relu(Vmm v, const int off = 0) {
        if (with_relu_inf_only_) {
            if (alpha != 0.f)
                fwd_process_relu_alpha(v);
            else
                h_->uni_vmaxps(v, v, vzero_);
        } else if (with_relu_) {
            if (isa == avx512_core)
                fwd_process_relu_avx512_common(v, off);
            else if (isa == avx2)
                fwd_process_relu_avx2(v, off);
            else
                assert(false);
        }
    }

    void bwd_process_relu(Vmm v, const int off = 0) {
        if (with_relu_) {
            if (isa == avx512_core)
                bwd_process_relu_avx512_common(v, off);
            else if (isa == avx2)
                bwd_process_relu_avx2(v, off);
            else
                assert(false);
        }
    }

    void fwd_process_relu_avx2(Vmm vdst, const int off = 0) {
        Reg64 reg_store_mask = reg_tmp_;
        h_->shr(reg_off_dat_, bit_shift_);
        h_->vcmpps(vstore_mask_, vzero_, vdst, jit_generator::_cmp_lt_os);
        h_->vmovmskps(reg_store_mask, vstore_mask_);
        h_->mov(h_->ptr[reg_ptr_ws_ + reg_off_dat_ + off],
                reg_store_mask.cvt8());
        h_->vblendvps(vdst, vzero_, vdst, vstore_mask_);
        h_->shl(reg_off_dat_, bit_shift_);
    }

    void fwd_process_relu_avx512_common(Vmm vdst, const int off = 0) {
        h_->shr(reg_off_dat_, bit_shift_);
        h_->vcmpps(kstore_mask_, vzero_, vdst, jit_generator::_cmp_lt_os);
        h_->kmovw(h_->ptr[reg_ptr_ws_ + reg_off_dat_ + off], kstore_mask_);
        h_->vblendmps(vdst | kstore_mask_, vzero_, vdst);
        h_->shl(reg_off_dat_, bit_shift_);
    }

    void fwd_process_relu_alpha(Vmm vmm_dst) {
        if (isa == avx512_core)
            fwd_process_relu_alpha_avx512_common(vmm_dst);
        else {
            assert(utils::one_of(isa, avx2, sse41));
            fwd_process_relu_alpha_avx2(vmm_dst);
        }
    }

    void fwd_process_relu_alpha_avx512_common(Vmm vmm_dst) {
        const Xmm xmm_aux = Xmm(valpha.getIdx());
        h_->vmovq(xmm_aux, reg_alpha);
        h_->vbroadcastss(valpha, xmm_aux);
        h_->vcmpps(kstore_mask_, vzero_, vmm_dst, jit_generator::_cmp_lt_os);
        h_->vmulps(valpha, vmm_dst, valpha);
        h_->vblendmps(vmm_dst | kstore_mask_, valpha, vmm_dst);
    }

    void fwd_process_relu_alpha_avx2(Vmm vmm_dst) {
        const Xmm xmm_aux = Xmm(valpha.getIdx());
        h_->uni_vpxor(vmask, vmask, vmask);
        h_->uni_vmovq(xmm_aux, reg_alpha);
        h_->uni_vbroadcastss(valpha, xmm_aux);
        h_->uni_vcmpps(vmask, vmm_dst, vzero_, jit_generator::_cmp_lt_os);
        h_->uni_vmulps(valpha, valpha, vmm_dst);
        h_->uni_vblendvps(
                vmm_dst, vmm_dst, valpha, vmask); // swaped aux and dst
    }

    void bwd_process_relu_avx2(Vmm vdiff_dst, const int off = 0) {
        h_->shr(reg_off_dat_, bit_shift_);
        h_->vpbroadcastb(
                vstore_mask_, h_->ptr[reg_ptr_ws_ + reg_off_dat_ + off]);
        h_->vpand(vstore_mask_, vstore_mask_,
                h_->ptr[Xbyak::util::rip + l_relu_mask_avx2_]);
        h_->vpcmpeqd(vstore_mask_, vstore_mask_,
                h_->ptr[Xbyak::util::rip + l_relu_mask_avx2_]);
        h_->vblendvps(vdiff_dst, vzero_, vdiff_dst, vstore_mask_);
        h_->shl(reg_off_dat_, bit_shift_);
    }

    void bwd_process_relu_avx512_common(Vmm vdiff_dst, const int off = 0) {
        h_->shr(reg_off_dat_, bit_shift_);
        h_->kmovw(kstore_mask_, h_->ptr[reg_ptr_ws_ + reg_off_dat_ + off]);
        h_->vmovups(vdiff_dst | kstore_mask_ | h_->T_z, vdiff_dst);
        h_->shl(reg_off_dat_, bit_shift_);
    }
};

template <cpu_isa_t isa>
struct helper_vmovups_data_t {
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    helper_vmovups_data_t(const batch_normalization_pd_t *pd,
            jit_generator *host, Zmm zmm_reserved_1, Zmm zmm_reserved_2,
            Zmm zmm_reserved_3, Zmm zmm_reserved_4, Reg64 reg_tmp)
        : h_(host)
        , bf16_emu_(nullptr)
        , is_bf16_(pd->src_md()->data_type == data_type::bf16)
        , is_f16_(pd->src_md()->data_type == data_type::f16) {

        if (is_bf16_ && isa == avx512_core && !mayiuse(avx512_core_bf16)) {
            bf16_emu_ = utils::make_unique<bf16_emulation_t>(h_, zmm_reserved_1,
                    zmm_reserved_2, zmm_reserved_3, reg_tmp, zmm_reserved_4,
                    zmm_reserved_4);
        }
    }

    jit_generator *const h_;
    std::unique_ptr<bf16_emulation_t> bf16_emu_;
    bool is_bf16_;
    bool is_f16_;

    void merge_interleaved_to_plain(const Vmm &vmm_even, const Vmm &vmm_odd,
            const Vmm &vmm_aux0) const {
        Ymm ymm_even = Ymm(vmm_even.getIdx());
        Ymm ymm_odd = Ymm(vmm_odd.getIdx());
        Ymm ymm_aux0 = Ymm(vmm_aux0.getIdx());
        Ymm ymm_aux1 = ymm_odd;

        h_->vpunpckldq(ymm_aux0, ymm_even, ymm_odd);
        h_->vpunpckhdq(ymm_aux1, ymm_even, ymm_odd);
        h_->vperm2i128(ymm_even, ymm_aux0, ymm_aux1, 0x20);
        h_->vperm2i128(ymm_odd, ymm_aux0, ymm_aux1, 0x31);
    }

    void operator()(const Vmm &vmm_even, const Vmm &vmm_odd,
            const Address &addr) const {
        // load two simd_w data from addr into two registers
        if (is_bf16_) {
            // convert bf16 input to f32
            h_->vcvtneebf162ps(vmm_even, addr);
            h_->vcvtneobf162ps(vmm_odd, addr);
        } else if (is_f16_) {
            h_->vcvtneeph2ps(vmm_even, addr);
            h_->vcvtneoph2ps(vmm_odd, addr);
        } else
            assert(!"unsupported data type");
    }

    void operator()(const Operand &dst, const Operand &src) const {
        if (dst.isMEM()) {
            if (is_bf16_) {
                constexpr bool isAvx2 = isa == avx2;
                const typename std::conditional<isAvx2, Xmm, Ymm>::type
                        dst_reg {src.getIdx()};
                const typename std::conditional<isAvx2, Ymm, Zmm>::type
                        src_reg {src.getIdx()};

                // convert f32 output to bf16
                if (!bf16_emu_)
                    h_->vcvtneps2bf16(dst_reg, src_reg, h_->get_encoding());
                else
                    bf16_emu_->vcvtneps2bf16(dst_reg, src_reg);

                h_->uni_vmovups(dst.getAddress(), dst_reg);
            } else if (is_f16_) {
                auto src_reg = Vmm(src.getIdx());
                h_->vcvtps2ph(
                        dst.getAddress(), src_reg, jit_generator::_op_mxcsr);
            } else {
                h_->uni_vmovups(dst.getAddress(), Vmm(src.getIdx()));
            }
        } else {
            if (is_bf16_) {
                // convert bf16 input to f32
                h_->vpmovzxwd(Vmm(dst.getIdx()), src.getAddress());
                h_->vpslld(Vmm(dst.getIdx()), Vmm(dst.getIdx()), 0x10);
            } else if (is_f16_) {
                if (mayiuse(avx512_core_fp16))
                    h_->vcvtph2psx(Vmm(dst.getIdx()), src.getAddress());
                else
                    h_->vcvtph2ps(Vmm(dst.getIdx()), src.getAddress());
            } else {
                h_->uni_vmovups(Vmm(dst.getIdx()), src.getAddress());
            }
        }
    }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(helper_vmovups_data_t);
};

template <cpu_isa_t isa>
struct jit_bnorm_fwd_statistics_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_fwd_statistics_t)
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    const AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                      ? yword
                                                 : zword;

    struct call_params_t {
        size_t N, C, S;
        const void *src;
        const acc_data_t *mean;
        const acc_data_t *var;
        size_t blk_has_tail;
        size_t do_normalise;
    };

    const Reg64 reg_param_ = abi_param1;
    const Reg64 reg_tmp_ = abi_not_param1;
    const Reg64 reg_N_ = rsi;
    const Reg64 reg_S_ = rax;
    const Reg64 reg_C_ = rdx;
    const Reg64 reg_off_c_ = rbx;
    const Reg64 reg_blk_has_tail_ = rbp;

    const Reg64 reg_off_dat_ = r8;
    const Reg64 reg_off_dat_save_ = r9;
    const Reg64 reg_ptr_mean_ = r10;
    const Reg64 reg_ptr_var_ = r11;
    const Reg64 reg_ptr_src_ = r12;
    const Reg64 reg_do_normalise_ = r13;
    const Reg64 reg_ptr_stat_ = r14;

    const Vmm v_ = Vmm(0);
    const Vmm vtmp_ = Vmm(1);
    const Vmm vtail_mask_ = Vmm(2);
    const Vmm vNS_ = Vmm(3);
    const Vmm vzero_ = Vmm(4);
    const Vmm vsrc_aux = Vmm(2); //use for xf16 nspc on AVX2
    // When variance is computed then two vmms(one for variance and
    // one for mean) are needed to unroll one c block at any moment,
    // therefore the number of registers which are used to unrolling
    // must to be divisible by two.
    static constexpr int min_idx_to_unroll_ = 4;
    static constexpr int max_idx_to_unroll_ = isa == avx512_core ? 28 : 16;
    static constexpr int number_of_vmms_to_unrolling_variables_
            = max_idx_to_unroll_ - min_idx_to_unroll_;
    static_assert(number_of_vmms_to_unrolling_variables_ % 2 == 0
                    && number_of_vmms_to_unrolling_variables_ != 0,
            "Number of register to unrolling must to be divisible by 2.");

    const Opmask ktail_mask_ = k2;

    const batch_normalization_pd_t *pd_;
    const jit_memory_tag_kind_t tag_kind_;
    const int vlen;
    const int simd_w;
    const bool is_avx2_ne_xf16_;
    jit_bnorm_process_tail_t<isa> jit_tail_;
    helper_vmovups_data_t<isa> helper_vmovups_;
    int stride_N_, stride_S_, stride_C_;
    size_t data_type_size_, acc_type_size_;

    void load_common_params() {
#define PARAM_PTR(x) ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src_, PARAM_PTR(src));
        mov(reg_ptr_mean_, PARAM_PTR(mean));
        mov(reg_ptr_var_, PARAM_PTR(var));
#undef PARAM_PTR
        mov(reg_blk_has_tail_, dword[PARAM_ADDR(blk_has_tail)]);
        mov(reg_do_normalise_, dword[PARAM_ADDR(do_normalise)]);
    }

    void zeroise() {
        Label label_zeroise;
        xor_(reg_off_c_, reg_off_c_);
        uni_vpxor(vzero_, vzero_, vzero_);
        mov(reg_C_, dword[PARAM_ADDR(C)]);
        L(label_zeroise);
        {
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_stat_ + reg_off_c_], vzero_);
            if (isa == sse41 && tag_kind_ == jit_memory_tag_kind_t::blocked) {
                jit_tail_.uni_vmovups_maybe_tail(
                        vmmword[reg_ptr_stat_ + reg_off_c_ + vlen / 2], vzero_);
            }
            add(reg_off_c_, simd_w * acc_type_size_);
            dec(reg_C_);
            jnz(label_zeroise);
        }
    }

    void load_stat(bool compute_mean, const int c_blks_to_unroll = 1) {
        int start_idx = min_idx_to_unroll_;
        int end_idx = c_blks_to_unroll + min_idx_to_unroll_;
        const int step = simd_w * acc_type_size_;

        // load mean or variance
        for (int idx = start_idx, off = 0; idx < end_idx; idx++, off += step) {
            const Vmm vstat = Vmm(idx);
            jit_tail_.uni_vmovups_maybe_tail(
                    vstat, vmmword[reg_ptr_stat_ + reg_off_c_ + off]);
        }

        // if variance is counted then mean also is needed
        if (!compute_mean) {
            start_idx = min_idx_to_unroll_ + c_blks_to_unroll;
            end_idx = min_idx_to_unroll_ + 2 * c_blks_to_unroll;

            for (int idx = start_idx, off = 0; idx < end_idx;
                    idx++, off += step) {
                const Vmm vmean = Vmm(idx);
                jit_tail_.uni_vmovups_maybe_tail(
                        vmean, vmmword[reg_ptr_mean_ + reg_off_c_ + off]);
            }
        }
    }

    void compute_stat(bool compute_mean, const int c_blks_to_unroll = 1) {
        const int start_idx = min_idx_to_unroll_;
        const int end_idx = c_blks_to_unroll + min_idx_to_unroll_;
        const int step = simd_w * data_type_size_;

        for (int idx = start_idx, off = 0; idx < end_idx; idx++, off += step) {
            const Vmm vstat = Vmm(idx);

            helper_vmovups_(v_, vmmword[reg_ptr_src_ + reg_off_dat_ + off]);

            if (compute_mean) {
                uni_vaddps(vstat, vstat, v_);
            } else {
                const Vmm vmean = Vmm(idx + c_blks_to_unroll);

                // var += (src - mean)^2
                uni_vsubps(vtmp_, v_, vmean, vtmp_);
                uni_vfmadd231ps(vstat, vtmp_, vtmp_);
            }
        }
    }

    void compute_stat_avx2_ne_xf16(
            bool compute_mean, const int c_blks_to_unroll = 1) {
        const int start_idx = min_idx_to_unroll_;
        const int end_idx = c_blks_to_unroll + min_idx_to_unroll_;
        const int step = simd_w * data_type_size_;

        for (int idx = start_idx, off = 0; idx < end_idx;
                idx += 2, off += 2 * step) {
            const bool is_c_blks_tail = (end_idx - idx) < 2;
            const Vmm vsrc_even = v_;
            const Vmm vsrc_odd = vsrc_aux;
            if (is_c_blks_tail)
                helper_vmovups_(
                        vsrc_even, vmmword[reg_ptr_src_ + reg_off_dat_ + off]);
            else
                helper_vmovups_(vsrc_even, vsrc_odd,
                        vmmword[reg_ptr_src_ + reg_off_dat_ + off]);
            for (int i_odd = 0; i_odd < 2 && idx + i_odd < end_idx; ++i_odd) {
                const Vmm vstat = Vmm(idx + i_odd);
                const Vmm vsrc = i_odd ? vsrc_odd : vsrc_even;
                if (compute_mean) {
                    uni_vaddps(vstat, vstat, vsrc);
                } else {
                    const Vmm vmean = Vmm(idx + i_odd + c_blks_to_unroll);
                    uni_vsubps(vtmp_, vsrc, vmean, vtmp_);
                    uni_vfmadd231ps(vstat, vtmp_, vtmp_);
                }
            }
        }
    }

    void store_stat(const int c_blks_to_unroll = 1) {
        const int start_idx = min_idx_to_unroll_;
        const int end_idx = c_blks_to_unroll + min_idx_to_unroll_;
        const int step = simd_w * acc_type_size_;

        for (int idx = start_idx, off = 0; idx < end_idx; idx++, off += step) {
            const Vmm vstat = Vmm(idx);

            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_stat_ + reg_off_c_ + off], vstat);
        }
    }

    void compute_blocked(bool compute_mean) {
        Label label_C, label_S;
        mov(reg_C_, dword[PARAM_ADDR(C)]);
        L(label_C);
        {
            mov(reg_off_dat_, reg_off_dat_save_);

            load_stat(compute_mean);

            mov(reg_S_, dword[PARAM_ADDR(S)]);
            L(label_S);
            {
                compute_stat(compute_mean);

                add(reg_off_dat_, stride_S_ * data_type_size_);

                dec(reg_S_);
                jnz(label_S);
            }

            store_stat();

            add(reg_off_dat_save_, stride_C_ * data_type_size_);
            add(reg_off_c_, simd_w * acc_type_size_);

            dec(reg_C_);
            jnz(label_C);
        }
    }

    void compute_nspc(bool compute_mean) {
        mov(reg_C_, dword[PARAM_ADDR(C)]);

        // When a variance is computed, two values are unrolled: mean and variance,
        // so number_of_vmms_to_unrolling_variables_ is divided by 2.
        const int max_of_unrolled_c_blks = compute_mean
                ? number_of_vmms_to_unrolling_variables_
                : number_of_vmms_to_unrolling_variables_ / 2;
        std::vector<Label> c_unroll_label(max_of_unrolled_c_blks + 1);

        for (int c_blks_to_unroll = max_of_unrolled_c_blks;
                c_blks_to_unroll > 0; --c_blks_to_unroll) {
            L(c_unroll_label[c_blks_to_unroll]);
            {
                cmp(reg_C_, c_blks_to_unroll);
                jl(c_unroll_label[c_blks_to_unroll - 1], T_NEAR);

                mov(reg_off_dat_, reg_off_dat_save_);

                load_stat(compute_mean, c_blks_to_unroll);

                Label label_S;
                mov(reg_S_, dword[PARAM_ADDR(S)]);
                L(label_S);
                {
                    is_avx2_ne_xf16_
                            ? compute_stat_avx2_ne_xf16(
                                    compute_mean, c_blks_to_unroll)
                            : compute_stat(compute_mean, c_blks_to_unroll);

                    add(reg_off_dat_, stride_S_ * data_type_size_);

                    dec(reg_S_);
                    jnz(label_S);
                }

                store_stat(c_blks_to_unroll);

                add(reg_off_c_, c_blks_to_unroll * simd_w * acc_type_size_);
                add(reg_off_dat_save_,
                        c_blks_to_unroll * stride_C_ * data_type_size_);

                sub(reg_C_, c_blks_to_unroll);
                jmp(c_unroll_label[c_blks_to_unroll], T_NEAR);
            }
        }
        L(c_unroll_label[0]);
    }

    void compute(bool compute_mean) {
        Label label_N;
        mov(reg_N_, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save_, reg_off_dat_save_);
            xor_(reg_off_c_, reg_off_c_);

            tag_kind_ == jit_memory_tag_kind_t::nspc
                    ? compute_nspc(compute_mean)
                    : compute_blocked(compute_mean);

            if (isa == sse41 && tag_kind_ == jit_memory_tag_kind_t::blocked) {
                xor_(reg_off_dat_save_, reg_off_dat_save_);
                xor_(reg_off_c_, reg_off_c_);
                add(reg_off_dat_save_, vlen / 2);
                add(reg_off_c_, vlen / 2);

                compute_blocked(compute_mean);
            }

            add(reg_ptr_src_, stride_N_ * data_type_size_);
            dec(reg_N_);
            jnz(label_N);
        }
    }

    void normalize() {
        Label label_ret, label_normalise;
        cmp(reg_do_normalise_, 0);
        jz(label_ret);

        const int S = pd_->D() * pd_->H() * pd_->W();
        mov(reg_tmp_, float2int(pd_->MB() * S));
        Xmm xtmp = Xmm(vtmp_.getIdx());
        uni_vmovq(xtmp, reg_tmp_);
        uni_vbroadcastss(vNS_, xtmp);

        xor_(reg_off_c_, reg_off_c_);
        mov(reg_C_, dword[PARAM_ADDR(C)]);
        L(label_normalise);
        {
            jit_tail_.uni_vmovups_maybe_tail(
                    v_, vmmword[reg_ptr_stat_ + reg_off_c_]);
            uni_vdivps(v_, v_, vNS_);
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_stat_ + reg_off_c_], v_);

            if (isa == sse41 && tag_kind_ == jit_memory_tag_kind_t::blocked) {
                jit_tail_.uni_vmovups_maybe_tail(
                        v_, vmmword[reg_ptr_stat_ + reg_off_c_ + vlen / 2]);
                uni_vdivps(v_, v_, vNS_);
                jit_tail_.uni_vmovups_maybe_tail(
                        vmmword[reg_ptr_stat_ + reg_off_c_ + vlen / 2], v_);
            }

            add(reg_off_c_, simd_w * acc_type_size_);
            dec(reg_C_);
            jnz(label_normalise);
        }

        L(label_ret);
    }

    jit_bnorm_fwd_statistics_t(const batch_normalization_pd_t *pd,
            const jit_memory_tag_kind_t tag_kind)
        : jit_generator(jit_name())
        , pd_(pd)
        , tag_kind_(tag_kind)
        , vlen(get_vlen<isa>(tag_kind))
        , simd_w(get_simd_w<isa>(tag_kind))
        , is_avx2_ne_xf16_(is_avx2_ne_xf16<isa>(pd))
        , jit_tail_(pd, this, reg_tmp_, reg_blk_has_tail_, reg_C_, vtail_mask_,
                  ktail_mask_)
        , helper_vmovups_(pd, this, zmm28, zmm29, zmm30, zmm31, reg_tmp_)
        , data_type_size_(types::data_type_size(pd->src_md()->data_type))
        , acc_type_size_(sizeof(acc_data_t)) {
        static_assert(utils::one_of(isa, sse41, avx2, avx512_core),
                "unsupported isa");

        std::tie(stride_N_, stride_S_, stride_C_)
                = get_data_strides<isa>(pd_, tag_kind);
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_fwd_mean_t : jit_bnorm_fwd_statistics_t<isa> {
    using call_params_t =
            typename jit_bnorm_fwd_statistics_t<isa>::call_params_t;

    jit_bnorm_fwd_mean_t(const batch_normalization_pd_t *pd,
            const jit_memory_tag_kind_t tag_kind)
        : jit_bnorm_fwd_statistics_t<isa>(pd, tag_kind) {}

    void generate() override {
        this->preamble();
        this->load_common_params();
        this->mov(this->reg_ptr_stat_, this->reg_ptr_mean_);
        this->jit_tail_.prepare_tail();
        this->zeroise();
        this->compute(true);
        this->normalize();
        this->postamble();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_fwd_var_t : jit_bnorm_fwd_statistics_t<isa> {
    using call_params_t =
            typename jit_bnorm_fwd_statistics_t<isa>::call_params_t;

    jit_bnorm_fwd_var_t(const batch_normalization_pd_t *pd,
            const jit_memory_tag_kind_t tag_kind)
        : jit_bnorm_fwd_statistics_t<isa>(pd, tag_kind) {}

    void generate() override {
        this->preamble();
        this->load_common_params();
        this->mov(this->reg_ptr_stat_, this->reg_ptr_var_);
        this->jit_tail_.prepare_tail();
        this->zeroise();
        this->compute(false);
        this->normalize();
        this->postamble();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_fwd_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_fwd_t)
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    const AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                      ? yword
                                                 : zword;

    struct call_params_t {
        size_t N, C, S;
        const void *src;
        void *dst;
        const uint8_t *ws;
        const acc_data_t *mean, *var;
        const acc_data_t *scale, *shift;
        size_t blk_has_tail;
    };

    const Reg64 reg_param_ = abi_param1;
    const Reg64 reg_tmp_ = abi_not_param1;
    const Reg64 reg_N_ = rsi;
    const Reg64 reg_S_ = rax;
    const Reg64 reg_C_ = rdx;
    const Reg64 reg_off_c_ = rbx;
    const Reg64 reg_blk_has_tail_ = rbp;

    const Reg64 reg_off_dat_ = r8;
    const Reg64 reg_off_dat_save_ = r9;
    const Reg64 reg_ptr_ws_ = r10;
    const Reg64 reg_ptr_scale_ = r11;
    const Reg64 reg_ptr_shift_ = reg_N_;
    const Reg64 reg_ptr_var_ = r12;
    const Reg64 reg_ptr_mean_ = r13;
    const Reg64 reg_ptr_dst_ = r14;
    const Reg64 reg_ptr_src_ = r15;
    const Reg64 reg_alpha_ = reg_ptr_ws_;

    const Vmm vmask = Vmm(0); // required for avx2 and sse41 ReLU computation
    const Vmm vone_ = Vmm(1);
    const Vmm vmean_ = Vmm(2);
    const Vmm vvar_ = Vmm(3);
    const Vmm vsqrtvar_ = Vmm(4);
    const Vmm vgamma_ = Vmm(5);
    const Vmm vbeta_ = Vmm(6);
    const Vmm veps_ = Vmm(7);
    const Vmm vtmp_ = Vmm(8);
    const Vmm v_ = Vmm(9);
    const Vmm vzero_ = Vmm(10);
    const Vmm vtail_mask_ = Vmm(11);
    const Vmm valpha = Vmm(12);
    const Vmm vsrc_aux = Vmm(13);
    const Vmm vstore_mask_ = vtmp_;
    const Vmm vmean_even_ = vmean_;
    const Vmm vmean_odd_ = Vmm(14);
    const Vmm vsqrtvar_even_ = vsqrtvar_;
    const Vmm vsqrtvar_odd_ = Vmm(15);
    const Vmm vvar_even_ = vvar_;
    const Vmm vvar_odd_ = vsrc_aux;

    const Opmask kstore_mask_ = k1;
    const Opmask ktail_mask_ = k2;

    const batch_normalization_pd_t *pd_;
    const jit_memory_tag_kind_t tag_kind_;
    const int vlen;
    const int simd_w;
    const bool is_avx2_ne_xf16_;
    jit_bnorm_process_tail_t<isa> jit_tail_;
    jit_bnorm_process_relu_t<isa> jit_relu_;
    helper_vmovups_data_t<isa> helper_vmovups_;
    int stride_N_, stride_S_, stride_C_;
    size_t data_type_size_, acc_type_size_;

    enum {
        stack_off_N = 0,
        stack_off_shift = 8,
        stack_size_required = 16,
    };

    void load_common_params() {
#define PARAM_PTR(x) ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src_, PARAM_PTR(src));
        mov(reg_ptr_dst_, PARAM_PTR(dst));
        mov(reg_ptr_mean_, PARAM_PTR(mean));
        mov(reg_ptr_var_, PARAM_PTR(var));
        mov(reg_ptr_scale_, PARAM_PTR(scale));
        if (jit_relu_.with_relu_ && !jit_relu_.with_relu_inf_only_)
            mov(reg_ptr_ws_, PARAM_PTR(ws));

        Xmm x = Xmm(v_.getIdx());

        mov(reg_tmp_, float2int(pd_->desc()->batch_norm_epsilon));
        uni_vmovq(x, reg_tmp_);
        uni_vbroadcastss(veps_, x);

        mov(reg_tmp_, float2int(1.f));
        uni_vmovq(x, reg_tmp_);
        uni_vbroadcastss(vone_, x);

        mov(reg_blk_has_tail_, dword[PARAM_ADDR(blk_has_tail)]);

        mov(reg_tmp_, PARAM_PTR(shift));
        mov(ptr[rsp + stack_off_shift], reg_tmp_);
        mov(reg_tmp_, PARAM_PTR(N));
        mov(ptr[rsp + stack_off_N], reg_tmp_);
#undef PARAM_PTR
    }

    void load_c_specifics(
            const bool has_load_mean_sqrtvar, const int offt = 0) {
        if (!has_load_mean_sqrtvar) {
            jit_tail_.uni_vmovups_maybe_tail(
                    vmean_, vmmword[reg_ptr_mean_ + reg_off_c_ + offt]);
            jit_tail_.uni_vmovups_maybe_tail(
                    vvar_, vmmword[reg_ptr_var_ + reg_off_c_ + offt]);

            uni_vmovups(vsqrtvar_, vvar_);
            uni_vaddps(vsqrtvar_, vsqrtvar_, veps_);
            uni_vsqrtps(vsqrtvar_, vsqrtvar_);
            if (isa == sse41) {
                movups(vtmp_, vone_);
                divps(vtmp_, vsqrtvar_);
                movups(vsqrtvar_, vtmp_);
            } else
                vdivps(vsqrtvar_, vone_, vsqrtvar_);
        }

        if (pd_->use_scale())
            jit_tail_.uni_vmovups_maybe_tail(
                    vgamma_, vmmword[reg_ptr_scale_ + reg_off_c_ + offt]);
        if (pd_->use_shift())
            jit_tail_.uni_vmovups_maybe_tail(
                    vbeta_, vmmword[reg_ptr_shift_ + reg_off_c_ + offt]);
    }

    void compute_bnorm(const Vmm &v, const Vmm &vmean, const Vmm &vsqrtvar,
            bool stream_store_allowed, bool has_load_src, const int offt = 0) {
        if (!has_load_src)
            helper_vmovups_(v, vmmword[reg_ptr_src_ + reg_off_dat_ + offt]);
        uni_vsubps(v, v, vmean);
        uni_vmulps(v, v, vsqrtvar);

        if (pd_->use_scale() && pd_->use_shift())
            uni_vfmadd213ps(v, vgamma_, vbeta_);
        else if (pd_->use_scale())
            uni_vmulps(v, v, vgamma_);
        else if (pd_->use_shift())
            uni_vaddps(v, v, vbeta_);

        jit_relu_.fwd_process_relu(v);

        if (stream_store_allowed) {
            uni_vmovntps(vmmword[reg_ptr_dst_ + reg_off_dat_ + offt], v);
        } else {
            helper_vmovups_(vmmword[reg_ptr_dst_ + reg_off_dat_ + offt], v);
        }
    }

    void load_two_c_mean_sqrtvar() {
        const int offt = simd_w * acc_type_size_;
        jit_tail_.uni_vmovups_maybe_tail(
                vmean_even_, vmmword[reg_ptr_mean_ + reg_off_c_]);
        jit_tail_.uni_vmovups_maybe_tail(
                vmean_odd_, vmmword[reg_ptr_mean_ + reg_off_c_ + offt]);
        jit_tail_.uni_vmovups_maybe_tail(
                vvar_even_, vmmword[reg_ptr_var_ + reg_off_c_]);
        jit_tail_.uni_vmovups_maybe_tail(
                vvar_odd_, vmmword[reg_ptr_var_ + reg_off_c_ + offt]);

        // merge mean and variance in interleave to plain layout when needed
        if (!pd_->stats_is_src()) {
            helper_vmovups_.merge_interleaved_to_plain(
                    vmean_even_, vmean_odd_, vtmp_);
            helper_vmovups_.merge_interleaved_to_plain(
                    vvar_even_, vvar_odd_, vtmp_);
        }
        uni_vmovups(vsqrtvar_even_, vvar_even_);
        uni_vaddps(vsqrtvar_even_, vsqrtvar_even_, veps_);
        uni_vsqrtps(vsqrtvar_even_, vsqrtvar_even_);
        vdivps(vsqrtvar_even_, vone_, vsqrtvar_even_);

        uni_vmovups(vsqrtvar_odd_, vvar_odd_);
        uni_vaddps(vsqrtvar_odd_, vsqrtvar_odd_, veps_);
        uni_vsqrtps(vsqrtvar_odd_, vsqrtvar_odd_);
        vdivps(vsqrtvar_odd_, vone_, vsqrtvar_odd_);
    }

    void compute_bnorm_avx2_ne_xf16(
            const bool is_c_blks_tail, bool stream_store_allowed) {
        const Vmm vsrc_even = v_;
        const Vmm vsrc_odd = vsrc_aux;
        if (is_c_blks_tail) {
            compute_bnorm(
                    vsrc_even, vmean_, vsqrtvar_, stream_store_allowed, false);
        } else {
            helper_vmovups_(
                    vsrc_even, vsrc_odd, vmmword[reg_ptr_src_ + reg_off_dat_]);
            helper_vmovups_.merge_interleaved_to_plain(
                    vsrc_even, vsrc_odd, vtmp_);
            load_c_specifics(true);
            compute_bnorm(vsrc_even, vmean_even_, vsqrtvar_even_,
                    stream_store_allowed, true);

            load_c_specifics(true, simd_w * acc_type_size_);
            compute_bnorm(vsrc_odd, vmean_odd_, vsqrtvar_odd_,
                    stream_store_allowed, true, stride_C_ * data_type_size_);
        }
    }

    void compute_avx2_ne_xf16(bool stream_store_allowed) {
        Label label_C, label_S, label_C_tail, label_C_end, label_S_C_tail;
        mov(reg_C_, dword[PARAM_ADDR(C)]);
        L(label_C);
        {
            cmp(reg_C_, 1);
            jle(label_C_tail, T_NEAR);

            mov(reg_off_dat_, reg_off_dat_save_);
            load_two_c_mean_sqrtvar();

            mov(reg_S_, dword[PARAM_ADDR(S)]);
            L(label_S);
            {
                compute_bnorm_avx2_ne_xf16(false, stream_store_allowed);

                add(reg_off_dat_, stride_S_ * data_type_size_);
                dec(reg_S_);
                jnz(label_S, T_NEAR);
            }
            add(reg_off_dat_save_, 2 * stride_C_ * data_type_size_);
            add(reg_off_c_, 2 * simd_w * acc_type_size_);

            sub(reg_C_, 2);
            jnz(label_C, T_NEAR);
        }

        L(label_C_tail);
        {
            cmp(reg_C_, 0);
            jz(label_C_end, T_NEAR);

            mov(reg_off_dat_, reg_off_dat_save_);
            load_c_specifics(false);

            mov(reg_S_, dword[PARAM_ADDR(S)]);
            L(label_S_C_tail);
            {
                compute_bnorm_avx2_ne_xf16(true, stream_store_allowed);

                add(reg_off_dat_, stride_S_ * data_type_size_);
                dec(reg_S_);
                jnz(label_S_C_tail, T_NEAR);
            }
        }
        L(label_C_end);
    }

    void compute_blocked(bool stream_store_allowed) {
        Label label_C, label_S;
        mov(reg_C_, dword[PARAM_ADDR(C)]);
        L(label_C);
        {
            mov(reg_off_dat_, reg_off_dat_save_);

            load_c_specifics(false);

            mov(reg_S_, dword[PARAM_ADDR(S)]);
            L(label_S);
            {
                compute_bnorm(
                        v_, vmean_, vsqrtvar_, stream_store_allowed, false);

                add(reg_off_dat_, stride_S_ * data_type_size_);

                dec(reg_S_);
                jnz(label_S);
            }

            add(reg_off_dat_save_, stride_C_ * data_type_size_);
            add(reg_off_c_, simd_w * acc_type_size_);

            dec(reg_C_);
            jnz(label_C);
        }
    }

    void compute(bool stream_store_allowed) {
        Label label_N;
        mov(reg_N_, ptr[rsp + stack_off_N]);
        L(label_N);
        {
            // save reg_N_, because register is shared with reg_ptr_shift_
            mov(ptr[rsp + stack_off_N], reg_N_);
            mov(reg_ptr_shift_, ptr[rsp + stack_off_shift]);

            xor_(reg_off_dat_save_, reg_off_dat_save_);
            xor_(reg_off_c_, reg_off_c_);

            is_avx2_ne_xf16_ ? compute_avx2_ne_xf16(stream_store_allowed)
                             : compute_blocked(stream_store_allowed);

            if (isa == sse41 && tag_kind_ == jit_memory_tag_kind_t::blocked) {
                xor_(reg_off_dat_save_, reg_off_dat_save_);
                xor_(reg_off_c_, reg_off_c_);
                add(reg_off_dat_save_, vlen / 2);
                add(reg_off_c_, vlen / 2);

                compute_blocked(stream_store_allowed);
            }

            add(reg_ptr_src_, stride_N_ * data_type_size_);
            add(reg_ptr_dst_, stride_N_ * data_type_size_);
            if (jit_relu_.with_relu_ && !jit_relu_.with_relu_inf_only_)
                add(reg_ptr_ws_, stride_N_ / bits_per_byte);

            // restore reg_N_, because register is shared with reg_ptr_shift_
            mov(reg_N_, ptr[rsp + stack_off_N]);
            dec(reg_N_);
            jnz(label_N);
        }
    }

    jit_bnorm_fwd_t(const batch_normalization_pd_t *pd,
            const jit_memory_tag_kind_t tag_kind)
        : jit_generator(jit_name())
        , pd_(pd)
        , tag_kind_(tag_kind)
        , vlen(get_vlen<isa>(tag_kind))
        , simd_w(get_simd_w<isa>(tag_kind))
        , is_avx2_ne_xf16_(is_avx2_ne_xf16<isa>(pd))
        , jit_tail_(pd, this, reg_tmp_, reg_blk_has_tail_, reg_C_, vtail_mask_,
                  ktail_mask_)
        , jit_relu_(pd, this, reg_off_dat_, reg_tmp_, reg_ptr_ws_, vzero_,
                  vstore_mask_, kstore_mask_, valpha, vmask, reg_alpha_)
        , helper_vmovups_(pd, this, zmm28, zmm29, zmm30, zmm31, reg_tmp_)
        , data_type_size_(types::data_type_size(pd->src_md()->data_type))
        , acc_type_size_(sizeof(acc_data_t)) {
        static_assert(utils::one_of(isa, sse41, avx2, avx512_core),
                "unsupported isa");

        std::tie(stride_N_, stride_S_, stride_C_)
                = get_data_strides<isa>(pd_, tag_kind);
    }

    void generate() override {
        bool is_xf16 = utils::one_of(
                pd_->src_md()->data_type, data_type::bf16, data_type::f16);
        const bool is_tail_in_nspc_format
                = tag_kind_ == jit_memory_tag_kind_t::nspc
                && jit_tail_.tail_ != 0;
        const bool stream_store_allowed = !is_xf16 && !is_tail_in_nspc_format;

        preamble();
        if (helper_vmovups_.bf16_emu_)
            helper_vmovups_.bf16_emu_->init_vcvtneps2bf16();
        sub(rsp, stack_size_required);
        load_common_params();
        jit_relu_.fwd_prepare_relu();
        jit_tail_.prepare_tail();

        Label normal_store, end_store;
        test(reg_ptr_dst_, vlen - 1);
        jnz(normal_store, T_NEAR);
        compute(stream_store_allowed);
        jmp(end_store, T_NEAR);
        L(normal_store);
        { compute(false); }
        L(end_store);

        add(rsp, stack_size_required);
        postamble();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_bwd_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_bwd_t)
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    const AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                      ? yword
                                                 : zword;

    struct call_params_t {
        size_t N, C, S;
        const void *src, *diff_src, *diff_dst;
        const uint8_t *ws;
        const acc_data_t *mean, *var;
        const acc_data_t *scale, *diff_scale, *diff_shift;
        size_t blk_has_tail;
    };

    const Reg64 reg_param_ = abi_param1;
    const Reg64 reg_tmp_ = abi_not_param1;
    const Reg64 reg_N_ = rsi;
    const Reg64 reg_S_ = rax;
    const Reg64 reg_C_ = rdx;
    const Reg64 reg_off_c_ = rbx;
    const Reg64 reg_blk_has_tail_ = rbp;

    const Reg64 reg_off_dat_ = r8;
    const Reg64 reg_off_dat_save_ = r9;
    const Reg64 reg_ptr_c_ = r10;
    const Reg64 reg_ptr_ws_ = r11;
    const Reg64 reg_ptr_diff_dst_ = r12;
    const Reg64 reg_ptr_diff_src_ = r13;
    const Reg64 reg_ptr_src_ = r14;

    const Vmm vzero_ = Vmm(0);
    const Vmm vone_ = Vmm(1);
    const Vmm vmean_ = Vmm(2);
    const Vmm vsqrtvar_ = Vmm(3);
    const Vmm vgamma_ = Vmm(4);
    const Vmm vdiff_gamma_ = Vmm(5);
    const Vmm vdiff_beta_ = Vmm(6);
    const Vmm veps_ = Vmm(7);
    const Vmm vNS_ = Vmm(8);
    const Vmm vtmp_ = Vmm(9);
    const Vmm v_ = Vmm(10);
    const Vmm vtail_mask_ = Vmm(11);
    const Vmm vstore_mask_ = vtmp_;

    const Opmask kstore_mask_ = k1;
    const Opmask ktail_mask_ = k2;

    const batch_normalization_pd_t *pd_;
    const jit_memory_tag_kind_t tag_kind_;
    const int vlen;
    const int simd_w;
    jit_bnorm_process_tail_t<isa> jit_tail_;
    jit_bnorm_process_relu_t<isa> jit_relu_;
    helper_vmovups_data_t<isa> helper_vmovups_;
    int stride_N_, stride_S_, stride_C_;
    size_t data_type_size_, acc_type_size_;

    void load_common_params() {
#define PARAM_PTR(x) ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src_, PARAM_PTR(src));
        mov(reg_ptr_diff_src_, PARAM_PTR(diff_src));
        mov(reg_ptr_diff_dst_, PARAM_PTR(diff_dst));
        mov(reg_ptr_ws_, PARAM_PTR(ws));
#undef PARAM_PTR

        Xmm x = Xmm(v_.getIdx());

        mov(reg_tmp_, float2int(pd_->desc()->batch_norm_epsilon));
        uni_vmovq(x, reg_tmp_);
        uni_vbroadcastss(veps_, x);

        mov(reg_tmp_, float2int(1.f));
        uni_vmovq(x, reg_tmp_);
        uni_vbroadcastss(vone_, x);

        const int S = pd_->D() * pd_->H() * pd_->W();
        mov(reg_tmp_, float2int(pd_->MB() * S));
        uni_vmovq(x, reg_tmp_);
        uni_vbroadcastss(vNS_, x);

        mov(reg_blk_has_tail_, dword[PARAM_ADDR(blk_has_tail)]);
    }

    void load_c_specifics() {
        mov(reg_ptr_c_, ptr[PARAM_ADDR(mean)]);
        jit_tail_.uni_vmovups_maybe_tail(
                vmean_, vmmword[reg_ptr_c_ + reg_off_c_]);

        mov(reg_ptr_c_, ptr[PARAM_ADDR(var)]);
        jit_tail_.uni_vmovups_maybe_tail(
                vsqrtvar_, vmmword[reg_ptr_c_ + reg_off_c_]);
        uni_vaddps(vsqrtvar_, vsqrtvar_, veps_);
        uni_vsqrtps(vsqrtvar_, vsqrtvar_);

        if (isa == sse41) {
            movups(vtmp_, vone_);
            divps(vtmp_, vsqrtvar_);
            movups(vsqrtvar_, vtmp_);
        } else
            vdivps(vsqrtvar_, vone_, vsqrtvar_);

        if (pd_->use_scale()) {
            mov(reg_ptr_c_, ptr[PARAM_ADDR(scale)]);
            jit_tail_.uni_vmovups_maybe_tail(
                    vgamma_, vmmword[reg_ptr_c_ + reg_off_c_]);
        }

        if (calculate_diff_stats()) {
            mov(reg_ptr_c_, ptr[PARAM_ADDR(diff_scale)]);
            jit_tail_.uni_vmovups_maybe_tail(
                    vdiff_gamma_, vmmword[reg_ptr_c_ + reg_off_c_]);
            uni_vmulps(vdiff_gamma_, vdiff_gamma_, vsqrtvar_);
            uni_vdivps(vdiff_gamma_, vdiff_gamma_, vNS_);
            mov(reg_ptr_c_, ptr[PARAM_ADDR(diff_shift)]);
            jit_tail_.uni_vmovups_maybe_tail(
                    vdiff_beta_, vmmword[reg_ptr_c_ + reg_off_c_]);
            uni_vdivps(vdiff_beta_, vdiff_beta_, vNS_);
        }
    }

    void compute_bnorm(bool stream_store_allowed) {
        helper_vmovups_(v_, vmmword[reg_ptr_diff_dst_ + reg_off_dat_]);
        jit_relu_.bwd_process_relu(v_);

        if (calculate_diff_stats()) {
            uni_vsubps(v_, v_, vdiff_beta_);
            helper_vmovups_(vtmp_, vmmword[reg_ptr_src_ + reg_off_dat_]);
            uni_vsubps(vtmp_, vtmp_, vmean_);
            uni_vmulps(vtmp_, vtmp_, vdiff_gamma_);
            uni_vsubps(v_, v_, vtmp_);
        }

        if (pd_->use_scale()) uni_vmulps(v_, v_, vgamma_);
        uni_vmulps(v_, v_, vsqrtvar_);

        if (stream_store_allowed) {
            uni_vmovntps(vmmword[reg_ptr_diff_src_ + reg_off_dat_], v_);
        } else {
            helper_vmovups_(vmmword[reg_ptr_diff_src_ + reg_off_dat_], v_);
        }
    }

    void compute_blocked(bool stream_store_allowed) {
        Label label_C, label_S;
        mov(reg_C_, dword[PARAM_ADDR(C)]);
        L(label_C);
        {
            mov(reg_off_dat_, reg_off_dat_save_);

            load_c_specifics();

            mov(reg_S_, dword[PARAM_ADDR(S)]);
            L(label_S);
            {
                compute_bnorm(stream_store_allowed);

                add(reg_off_dat_, stride_S_ * data_type_size_);

                dec(reg_S_);
                jnz(label_S);
            }

            add(reg_off_dat_save_, stride_C_ * data_type_size_);
            add(reg_off_c_, simd_w * acc_type_size_);

            dec(reg_C_);
            jnz(label_C);
        }
    }

    void compute_nspc(bool stream_store_allowed) {
        Label label_C, label_S;
        mov(reg_S_, dword[PARAM_ADDR(S)]);
        L(label_S);
        {
            mov(reg_off_dat_, reg_off_dat_save_);
            xor_(reg_off_c_, reg_off_c_);

            mov(reg_C_, dword[PARAM_ADDR(C)]);
            L(label_C);
            {
                load_c_specifics();

                compute_bnorm(stream_store_allowed);

                add(reg_off_c_, simd_w * acc_type_size_);
                add(reg_off_dat_, stride_C_ * data_type_size_);

                dec(reg_C_);
                jnz(label_C);
            }

            add(reg_off_dat_save_, stride_S_ * data_type_size_);

            dec(reg_S_);
            jnz(label_S);
        }
    }

    void compute(bool stream_store_allowed) {
        Label label_N;
        mov(reg_N_, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save_, reg_off_dat_save_);
            xor_(reg_off_c_, reg_off_c_);

            tag_kind_ == jit_memory_tag_kind_t::nspc
                    ? compute_nspc(stream_store_allowed)
                    : compute_blocked(stream_store_allowed);

            if (isa == sse41 && tag_kind_ == jit_memory_tag_kind_t::blocked) {
                xor_(reg_off_dat_save_, reg_off_dat_save_);
                xor_(reg_off_c_, reg_off_c_);
                add(reg_off_dat_save_, vlen / 2);
                add(reg_off_c_, vlen / 2);

                compute_blocked(stream_store_allowed);
            }

            add(reg_ptr_src_, stride_N_ * data_type_size_);
            add(reg_ptr_diff_src_, stride_N_ * data_type_size_);
            add(reg_ptr_diff_dst_, stride_N_ * data_type_size_);
            add(reg_ptr_ws_, stride_N_ / bits_per_byte);

            dec(reg_N_);
            jnz(label_N);
        }
    }

    bool calculate_diff_stats() const { return !pd_->use_global_stats(); }

    jit_bnorm_bwd_t(const batch_normalization_pd_t *pd,
            const jit_memory_tag_kind_t tag_kind)
        : jit_generator(jit_name())
        , pd_(pd)
        , tag_kind_(tag_kind)
        , vlen(get_vlen<isa>(tag_kind))
        , simd_w(get_simd_w<isa>(tag_kind))
        , jit_tail_(pd, this, reg_tmp_, reg_blk_has_tail_, reg_C_, vtail_mask_,
                  ktail_mask_)
        , jit_relu_(pd, this, reg_off_dat_, reg_tmp_, reg_ptr_ws_, vzero_,
                  vstore_mask_, kstore_mask_)
        , helper_vmovups_(pd, this, zmm28, zmm29, zmm30, zmm31, reg_tmp_)
        , data_type_size_(types::data_type_size(pd->src_md()->data_type))
        , acc_type_size_(sizeof(acc_data_t)) {
        static_assert(utils::one_of(isa, sse41, avx2, avx512_core),
                "unsupported isa");

        std::tie(stride_N_, stride_S_, stride_C_)
                = get_data_strides<isa>(pd_, tag_kind);
    }

    void generate() override {
        bool is_bf16 = pd_->src_md()->data_type == data_type::bf16;
        bool is_f16 = pd_->src_md()->data_type == data_type::f16;
        const bool is_tail_in_nspc_format
                = tag_kind_ == jit_memory_tag_kind_t::nspc
                && jit_tail_.tail_ != 0;
        const bool stream_store_allowed
                = !is_bf16 && !is_f16 && !is_tail_in_nspc_format;

        preamble();
        if (helper_vmovups_.bf16_emu_)
            helper_vmovups_.bf16_emu_->init_vcvtneps2bf16();
        load_common_params();
        jit_relu_.bwd_prepare_relu();
        jit_tail_.prepare_tail();

        Label normal_store, end_store;
        test(reg_ptr_diff_src_, vlen - 1);
        jnz(normal_store, T_NEAR);
        compute(stream_store_allowed);
        jmp(end_store, T_NEAR);
        L(normal_store);
        { compute(false); }
        L(end_store);

        postamble();
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_bwd_diff_ss_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_bwd_diff_ss_t)
    using Vmm = typename cpu_isa_traits<isa>::Vmm;

    const AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                      ? yword
                                                 : zword;

    struct call_params_t {
        size_t N, C, S;
        const void *src, *diff_dst;
        const uint8_t *ws;
        const acc_data_t *mean, *var;
        const acc_data_t *diff_gamma, *diff_beta;
        size_t blk_has_tail;
    };

    const Reg64 reg_param_ = abi_param1;
    const Reg64 reg_tmp_ = abi_not_param1;
    const Reg64 reg_N_ = rsi;
    const Reg64 reg_S_ = rax;
    const Reg64 reg_C_ = rdx;
    const Reg64 reg_off_c_ = rbx;
    const Reg64 reg_blk_has_tail_ = rbp;

    const Reg64 reg_off_dat_ = r8;
    const Reg64 reg_off_dat_save_ = r9;
    const Reg64 reg_ptr_c_ = r10;
    const Reg64 reg_ptr_diff_gamma_ = r11;
    const Reg64 reg_ptr_diff_beta_ = r12;
    const Reg64 reg_ptr_ws_ = r13;
    const Reg64 reg_ptr_diff_dst_ = r14;
    const Reg64 reg_ptr_src_ = r15;

    const Vmm vtail_mask_ = Vmm(0);
    const Vmm v_ = Vmm(1);
    const Vmm vtmp_ = Vmm(2);
    const Vmm vstore_mask_ = vtmp_;
    const Vmm vzero_ = Vmm(3);
    const Vmm veps_ = Vmm(4);
    const Vmm vone_ = Vmm(5);
    // Diff_beta, diff_gamma and one of the statistic values(mean or sqrtvar)
    // are unrolled i.e.three vmms are needed to unroll one c block at any moment,
    // therefore the number of registers which are used to unrolling must to be
    // divisible by three.
    static constexpr int min_idx_to_unroll_ = 6;
    static constexpr int max_idx_to_unroll_ = isa == avx512_core ? 27 : 15;
    static constexpr int number_of_unrolled_variables_ = 3;
    static constexpr int number_of_vmms_to_unrolling_variables_
            = max_idx_to_unroll_ - min_idx_to_unroll_;
    static_assert(number_of_vmms_to_unrolling_variables_
                                    % number_of_unrolled_variables_
                            == 0
                    && number_of_vmms_to_unrolling_variables_ != 0,
            "Number of register to unrolling must to be divisible by 3.");

    const Opmask kstore_mask_ = k1;
    const Opmask ktail_mask_ = k2;

    const batch_normalization_pd_t *pd_;
    const jit_memory_tag_kind_t tag_kind_;
    const int vlen;
    const int simd_w;
    jit_bnorm_process_tail_t<isa> jit_tail_;
    jit_bnorm_process_relu_t<isa> jit_relu_;
    helper_vmovups_data_t<isa> helper_vmovups_;
    int stride_N_, stride_S_, stride_C_;
    size_t data_type_size_, acc_type_size_;

    void load_common_params() {
#define PARAM_PTR(x) ptr[PARAM_ADDR(x)]
        mov(reg_ptr_src_, PARAM_PTR(src));
        mov(reg_ptr_diff_dst_, PARAM_PTR(diff_dst));
        mov(reg_ptr_ws_, PARAM_PTR(ws));
        mov(reg_ptr_diff_gamma_, PARAM_PTR(diff_gamma));
        mov(reg_ptr_diff_beta_, PARAM_PTR(diff_beta));
#undef PARAM_PTR

        Xmm x = Xmm(v_.getIdx());

        mov(reg_tmp_, float2int(pd_->desc()->batch_norm_epsilon));
        uni_vmovq(x, reg_tmp_);
        uni_vbroadcastss(veps_, x);

        mov(reg_tmp_, float2int(1.f));
        uni_vmovq(x, reg_tmp_);
        uni_vbroadcastss(vone_, x);

        mov(reg_blk_has_tail_, dword[PARAM_ADDR(blk_has_tail)]);
    }

    void zeroise() {
        Label label_zeroise;
        xor_(reg_off_c_, reg_off_c_);
        uni_vpxor(vzero_, vzero_, vzero_);
        mov(reg_C_, dword[PARAM_ADDR(C)]);
        L(label_zeroise);
        {
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_diff_gamma_ + reg_off_c_], vzero_);
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_diff_beta_ + reg_off_c_], vzero_);
            if (isa == sse41 && tag_kind_ == jit_memory_tag_kind_t::blocked) {
                jit_tail_.uni_vmovups_maybe_tail(
                        vmmword[reg_ptr_diff_gamma_ + reg_off_c_ + vlen / 2],
                        vzero_);
                jit_tail_.uni_vmovups_maybe_tail(
                        vmmword[reg_ptr_diff_beta_ + reg_off_c_ + vlen / 2],
                        vzero_);
            }
            add(reg_off_c_, simd_w * acc_type_size_);
            dec(reg_C_);
            jnz(label_zeroise);
        }
    }

    void load_mean(const int c_blks_to_unroll = 1) {
        mov(reg_ptr_c_, ptr[PARAM_ADDR(mean)]);

        const int start_idx = min_idx_to_unroll_;
        const int end_idx = number_of_unrolled_variables_ * c_blks_to_unroll
                + min_idx_to_unroll_;
        const int step = simd_w * acc_type_size_;

        for (int idx = start_idx, off = 0; idx < end_idx;
                idx += number_of_unrolled_variables_, off += step) {
            const Vmm vmean = Vmm(idx);

            jit_tail_.uni_vmovups_maybe_tail(
                    vmean, vmmword[reg_ptr_c_ + reg_off_c_ + off]);
        }
    }

    void zeroise_diff_beta_and_diff_gamma(const int c_blks_to_unroll = 1) {
        const int start_idx = min_idx_to_unroll_;
        const int end_idx = number_of_unrolled_variables_ * c_blks_to_unroll
                + min_idx_to_unroll_;

        for (int idx = start_idx; idx < end_idx;
                idx += number_of_unrolled_variables_) {
            const Vmm vdiff_beta = Vmm(idx + 1);
            const Vmm vdiff_gamma = Vmm(idx + 2);

            uni_vpxor(vdiff_beta, vdiff_beta, vdiff_beta);
            uni_vpxor(vdiff_gamma, vdiff_gamma, vdiff_gamma);
        }
    }

    void load_and_prepare_sqrtvar(const int c_blks_to_unroll = 1) {
        mov(reg_ptr_c_, ptr[PARAM_ADDR(var)]);

        const int start_idx = min_idx_to_unroll_;
        const int end_idx = number_of_unrolled_variables_ * c_blks_to_unroll
                + min_idx_to_unroll_;
        const int step = simd_w * acc_type_size_;

        for (int idx = start_idx, off = 0; idx < end_idx;
                idx += number_of_unrolled_variables_, off += step) {
            const Vmm vsqrtvar = Vmm(idx);

            jit_tail_.uni_vmovups_maybe_tail(
                    vsqrtvar, vmmword[reg_ptr_c_ + reg_off_c_ + off]);

            // 1.0 / sqrt(var + eps)
            uni_vaddps(vsqrtvar, vsqrtvar, veps_);
            uni_vsqrtps(vsqrtvar, vsqrtvar);

            if (isa == sse41) {
                movups(vtmp_, vone_);
                divps(vtmp_, vsqrtvar);
                movups(vsqrtvar, vtmp_);
            } else
                vdivps(vsqrtvar, vone_, vsqrtvar);
        }
    }

    void compute_diff_beta_and_diff_gamma(const int c_blks_to_unroll = 1) {
        const int start_idx = min_idx_to_unroll_;
        const int end_idx = number_of_unrolled_variables_ * c_blks_to_unroll
                + min_idx_to_unroll_;
        const int step = simd_w * data_type_size_;

        for (int idx = start_idx, off = 0; idx < end_idx;
                idx += number_of_unrolled_variables_, off += step) {
            const Vmm vmean = Vmm(idx);
            const Vmm vdiff_beta = Vmm(idx + 1);
            const Vmm vdiff_gamma = Vmm(idx + 2);

            helper_vmovups_(
                    v_, vmmword[reg_ptr_diff_dst_ + reg_off_dat_ + off]);

            jit_relu_.bwd_process_relu(
                    v_, off / (bits_per_byte * data_type_size_));

            // diff_beta
            uni_vaddps(vdiff_beta, vdiff_beta, v_);

            helper_vmovups_(vtmp_, vmmword[reg_ptr_src_ + reg_off_dat_ + off]);

            // diff_gamma, note that diff_gamma will be multiplied
            // by sqrtvar before store
            uni_vsubps(vtmp_, vtmp_, vmean);
            uni_vfmadd231ps(vdiff_gamma, vtmp_, v_);
        }
    }

    void store_diff_beta_and_diff_gamma(const int c_blks_to_unroll = 1) {
        const int start_idx = min_idx_to_unroll_;
        const int end_idx = number_of_unrolled_variables_ * c_blks_to_unroll
                + min_idx_to_unroll_;
        const int step = simd_w * acc_type_size_;

        for (int idx = start_idx, off = 0; idx < end_idx;
                idx += number_of_unrolled_variables_, off += step) {
            const Vmm vdiff_beta = Vmm(idx + 1);

            jit_tail_.uni_vmovups_maybe_tail(
                    vtmp_, vmmword[reg_ptr_diff_beta_ + reg_off_c_ + off]);
            uni_vaddps(vdiff_beta, vdiff_beta, vtmp_);
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_diff_beta_ + reg_off_c_ + off], vdiff_beta);
        }

        for (int idx = start_idx, off = 0; idx < end_idx;
                idx += number_of_unrolled_variables_, off += step) {
            const Vmm vsqrtvar = Vmm(idx);
            const Vmm vdiff_gamma = Vmm(idx + 2);

            // multiply diff_gamma by 1.0/sqrt(var + eps)
            uni_vmulps(vdiff_gamma, vdiff_gamma, vsqrtvar);

            jit_tail_.uni_vmovups_maybe_tail(
                    vtmp_, vmmword[reg_ptr_diff_gamma_ + reg_off_c_ + off]);
            uni_vaddps(vdiff_gamma, vdiff_gamma, vtmp_);
            jit_tail_.uni_vmovups_maybe_tail(
                    vmmword[reg_ptr_diff_gamma_ + reg_off_c_ + off],
                    vdiff_gamma);
        }
    }

    void compute_blocked() {
        Label label_C, label_S;
        mov(reg_C_, dword[PARAM_ADDR(C)]);
        L(label_C);
        {
            mov(reg_off_dat_, reg_off_dat_save_);

            load_mean();
            zeroise_diff_beta_and_diff_gamma();

            mov(reg_S_, dword[PARAM_ADDR(S)]);
            L(label_S);
            {
                compute_diff_beta_and_diff_gamma();

                add(reg_off_dat_, stride_S_ * data_type_size_);

                dec(reg_S_);
                jnz(label_S);
            }

            load_and_prepare_sqrtvar();
            store_diff_beta_and_diff_gamma();

            add(reg_off_dat_save_, stride_C_ * data_type_size_);
            add(reg_off_c_, simd_w * acc_type_size_);

            dec(reg_C_);
            jnz(label_C);
        }
    }

    void compute_nspc() {
        mov(reg_C_, dword[PARAM_ADDR(C)]);

        constexpr int max_of_unrolled_c_blks
                = number_of_vmms_to_unrolling_variables_
                / number_of_unrolled_variables_;
        std::vector<Label> c_unroll_label(max_of_unrolled_c_blks + 1);

        for (int c_blks_to_unroll = max_of_unrolled_c_blks;
                c_blks_to_unroll > 0; --c_blks_to_unroll) {
            L(c_unroll_label[c_blks_to_unroll]);
            {
                cmp(reg_C_, c_blks_to_unroll);
                jl(c_unroll_label[c_blks_to_unroll - 1], T_NEAR);

                mov(reg_off_dat_, reg_off_dat_save_);

                load_mean(c_blks_to_unroll);
                zeroise_diff_beta_and_diff_gamma(c_blks_to_unroll);

                Label label_S;
                mov(reg_S_, dword[PARAM_ADDR(S)]);
                L(label_S);
                {
                    compute_diff_beta_and_diff_gamma(c_blks_to_unroll);

                    add(reg_off_dat_, stride_S_ * data_type_size_);

                    dec(reg_S_);
                    jnz(label_S);
                }

                load_and_prepare_sqrtvar(c_blks_to_unroll);
                store_diff_beta_and_diff_gamma(c_blks_to_unroll);

                add(reg_off_c_, c_blks_to_unroll * simd_w * acc_type_size_);
                add(reg_off_dat_save_,
                        c_blks_to_unroll * stride_C_ * data_type_size_);

                sub(reg_C_, c_blks_to_unroll);
                jmp(c_unroll_label[c_blks_to_unroll], T_NEAR);
            }
        }
        L(c_unroll_label[0]);
    }

    void compute() {
        Label label_N;
        mov(reg_N_, dword[PARAM_ADDR(N)]);
        L(label_N);
        {
            xor_(reg_off_dat_save_, reg_off_dat_save_);
            xor_(reg_off_c_, reg_off_c_);

            tag_kind_ == jit_memory_tag_kind_t::nspc ? compute_nspc()
                                                     : compute_blocked();

            if (isa == sse41 && tag_kind_ == jit_memory_tag_kind_t::blocked) {
                xor_(reg_off_dat_save_, reg_off_dat_save_);
                xor_(reg_off_c_, reg_off_c_);
                add(reg_off_dat_save_, vlen / 2);
                add(reg_off_c_, vlen / 2);

                compute_blocked();
            }

            add(reg_ptr_src_, stride_N_ * data_type_size_);
            add(reg_ptr_diff_dst_, stride_N_ * data_type_size_);
            add(reg_ptr_ws_, stride_N_ / bits_per_byte);

            dec(reg_N_);
            jnz(label_N);
        }
    }

    jit_bnorm_bwd_diff_ss_t(const batch_normalization_pd_t *pd,
            const jit_memory_tag_kind_t tag_kind)
        : jit_generator(jit_name())
        , pd_(pd)
        , tag_kind_(tag_kind)
        , vlen(get_vlen<isa>(tag_kind))
        , simd_w(get_simd_w<isa>(tag_kind))
        , jit_tail_(pd, this, reg_tmp_, reg_blk_has_tail_, reg_C_, vtail_mask_,
                  ktail_mask_)
        , jit_relu_(pd, this, reg_off_dat_, reg_tmp_, reg_ptr_ws_, vzero_,
                  vstore_mask_, kstore_mask_)
        , helper_vmovups_(pd, this, zmm28, zmm29, zmm30, zmm31, reg_tmp_)
        , data_type_size_(types::data_type_size(pd->src_md()->data_type))
        , acc_type_size_(sizeof(acc_data_t)) {
        static_assert(utils::one_of(isa, sse41, avx2, avx512_core),
                "unsupported isa");

        std::tie(stride_N_, stride_S_, stride_C_)
                = get_data_strides<isa>(pd_, tag_kind);
    }

    void generate() override {
        preamble();
        load_common_params();
        jit_relu_.bwd_prepare_relu();
        jit_tail_.prepare_tail();
        zeroise();
        compute();
        postamble();
    }
};

namespace bnorm_tbb_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {
private:
    struct bnorm_dims_t {
        dim_t N, C, S;
        dim_t glob;
    };

    DNNL_DISALLOW_COPY_AND_ASSIGN(driver_t);

public:
    driver_t(const batch_normalization_pd_t *pd,
            const jit_memory_tag_kind_t tag_kind)
        : pd_(pd)
        , tag_kind_(tag_kind)
        , simd_w(get_simd_w<isa>(tag_kind))
        , nthr_(dnnl_get_max_threads())
        , N_(pd_->MB())
        , S_(pd_->D() * pd_->H() * pd_->W())
        , C_(pd_->C())
        , C_blks_(get_c_padded(pd_) / simd_w)
        , dt_size_(types::data_type_size(pd_->src_md()->data_type)) {

        const size_t l3_size = platform::get_per_core_cache_size(3) * nthr_ / 2;
        int num_tensors = pd_->is_fwd() ? 1 : 2;

        const size_t working_set_size
                = dt_size_ * N_ * S_ * simd_w * num_tensors;

        do_blocking_ = tag_kind_ == jit_memory_tag_kind_t::nspc
                ? false
                : working_set_size * C_blks_ >= l3_size / 2 && l3_size > 0;

        if (tag_kind_ == jit_memory_tag_kind_t::nspc) {
            if (normalize_only(pd_)) {
                // blocks have to fit in a 4rth of L1 so that they don't get evicted
                // There are at most 6 tensors: src, dst, mean, var, scale, shift
                dim_t n_tensors = 2 + pd_->use_scale() + pd_->use_shift();
                C_blk_step_ = utils::saturate<dim_t>(1, C_blks_,
                        platform::get_per_core_cache_size(1)
                                / get_vlen<isa>(jit_memory_tag_kind_t::nspc)
                                / n_tensors);
            } else
                C_blk_step_ = C_blks_;
        } else {
            C_blk_step_ = utils::saturate<dim_t>(
                    1, C_blks_, l3_size / working_set_size);
        }
    }

    status_t create_kernel() {
        if (pd_->is_fwd()) {
            CHECK(safe_ptr_assign(
                    ker_fwd_, new jit_bnorm_fwd_t<isa>(pd_, tag_kind_)));
            CHECK(ker_fwd_->create_kernel());
            if (!pd_->stats_is_src()) {
                CHECK(safe_ptr_assign(ker_fwd_mean_,
                        new jit_bnorm_fwd_mean_t<isa>(pd_, tag_kind_)));
                CHECK(safe_ptr_assign(ker_fwd_var_,
                        new jit_bnorm_fwd_var_t<isa>(pd_, tag_kind_)));
                CHECK(ker_fwd_mean_->create_kernel());
                CHECK(ker_fwd_var_->create_kernel());
            }
        } else {
            CHECK(safe_ptr_assign(
                    ker_bwd_, new jit_bnorm_bwd_t<isa>(pd_, tag_kind_)));
            CHECK(safe_ptr_assign(ker_bwd_diff_ss_,
                    new jit_bnorm_bwd_diff_ss_t<isa>(pd_, tag_kind_)));
            CHECK(ker_bwd_->create_kernel());
            CHECK(ker_bwd_diff_ss_->create_kernel());
        }
        return status::success;
    }

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const batch_normalization_pd_t *pd) {

        int nthrs = dnnl_get_max_threads();
        int C_PADDED = get_c_padded(pd);

        auto sbuf_sz = use_tmp_stats(pd) * 2 * C_PADDED;
        auto pbuf_sz
                = (use_tmp_diff_scale(pd) + use_tmp_diff_shift(pd)) * C_PADDED;
        auto rbuf_sz = (pd->is_fwd() ? 1 : 2) * C_PADDED * nthrs;

        scratchpad.book<acc_data_t>(key_bnorm_tmp_stats, sbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_tmp_diff_ss, pbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_reduction, rbuf_sz);
    }

    void exec_fwd_step_stats(const dim_t C_blks, const bnorm_dims_t &nthr,
            const void *src, acc_data_t *mean, acc_data_t *var,
            acc_data_t *rbuf, bool blk_has_tail) {
        size_t stride_C, stride_N, stride_S;
        std::tie(stride_N, stride_S, stride_C)
                = get_data_strides<isa>(pd_, tag_kind_);

        const int nthr_NS = nthr.N * nthr.S;
        const bool need_reduction = nthr_NS > 1;
        const dim_t tail_size = blk_has_tail ? C_ % simd_w : simd_w;

        const dim_t size_C_stat = (C_blks - 1) * simd_w + tail_size;

        auto reduce = [&](acc_data_t *stat, acc_data_t *r_stat) {
            if (!need_reduction) return;
            acc_data_t *loc_stat = r_stat;

            for (dim_t c = 0; c < size_C_stat; ++c)
                stat[c] = loc_stat[c];

            for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                loc_stat += size_C_stat;
                for (dim_t c = 0; c < size_C_stat; ++c)
                    stat[c] += loc_stat[c];
            }

            for (dim_t c = 0; c < size_C_stat; ++c)
                stat[c] /= N_ * S_;
        };

        // find local mean
        acc_data_t *r_mean = need_reduction ? rbuf : mean;
        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            auto c = typename jit_bnorm_fwd_mean_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * stride_S;
            c.src = (void *)((char *)src + d_off * dt_size_);
            const int ithr_NS = ithr.N * nthr.S + ithr.S;
            c.mean = &r_mean[ithr_NS * size_C_stat + start.C * simd_w];
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;
            c.do_normalise = !need_reduction;
            (*ker_fwd_mean_)(&c);
        });

        // mean reduction
        reduce(mean, r_mean);

        // find local var
        acc_data_t *r_var = need_reduction ? rbuf : var;
        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            auto c = typename jit_bnorm_fwd_var_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * stride_S;
            c.src = (void *)((char *)src + d_off * dt_size_);
            const int ithr_NS = ithr.N * nthr.S + ithr.S;
            c.mean = &mean[start.C * simd_w];
            c.var = &r_var[ithr_NS * size_C_stat + start.C * simd_w];
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;
            c.do_normalise = !need_reduction;
            (*ker_fwd_var_)(&c);
        });

        // var reduction
        reduce(var, r_var);
    }

    void exec_fwd_step_normalization(const dim_t C_blks,
            const bnorm_dims_t &nthr, const void *src, void *dst,
            const acc_data_t *scale, const acc_data_t *shift,
            const acc_data_t *mean, const acc_data_t *var, uint8_t *ws,
            bool blk_has_tail) {
        size_t stride_C, stride_N, stride_S;
        std::tie(stride_N, stride_S, stride_C)
                = get_data_strides<isa>(pd_, tag_kind_);

        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            auto c = typename jit_bnorm_fwd_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;
            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * stride_S;
            c.src = (void *)((char *)src + d_off * dt_size_);
            c.dst = (void *)((char *)dst + d_off * dt_size_);
            c.ws = ws ? &ws[d_off / bits_per_byte] : nullptr;
            c.mean = &mean[start.C * simd_w];
            c.var = &var[start.C * simd_w];
            c.scale = scale ? &scale[start.C * simd_w] : nullptr;
            c.shift = shift ? &shift[start.C * simd_w] : nullptr;
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;
            (*ker_fwd_)(&c);
        });
    }

    void exec_fwd(const void *src, void *dst, const acc_data_t *scale,
            const acc_data_t *shift, acc_data_t *mean, acc_data_t *var,
            uint8_t *ws, const memory_tracking::grantor_t &scratchpad) {
        auto rbuf = scratchpad.get<acc_data_t>(key_bnorm_reduction);
        if (use_tmp_stats(pd_)) {
            auto sbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_stats);
            mean = sbuf;
            var = sbuf + C_blks_ * simd_w;
        }

        size_t stride_C;
        std::tie(std::ignore, std::ignore, stride_C)
                = get_data_strides<isa>(pd_, tag_kind_);

        dim_t C_blk_step = C_blk_step_;
        auto nthr = bnorm_dims_t();

        thread_distribution(C_blk_step, nthr);

        for (dim_t C_blk_st = 0; C_blk_st < C_blks_; C_blk_st += C_blk_step) {
            if (C_blk_st + C_blk_step > C_blks_) {
                C_blk_step = C_blks_ - C_blk_st;
                thread_distribution(C_blk_step, nthr);
            }

            if (!pd_->stats_is_src()) {
                exec_fwd_step_stats(C_blk_step, nthr,
                        (void *)((char *)src
                                + (C_blk_st * stride_C) * dt_size_),
                        mean + C_blk_st * simd_w, var + C_blk_st * simd_w, rbuf,
                        (C_blk_st + C_blk_step) * simd_w > C_);
            }
            exec_fwd_step_normalization(C_blk_step, nthr,
                    (void *)((char *)src + (C_blk_st * stride_C) * dt_size_),
                    (void *)((char *)dst + (C_blk_st * stride_C) * dt_size_),
                    scale + C_blk_st * simd_w, shift + C_blk_st * simd_w,
                    mean + C_blk_st * simd_w, var + C_blk_st * simd_w,
                    ws + C_blk_st * stride_C / bits_per_byte,
                    (C_blk_st + C_blk_step) * simd_w > C_);
        }
    }

    void exec_bwd_step_diff_ss(const dim_t C_blks, const bnorm_dims_t &nthr,
            const void *src, const void *diff_dst, const acc_data_t *mean,
            const acc_data_t *var, const uint8_t *ws, acc_data_t *diff_scale,
            acc_data_t *diff_shift, acc_data_t *rbuf, bool blk_has_tail) {
        size_t stride_C, stride_N, stride_S;
        std::tie(stride_N, stride_S, stride_C)
                = get_data_strides<isa>(pd_, tag_kind_);

        const dim_t tail_size = blk_has_tail ? C_ % simd_w : simd_w;
        const dim_t size_C_stat = (C_blks - 1) * simd_w + tail_size;

        const int nthr_NS = nthr.N * nthr.S;
        const bool need_reduction = nthr_NS > 1;

        acc_data_t *diff_gamma = diff_scale;
        acc_data_t *diff_beta = diff_shift;

        acc_data_t *const r_diff_gamma = need_reduction ? rbuf : diff_gamma;
        acc_data_t *const r_diff_beta
                = need_reduction ? rbuf + nthr_NS * size_C_stat : diff_beta;

        auto reduce = [&]() {
            if (!need_reduction) return;

            // diff_gamma
            const acc_data_t *loc_diff_gamma = r_diff_gamma;
            for (dim_t c = 0; c < size_C_stat; ++c)
                diff_gamma[c] = loc_diff_gamma[c];
            for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                loc_diff_gamma += size_C_stat;
                for (dim_t c = 0; c < size_C_stat; ++c)
                    diff_gamma[c] += loc_diff_gamma[c];
            }

            // diff_beta
            const acc_data_t *loc_diff_beta = r_diff_beta;
            for (dim_t c = 0; c < size_C_stat; ++c)
                diff_beta[c] = loc_diff_beta[c];
            for (int thr_ns = 1; thr_ns < nthr_NS; ++thr_ns) {
                loc_diff_beta += size_C_stat;
                for (dim_t c = 0; c < size_C_stat; ++c)
                    diff_beta[c] += loc_diff_beta[c];
            }
        };

        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            const int ithr_NS = ithr.N * nthr.S + ithr.S;
            acc_data_t *loc_diff_gamma = &r_diff_gamma[ithr_NS * size_C_stat];
            acc_data_t *loc_diff_beta = &r_diff_beta[ithr_NS * size_C_stat];

            auto c = typename jit_bnorm_bwd_diff_ss_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * stride_S;
            c.src = (void *)((char *)src + d_off * dt_size_);
            c.diff_dst = (void *)((char *)diff_dst + d_off * dt_size_);
            c.ws = ws ? &ws[d_off / bits_per_byte] : nullptr;
            c.mean = &mean[start.C * simd_w];
            c.var = &var[start.C * simd_w];
            c.diff_gamma = &loc_diff_gamma[start.C * simd_w];
            c.diff_beta = &loc_diff_beta[start.C * simd_w];
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;

            (*ker_bwd_diff_ss_)(&c);
        });

        reduce();
    }

    void exec_bwd_step_normalization(const dim_t C_blks,
            const bnorm_dims_t &nthr, const void *src, void *diff_src,
            const void *diff_dst, const acc_data_t *mean, const acc_data_t *var,
            const uint8_t *ws, const acc_data_t *scale,
            const acc_data_t *diff_scale, const acc_data_t *diff_shift,
            bool blk_has_tail) {
        size_t stride_C, stride_N, stride_S;
        std::tie(stride_N, stride_S, stride_C)
                = get_data_strides<isa>(pd_, tag_kind_);

        parallel(nthr.glob, [&](int ithr_glob, int nthr_glob) {
            assert(nthr_glob == nthr.glob);
            const auto ithr = map_thread(ithr_glob, nthr);
            bnorm_dims_t start, stop;
            work_distribution(C_blks, ithr, nthr, start, stop);

            auto c = typename jit_bnorm_bwd_t<isa>::call_params_t();
            c.N = stop.N - start.N;
            c.C = stop.C - start.C;
            c.S = stop.S - start.S;

            const size_t d_off = start.N * stride_N + start.C * stride_C
                    + start.S * stride_S;
            c.src = (void *)((char *)src + d_off * dt_size_);
            c.diff_src = (void *)((char *)diff_src + d_off * dt_size_);
            c.diff_dst = (void *)((char *)diff_dst + d_off * dt_size_);
            c.ws = ws ? &ws[d_off / bits_per_byte] : nullptr;
            c.mean = &mean[start.C * simd_w];
            c.var = &var[start.C * simd_w];
            c.scale = scale ? &scale[start.C * simd_w] : nullptr;
            c.diff_scale = &diff_scale[start.C * simd_w];
            c.diff_shift = &diff_shift[start.C * simd_w];
            c.blk_has_tail = blk_has_tail && stop.C == C_blks;

            (*ker_bwd_)(&c);
        });
    }

    void exec_bwd(const void *src, void *diff_src, const void *diff_dst,
            const acc_data_t *scale, acc_data_t *diff_scale,
            acc_data_t *diff_shift, const acc_data_t *mean,
            const acc_data_t *var, const uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto rbuf = scratchpad.get<acc_data_t>(key_bnorm_reduction);
        if (use_tmp_diff_scale(pd_)) {
            auto pbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_diff_ss);
            diff_scale = pbuf;
        }
        if (use_tmp_diff_shift(pd_)) {
            auto pbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_diff_ss);
            size_t shift_off = use_tmp_diff_scale(pd_) ? pd_->C() : 0;
            diff_shift = &pbuf[shift_off];
        }

        size_t stride_C;
        std::tie(std::ignore, std::ignore, stride_C)
                = get_data_strides<isa>(pd_, tag_kind_);

        dim_t C_blk_step = C_blk_step_;
        auto nthr = bnorm_dims_t();

        thread_distribution(C_blk_step, nthr);

        for (dim_t C_blk_st = 0; C_blk_st < C_blks_; C_blk_st += C_blk_step) {
            if (C_blk_st + C_blk_step > C_blks_) {
                C_blk_step = C_blks_ - C_blk_st;
                thread_distribution(C_blk_step, nthr);
            }

            exec_bwd_step_diff_ss(C_blk_step, nthr,
                    (void *)((char *)src + (C_blk_st * stride_C) * dt_size_),
                    (void *)((char *)diff_dst
                            + (C_blk_st * stride_C) * dt_size_),
                    mean + C_blk_st * simd_w, var + C_blk_st * simd_w,
                    ws + C_blk_st * stride_C / bits_per_byte,
                    diff_scale + C_blk_st * simd_w,
                    diff_shift + C_blk_st * simd_w, rbuf,
                    (C_blk_st + C_blk_step) * simd_w > C_);

            exec_bwd_step_normalization(C_blk_step, nthr,
                    (void *)((char *)src + (C_blk_st * stride_C) * dt_size_),
                    (void *)((char *)diff_src
                            + (C_blk_st * stride_C) * dt_size_),
                    (void *)((char *)diff_dst
                            + (C_blk_st * stride_C) * dt_size_),
                    mean + C_blk_st * simd_w, var + C_blk_st * simd_w,
                    ws + C_blk_st * stride_C / bits_per_byte,
                    scale + C_blk_st * simd_w, diff_scale + C_blk_st * simd_w,
                    diff_shift + C_blk_st * simd_w,
                    (C_blk_st + C_blk_step) * simd_w > C_);
        }
    }

private:
    static bool use_tmp_stats(const batch_normalization_pd_t *pd) {
        return !pd->stats_is_src()
                && pd->desc()->prop_kind == prop_kind::forward_inference;
    }

    static bool use_tmp_diff_scale(const batch_normalization_pd_t *pd) {
        return (!pd->is_fwd() && !pd->use_scale())
                || pd->desc()->prop_kind == prop_kind::backward_data;
    }

    static bool use_tmp_diff_shift(const batch_normalization_pd_t *pd) {
        return (!pd->is_fwd() && !pd->use_shift())
                || pd->desc()->prop_kind == prop_kind::backward_data;
    }

    void thread_distribution_nspc(dim_t C_blks, bnorm_dims_t &nthr) {
        if (normalize_only(pd_)) {
            // We want to keep some granularity on S so that we can
            // stay on 1 socket if possible, this is why we divide
            // work in chunks fitting in L2

            dim_t n_stats_ss_tensors = pd_->use_scale() + pd_->use_shift();
            dim_t size_stats_ss_tensors = n_stats_ss_tensors * get_c_padded(pd_)
                    * sizeof(acc_data_t);

            dim_t size_src_dst = 2 * N_ * S_ * get_c_padded(pd_)
                    * types::data_type_size(pd_->src_md()->data_type);

            dim_t total_size = size_src_dst + size_stats_ss_tensors;

            // Try to create at least nthr_ chunks for realtime inference. Not
            // enabled for throughput inference to avoid potential regressions
            // for multi-socket runs with threadpool runtime.
            // TODO: Enable for throughput inference.
            const int n_chunks_min = nthr_ <= 8 ? nthr_ : 1;
            const size_t l2_per_core = platform::get_per_core_cache_size(2);
            dim_t n_chunks
                    = nstl::max<dim_t>(n_chunks_min, total_size / l2_per_core);

            // we prioritize parallelization on N, then S, and finally C
            nthr.N = utils::saturate<dim_t>(1, N_, n_chunks);
            nthr.S = utils::saturate<dim_t>(1, S_, n_chunks / nthr.N);
            nthr.C = utils::saturate<dim_t>(
                    1, C_blks, n_chunks / (nthr.N * nthr.S));
        } else {
            if ((nthr_ <= C_blks && nthr_ == 1) || C_blks <= 8)
                nthr.C = 1;
            else if (nthr_ >= 8 && C_blks <= 32)
                nthr.C = 8;
            else {
                nthr.C = math::gcd((dim_t)nthr_, C_blks);
                // Unroll by channels in JIT kernel
                if ((nthr.C == C_blks) || (nthr.C == nthr_)) nthr.C = 1;
            }
            nthr.N = utils::saturate((dim_t)1, N_, nthr_ / nthr.C);
            nthr.S = utils::saturate((dim_t)1, S_, nthr_ / (nthr.C * nthr.N));
        }
    }

    void thread_distribution(dim_t C_blks, bnorm_dims_t &nthr) {
        if (do_blocking_) {
            nthr.N = nstl::min<dim_t>(N_, nthr_);
            nthr.C = nstl::min<dim_t>(C_blks, nthr_ / nthr.N);
            nthr.S = utils::saturate((dim_t)1, S_, nthr_ / (nthr.C * nthr.N));
        } else {
            if (tag_kind_ == jit_memory_tag_kind_t::nspc) {
                thread_distribution_nspc(C_blks, nthr);
            } else {
                nthr.C = math::gcd((dim_t)nthr_, C_blks);
                nthr.N = utils::saturate((dim_t)1, N_, nthr_ / nthr.C);
                nthr.S = utils::saturate(
                        (dim_t)1, S_, nthr_ / (nthr.C * nthr.N));
            }
        }
        nthr.glob = nthr.N * nthr.C * nthr.S;
    }

    int map_thread_c(int ithr_glob, const bnorm_dims_t &nthr) {
        return ithr_glob / nthr.N / nthr.S;
    }

    bnorm_dims_t map_thread(int ithr_glob, const bnorm_dims_t &nthr) {
        auto ithr = bnorm_dims_t();
        ithr.glob = ithr_glob;
        ithr.C = map_thread_c(ithr.glob, nthr);
        ithr.N = ithr.glob / nthr.S % nthr.N;
        ithr.S = ithr.glob % nthr.S;
        return ithr;
    }

    void work_distribution_c(dim_t C_blks, int ithr_c, int nthr_c,
            dim_t &start_c, dim_t &stop_c) {
        balance211(C_blks, nthr_c, ithr_c, start_c, stop_c);
    }

    void work_distribution(dim_t C_blks, const bnorm_dims_t &ithr,
            const bnorm_dims_t &nthr, bnorm_dims_t &start, bnorm_dims_t &stop) {
        work_distribution_c(C_blks, ithr.C, nthr.C, start.C, stop.C);
        balance211(N_, nthr.N, ithr.N, start.N, stop.N);
        balance211(S_, nthr.S, ithr.S, start.S, stop.S);
    }

    const batch_normalization_pd_t *pd_;
    const jit_memory_tag_kind_t tag_kind_;
    const int simd_w;

    bool do_blocking_;

    int nthr_;

    dim_t N_, S_; // MB, D * H *W
    dim_t C_, C_blks_; // C / simd_w
    dim_t C_blk_step_; // for C_blks = 0 .. C_blks_, += C_blk_step_

    std::unique_ptr<jit_bnorm_fwd_t<isa>> ker_fwd_;
    std::unique_ptr<jit_bnorm_fwd_mean_t<isa>> ker_fwd_mean_;
    std::unique_ptr<jit_bnorm_fwd_var_t<isa>> ker_fwd_var_;
    std::unique_ptr<jit_bnorm_bwd_t<isa>> ker_bwd_;
    std::unique_ptr<jit_bnorm_bwd_diff_ss_t<isa>> ker_bwd_diff_ss_;

    size_t dt_size_;
};
} // namespace bnorm_tbb_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */
template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_fwd_t<isa>::pd_t::init(
        engine_t *engine) {
    VDISPATCH_BNORM(is_fwd(), VERBOSE_BAD_PROPKIND);

    // disabling verbose dispatch checks for unsupported isa for better readability
    if (!mayiuse(isa)) return status::unimplemented;

    VDISPATCH_BNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_BNORM(one_of(src_md()->data_type, f32, bf16, f16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_BNORM(src_md()->data_type == dst_md()->data_type,
            VERBOSE_INCONSISTENT_DT, "src", "dst");
    VDISPATCH_BNORM(IMPLICATION(src_md()->data_type == bf16,
                            is_superset(isa, avx512_core)
                                    || (isa == avx2 && mayiuse(avx2_vnni_2))),
            VERBOSE_ISA_DT_MISMATCH);
    // Note: re-using avx512_core/avx2 implementation for f16.
    // This is okay as currently, we do not support binary post-ops
    // for this primitive.
    VDISPATCH_BNORM(
            IMPLICATION(src_md()->data_type == f16,
                    (is_superset(isa, avx512_core) && mayiuse(avx512_core_fp16))
                            || (isa == avx2 && mayiuse(avx2_vnni_2))),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_BNORM(check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_FEATURE,
            "unsupported scale or shift data type");
    VDISPATCH_BNORM(
            (attr()->has_default_values() || with_relu_post_op(is_training())),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_BNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_BNORM(
            memory_desc_wrapper(src_md()) == memory_desc_wrapper(dst_md()),
            VERBOSE_INCONSISTENT_MDS, "src", "dst");

    // BN+Add+Relu fusion is not currently implemented
    VDISPATCH_BNORM(!(fuse_norm_add_relu()), VERBOSE_UNSUPPORTED_FEATURE,
            "sum+relu post-ops configuration is not supported");

    const format_tag_t blocked_tag = is_superset(isa, avx512_core)
            ? utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c)
            : utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);

    const format_tag_t blocked_format
            = memory_desc_matches_tag(*src_md(), blocked_tag)
            ? blocked_tag
            : format_tag::undef;
    const format_tag_t nspc_format
            = memory_desc_matches_one_of_tag(*src_md(), nc, nwc, nhwc, ndhwc);

    if (memory_desc_matches_tag(*dst_md(), blocked_format))
        tag_kind_ = jit_memory_tag_kind_t::blocked;
    else if (memory_desc_matches_tag(*dst_md(), nspc_format)) {
        tag_kind_ = jit_memory_tag_kind_t::nspc;
        const int simd_w = get_simd_w<isa>(tag_kind_);
        VDISPATCH_BNORM(
                C() % simd_w == 0, VERBOSE_BLOCKING_FAIL, "bad src dimensions");
    } else
        VDISPATCH_BNORM(false, VERBOSE_UNSUPPORTED_TAG);

    // AVX2 only supports xf16 on plain layout and inference
    VDISPATCH_BNORM(
            !(utils::one_of(src_md()->data_type, bf16, f16) && isa == avx2
                    && (is_training()
                            || !memory_desc_matches_tag(
                                    *dst_md(), nspc_format))),
            "unsupported dt, isa or format tag configuration");

    const bool isa_supports_avx2 = is_superset(isa, avx2);
    if (is_training() && fuse_norm_relu()) {
        VDISPATCH_BNORM(isa_supports_avx2, VERBOSE_UNSUPPORTED_ISA);
        init_default_ws(1);
    }

    VDISPATCH_BNORM(!(memory_desc_wrapper(src_md()).padded_dims()[1] != C()
                            && !isa_supports_avx2),
            VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "bad padded dimensions for current isa");

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_tbb_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_fwd_t<
        isa>::jit_uni_tbb_batch_normalization_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(bnorm_driver_,
            new bnorm_tbb_impl::driver_t<isa>(pd(), pd()->tag_kind_)));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto scale = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SHIFT);

    auto mean = pd()->stats_is_src() ? const_cast<acc_data_t *>(
                        CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN))
                                     : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_MEAN);
    auto var = pd()->stats_is_src()
            ? const_cast<acc_data_t *>(
                    CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE))
            : CTX_OUT_MEM(acc_data_t *, DNNL_ARG_VARIANCE);

    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, DNNL_ARG_WORKSPACE);

    auto scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->exec_fwd(src, dst, scale, shift, mean, var, ws, scratchpad);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_fwd_t<
        isa>::~jit_uni_tbb_batch_normalization_fwd_t()
        = default;

template struct jit_uni_tbb_batch_normalization_fwd_t<sse41>;
template struct jit_uni_tbb_batch_normalization_fwd_t<avx2>;
template struct jit_uni_tbb_batch_normalization_fwd_t<avx512_core>;

/* bwd */
template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_bwd_t<isa>::pd_t::init(
        engine_t *engine) {
    VDISPATCH_BNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);

    // disabling verbose dispatch checks for unsupported isa for better readability
    if (!mayiuse(isa)) return status::unimplemented;

    VDISPATCH_BNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_BNORM(one_of(src_md()->data_type, f32, bf16, f16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_BNORM(src_md()->data_type == diff_src_md()->data_type,
            VERBOSE_INCONSISTENT_DT, "src", "diff_src");
    VDISPATCH_BNORM(diff_src_md()->data_type == diff_dst_md()->data_type,
            VERBOSE_INCONSISTENT_DT, "diff_src", "diff_dst");
    VDISPATCH_BNORM(IMPLICATION(src_md()->data_type == bf16,
                            is_superset(isa, avx512_core)),
            VERBOSE_ISA_DT_MISMATCH);
    // Note: re-using avx512_core implementation for f16. This is okay
    // as currently, we do not support binary post-ops for this
    // primitive.
    VDISPATCH_BNORM(
            IMPLICATION(src_md()->data_type == f16,
                    is_superset(isa, avx512_core) && mayiuse(avx512_core_fp16)),
            VERBOSE_ISA_DT_MISMATCH);
    VDISPATCH_BNORM(check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_FEATURE,
            "unsupported scale or shift data type");
    VDISPATCH_BNORM(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_BNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_BNORM(memory_desc_wrapper(diff_src_md())
                    == memory_desc_wrapper(diff_dst_md()),
            VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");

    // BN+Add+Relu fusion is not currently implemented
    VDISPATCH_BNORM(!(fuse_norm_add_relu()), VERBOSE_UNSUPPORTED_FEATURE,
            "sum+relu post-ops configuration is not supported");

    const format_tag_t blocked_tag = is_superset(isa, avx512_core)
            ? utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c)
            : utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);

    const format_tag_t blocked_format
            = memory_desc_matches_tag(*src_md(), blocked_tag)
            ? blocked_tag
            : format_tag::undef;
    const format_tag_t nspc_format
            = memory_desc_matches_one_of_tag(*src_md(), nc, nwc, nhwc, ndhwc);

    if (memory_desc_matches_tag(*diff_src_md(), blocked_format))
        tag_kind_ = jit_memory_tag_kind_t::blocked;
    else if (memory_desc_matches_tag(*diff_src_md(), nspc_format)) {
        tag_kind_ = jit_memory_tag_kind_t::nspc;
        const int simd_w = get_simd_w<isa>(tag_kind_);
        VDISPATCH_BNORM(
                C() % simd_w == 0, VERBOSE_BLOCKING_FAIL, "bad src dimensions");
    } else
        VDISPATCH_BNORM(false, VERBOSE_UNSUPPORTED_TAG);

    const bool isa_supports_avx2 = is_superset(isa, avx2);
    VDISPATCH_BNORM(!((memory_desc_wrapper(src_md()).padded_dims()[1] != C()
                            && !isa_supports_avx2)),
            VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "bad padded dimensions for current isa");

    if (fuse_norm_relu()) {
        VDISPATCH_BNORM(isa_supports_avx2, VERBOSE_UNSUPPORTED_ISA);
        init_default_ws(1);
        VDISPATCH_BNORM(compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);
    }

    auto scratchpad = scratchpad_registry().registrar();
    bnorm_tbb_impl::driver_t<isa>::init_scratchpad(scratchpad, this);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_bwd_t<
        isa>::jit_uni_tbb_batch_normalization_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_bwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(bnorm_driver_,
            new bnorm_tbb_impl::driver_t<isa>(pd(), pd()->tag_kind_)));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_tbb_batch_normalization_bwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto mean = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_MEAN);
    auto var = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto scale = CTX_IN_MEM(const acc_data_t *, DNNL_ARG_SCALE);
    auto ws = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(void *, DNNL_ARG_DIFF_SRC);
    auto diff_scale = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SCALE);
    auto diff_shift = CTX_OUT_MEM(acc_data_t *, DNNL_ARG_DIFF_SHIFT);

    auto scratchpad = ctx.get_scratchpad_grantor();

    bnorm_driver_->exec_bwd(src, diff_src, diff_dst, scale, diff_scale,
            diff_shift, mean, var, ws, scratchpad);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_tbb_batch_normalization_bwd_t<
        isa>::~jit_uni_tbb_batch_normalization_bwd_t()
        = default;

template struct jit_uni_tbb_batch_normalization_bwd_t<sse41>;
template struct jit_uni_tbb_batch_normalization_bwd_t<avx2>;
template struct jit_uni_tbb_batch_normalization_bwd_t<avx512_core>;
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
