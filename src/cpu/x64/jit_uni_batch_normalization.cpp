/*******************************************************************************
* Copyright 2017-2023 Intel Corporation
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
#include <functional>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_batch_normalization_utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_uni_batch_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace memory_tracking::names;

using namespace Xbyak;
namespace barrier = simple_barrier;

using acc_data_t = float;

namespace {
dim_t get_c_padded(const batch_normalization_pd_t *pd) {
    return pd->src_md()->padded_dims[1];
}

bool is_nspc(const memory_desc_wrapper &d) {
    using namespace format_tag;
    const bool is_nspc = d.matches_one_of_tag(nc, nwc, nhwc, ndhwc);
    return is_nspc;
}
} // namespace

struct jit_bnorm_conf_t {
    // TODO: put all needed info here to avoid duplicate work and potentially
    // diverging definitions of derived parameters
    const batch_normalization_pd_t *pd_;

    int simd_w_ {0};
    size_t dt_size_ {0};
    bool is_nspc_ {false};

    // thread partition info
    bool do_blocking_ {false};
    bool is_spatial_thr_ {false};
    dim_t C_blks_per_iter_ {0};
    int C_nthr_ {0};
    int N_nthr_ {0};
    int S_nthr_ {0};
    int64_t iters_ {0};
    // C_blks and thread partition can change for last iteration
    dim_t C_blks_last_iter_ {0};
    int C_nthr_last_iter_ {0};
    int N_nthr_last_iter_ {0};
    int S_nthr_last_iter_ {0};

    jit_bnorm_conf_t(const batch_normalization_pd_t *pd, int nthr, int simd_w)
        : pd_(pd), simd_w_(simd_w) {

        const dim_t N = pd_->MB();
        const dim_t C_PADDED = get_c_padded(pd_);
        const dim_t D = pd_->D();
        const dim_t H = pd_->H();
        const dim_t W = pd_->W();
        const dim_t SP = D * H * W;

        const memory_desc_wrapper src_d(pd_->src_md());
        is_nspc_ = is_nspc(src_d);

        dt_size_ = types::data_type_size(pd_->src_md()->data_type);
        size_t data_size = dt_size_ * N * C_PADDED * SP;
        const size_t l3_size = platform::get_per_core_cache_size(3) * nthr;
        // TODO: cache balancing for nspc
        const size_t l3_filling_factor = 4;
        do_blocking_ = !is_nspc_ && data_size >= l3_size / l3_filling_factor;

        // find thread partition over N, C_blks and SP

        const dim_t C_blks = C_PADDED / simd_w_;

        if (do_blocking_) {
            const int num_tensors = pd_->is_fwd() ? 1 : 2;
            const size_t working_set_size
                    = dt_size_ * (N * SP * simd_w_) * num_tensors;
            bnorm_utils::cache_balance(working_set_size, C_blks, N, nthr,
                    C_blks_per_iter_, iters_);
            C_blks_last_iter_ = C_blks - (iters_ - 1) * C_blks_per_iter_;
        } else {
            C_blks_per_iter_ = C_blks;
            iters_ = 1;
        }

        is_spatial_thr_
                = this->thread_partition(/* is_spatial_thr_ = */ true, nthr,
                        /* dimensions */
                        N, C_blks_per_iter_, SP,
                        /* outputs */
                        C_nthr_, N_nthr_, S_nthr_);

        if (iters_ > 1)
            this->thread_partition(is_spatial_thr_, nthr,
                    /* dimensions */
                    N, C_blks_last_iter_, SP,
                    /* outputs */
                    C_nthr_last_iter_, N_nthr_last_iter_, S_nthr_last_iter_);
    }

    // given nthr and shape of problem, choose the thread partition
    // to use (ie set N_nthr, C_nthr, and S_nthr)
    bool thread_partition(bool spatial_thr_allowed, int nthr, dim_t N,
            dim_t C_blks, dim_t SP, int &C_nthr, int &N_nthr, int &S_nthr) {
        if (((nthr <= C_blks) && IMPLICATION(is_nspc_, N == 1))
                || !dnnl_thr_syncable()) {
            C_nthr = nthr;
            N_nthr = 1;
            S_nthr = 1;
        } else {
            if (is_nspc_) {
                if (C_blks <= 8)
                    C_nthr = 1;
                else if (nthr >= 8 && C_blks <= 32)
                    C_nthr = 8;
                else {
                    C_nthr = (int)math::gcd((dim_t)nthr, C_blks);
                    // Unroll by channels in JIT kernel
                    if ((C_nthr == C_blks) || (C_nthr == nthr)) C_nthr = 1;
                }
                N_nthr = (int)nstl::min<dim_t>(N, nthr / C_nthr);
                // heuristic for training on avx512_core_amx
                // TODO: test heuristic when global stats flag is set
                if (!pd_->use_global_stats() && 0 < dt_size_ && 0 < simd_w_
                        && 1 < C_nthr && nthr <= N
                        && mayiuse(avx512_core_amx)) {
                    const size_t data_size
                            = dt_size_ * N * SP * C_blks * simd_w_;
                    const size_t C_split_data_size
                            = utils::div_up(data_size, N_nthr);
                    const size_t N_split_data_size
                            = utils::div_up(data_size, nthr);
                    const size_t l2_size_per_core
                            = platform::get_per_core_cache_size(2);
                    const size_t l3_size_per_core
                            = platform::get_per_core_cache_size(3);
                    const size_t cache_size_per_core
                            = l2_size_per_core + l3_size_per_core;
                    // if current split is too big for cache, better to split by N
                    const bool condition1
                            = cache_size_per_core < C_split_data_size;
                    // if split by N is also too big for cache, bwd is better off as it was
                    const bool condition2 = pd_->is_fwd()
                            || cache_size_per_core >= N_split_data_size;
                    if (condition1 && condition2) {
                        C_nthr = 1;
                        N_nthr = nthr;
                    }
                }
                S_nthr = (int)nstl::min<dim_t>(SP, nthr / (C_nthr * N_nthr));
            } else {
                if (do_blocking_) {
                    N_nthr = (int)nstl::min<dim_t>(N, nthr);
                    C_nthr = (int)nstl::min<dim_t>(C_blks, nthr / N_nthr);
                    S_nthr = (int)nstl::min<dim_t>(
                            SP, nthr / (C_nthr * N_nthr));
                } else {
                    C_nthr = (int)math::gcd((dim_t)nthr, C_blks);
                    N_nthr = (int)nstl::min<dim_t>(N, nthr / C_nthr);
                    S_nthr = (int)nstl::min<dim_t>(
                            SP, nthr / (C_nthr * N_nthr));
                }
            }

            if (!spatial_thr_allowed) S_nthr = 1;

            if (S_nthr < 1) S_nthr = 1;
        }

        // spatial_thr_allowed is meant to help maintain
        // consistent decisions about spatial threading
        // between mutiple invocations of this routine.
        // It is caller's responsibility to check the
        // return value and pass it as a flag to the
        // next call if needed.
        if (S_nthr == 1) spatial_thr_allowed = false;

        return spatial_thr_allowed;
    }
};

template <cpu_isa_t isa>
struct jit_bnorm_t : public jit_generator {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        size_t N_ithr, N_nthr;
        size_t coff_max, soff_max;
        size_t mb_stride_Bc, spat_size, spat_size_loc;
        size_t S_s, S_tail;
        size_t is_cblk_tail;
        acc_data_t chan_size, eps, one;
        const acc_data_t *scale;
        const acc_data_t *shift;
        const acc_data_t *mean, *var;
        const acc_data_t *diff_scale;
        const acc_data_t *diff_shift;
        const void *src, *dst;
        const void *diff_src, *diff_dst;
        const acc_data_t *rbuf1, *rbuf2;
        const uint8_t *ws;
        barrier::ctx_64_t *barrier;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_bnorm_t)

    /* cpu specific part */
    using Vmm = typename utils::conditional3<isa == sse41, Xmm, isa == avx2,
            Ymm, Zmm>::type;
    const AddressFrame &vmmword = (isa == sse41) ? xword
            : (isa == avx2)                      ? yword
                                                 : zword;

    const int vlen = isa == sse41 ? 32 : cpu_isa_traits<isa>::vlen;
    int vlen_spat_data_
            = 0; // set by ctor depending on data type (xF16 or FP32);

    const batch_normalization_pd_t *pd_ = nullptr;
    const jit_bnorm_conf_t *jbp_ = nullptr;
    bool is_bf16_ = false;
    bool is_f16_ = false;
    bool is_avx2_ne_xf16_ = false;

    Reg64 reg_param = abi_param1;

    Reg64 reg_scale = rbx;
    Reg64 reg_rbuf1 = abi_not_param1;
    Reg64 reg_rbuf2 = rdx;
    Reg64 reg_coff_max_fwd_copy = reg_rbuf2;

    Reg64 reg_mean = rbp;
    Reg64 reg_var = reg_param;
    Reg64 reg_diff_scale = rax;
    Reg64 reg_coff_max_bwd_copy = reg_diff_scale;
    Reg64 reg_shift = reg_rbuf1;

    Reg64 reg_coff = r8;
    Reg64 reg_coff_max = r9;
    Reg64 reg_soff = r10;
    Reg64 reg_soff_max = r11;
    Reg64 reg_diff_shift = reg_soff_max;
    Reg64 reg_ctr = r12;
    Reg64 reg_roff = r13;

    Reg64 reg_mb_stride_Bc = r14;
    Reg64 reg_soff_nspc = reg_mb_stride_Bc;

    Reg64 reg_src = r15;
    Reg64 reg_diff_src = reg_rbuf1;
    Reg64 reg_dst = rsi;
    Reg64 reg_diff_dst = reg_dst;

    Reg64 reg_tmp_off = reg_roff;

    // Reuse loop counters
    Reg64 reg_bar = reg_coff;
    Reg64 reg_nnthr = reg_soff; // must be usable w/ loops over coff
    Reg64 reg_tmp = reg_ctr;

    // Relu section
    bool with_relu = false, with_relu_inf_only = false;
    Reg64 reg_ws = reg_roff;
    Reg64 reg_tmp_alpha = reg_diff_scale; // required in sse41
    Label l_relu_mask_avx2;
    Opmask kstore_mask = Opmask(1);

    // channel tail processing
    Opmask ktail_mask = Opmask(2);

    // FP32->BF16 emulation
    bf16_emulation_t *bf16_emu_ {nullptr};
    Reg64 reg_bf16_tmp = reg_tmp;
    Zmm bf16_emu_reserved_1 = Zmm(17);
    Zmm bf16_emu_reserved_2 = Zmm(18);
    Zmm bf16_emu_reserved_3 = Zmm(19);
    Zmm bf16_emu_reserved_4 = Zmm(20);

    size_t unroll_blocks;
    size_t unroll_regs;
    Vmm vdiff_beta = Vmm(isa == avx512_core ? 21 : 6);
    Vmm vdiff_gamma = Vmm(isa == avx512_core ? 22 : 7);
    Vmm vsqrtvar = Vmm(isa == avx512_core ? 23 : 8);
    Vmm vone = Vmm(isa == avx512_core ? 24 : 9);
    Vmm vmean = Vmm(isa == avx512_core ? 25 : 10);
    Vmm vgamma = Vmm(isa == avx512_core ? 26 : 11);
    Vmm vbeta = Vmm(isa == avx512_core ? 27 : 12);
    Vmm veps = Vmm(isa == avx512_core ? 28 : 13);
    Vmm vchan_size = Vmm(isa == avx512_core ? 29 : 14);
    Vmm vtail_mask = Vmm(isa == avx512_core ? 30 : 15);
    Vmm vtmp = Vmm(isa == avx512_core ? 31 : 5);
    Vmm vsrc_aux = vdiff_gamma; // used for xf16 with nspc ON AVX2
    Vmm vdst_aux = vdiff_gamma; // used for ReLU in AVX2 & sse41
    Vmm vmask = Vmm(0);
    Vmm vzero; // is_fwd() ? vdiff_beta : vbeta

    size_t spat_size;
    size_t chan_data_offt;
    size_t spat_step;
    size_t mb_offt;
    size_t ws_mb_offt;

    enum {
        stack_off_N_nthr = 0,
        stack_off_N_ithr = 8,
        stack_off_src = 16,
        stack_off_dst = 24,
        stack_off_diff_src = 32,
        stack_off_diff_dst = 40,
        stack_off_diff_scale = 48,
        stack_off_ws = 56,
        stack_off_barrier = 64,
        stack_off_spat_size_loc = 72,
        stack_off_s_s = 80,
        stack_off_s_tail = 88,
        stack_off_is_cblk_tail = 96,
        stack_off_ws_off_copy = 104,
        stack_off_shift = 112,
        stack_off_diff_shift = 120,
        stack_off_soff_max = 128,
        stack_off_relu_alpha = 136,
        stack_size_required = 144,
    };

    bool is_xf16() { return is_bf16_ || is_f16_; }
    int bit_shift() { return 5 - is_xf16(); }

    bool use_bf16_emulation() {
        return is_bf16_ && isa == avx512_core && !mayiuse(avx512_core_bf16);
    }

    bool stream_store_supported() {
        // keep original behavior for f32
        if (!is_xf16()) return true;
        // TODO: check performance of heuristic for other cases, such as:
        // blocked layout, pre-avx512_core_amx machines, and f32 datatype.
        const bool is_applicable = jbp_->is_nspc_ && mayiuse(avx512_core_amx);
        if (!is_applicable) return false;
        const size_t l2_size_per_core = platform::get_per_core_cache_size(2);
        const size_t l3_size_per_core = platform::get_per_core_cache_size(3);
        const size_t cache_size_per_core = l2_size_per_core + l3_size_per_core;
        const size_t buffer_count = pd_->is_fwd() ? 2 : 3;
        const size_t data_size = buffer_count * jbp_->dt_size_ * pd_->MB()
                * pd_->C() * pd_->D() * pd_->H() * pd_->W();
        // do not divide by C_nthr for nspc layout
        const size_t data_size_per_core
                = data_size / (jbp_->N_nthr_ * jbp_->S_nthr_);
        return cache_size_per_core < data_size_per_core;
    }

    bool is_c_padded() const {
        const memory_desc_wrapper data_d(pd_->src_md());
        return pd_->C() != data_d.padded_dims()[1];
    }

    void compute_static_strides() {
        spat_size = pd_->D() * pd_->W() * pd_->H();
        chan_data_offt = pd_->C() * sizeof(acc_data_t);
        spat_step = jbp_->is_nspc_ ? chan_data_offt / (1 + is_xf16())
                                   : vlen_spat_data_;
        mb_offt = spat_step * spat_size;
        ws_mb_offt = (spat_step / (is_xf16() ? 16 : 32)) * spat_size;
    }

    void load_common_params() {
#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_rbuf1, ptr[reg_param + PARAM_OFF(rbuf1)]);
        if (!pd_->is_fwd()) mov(reg_rbuf2, ptr[reg_param + PARAM_OFF(rbuf2)]);
        mov(reg_coff_max, ptr[reg_param + PARAM_OFF(coff_max)]);
        mov(reg_soff_max, ptr[reg_param + PARAM_OFF(soff_max)]);
        mov(reg_mb_stride_Bc, ptr[reg_param + PARAM_OFF(mb_stride_Bc)]);
        shl(reg_coff_max, 2);

        mov(reg_mean, ptr[reg_param + PARAM_OFF(mean)]);
        mov(reg_scale, ptr[reg_param + PARAM_OFF(scale)]);

        uni_vbroadcastss(vchan_size, vmmword[reg_param + PARAM_OFF(chan_size)]);
        uni_vbroadcastss(vone, vmmword[reg_param + PARAM_OFF(one)]);
        uni_vbroadcastss(veps, vmmword[reg_param + PARAM_OFF(eps)]);

        mov(reg_tmp, ptr[reg_param + PARAM_OFF(N_nthr)]);
        mov(ptr[rsp + stack_off_N_nthr], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(N_ithr)]);
        mov(ptr[rsp + stack_off_N_ithr], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(src)]);
        mov(ptr[rsp + stack_off_src], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(dst)]);
        mov(ptr[rsp + stack_off_dst], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_src)]);
        mov(ptr[rsp + stack_off_diff_src], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_dst)]);
        mov(ptr[rsp + stack_off_diff_dst], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(ws)]);
        mov(ptr[rsp + stack_off_ws], reg_tmp);
        mov(reg_tmp, ptr[reg_param + PARAM_OFF(barrier)]);
        mov(ptr[rsp + stack_off_barrier], reg_tmp);
        if (jbp_->is_spatial_thr_) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(spat_size_loc)]);
            mov(ptr[rsp + stack_off_spat_size_loc], reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(S_s)]);
            mov(ptr[rsp + stack_off_s_s], reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(S_tail)]);
            mov(ptr[rsp + stack_off_s_tail], reg_tmp);
        }
        if (is_c_padded()) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(is_cblk_tail)]);
            mov(ptr[rsp + stack_off_is_cblk_tail], reg_tmp);
        }

        if (pd_->is_fwd()) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(shift)]);
            mov(ptr[rsp + stack_off_shift], reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(var)]);
            mov(reg_var, reg_tmp);
        } else {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_scale)]);
            mov(ptr[rsp + stack_off_diff_scale], reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(diff_shift)]);
            mov(ptr[rsp + stack_off_diff_shift], reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(soff_max)]);
            mov(ptr[rsp + stack_off_soff_max], reg_tmp);
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(var)]);
            mov(reg_var, reg_tmp);
        }
        if (with_relu_inf_only && pd_->alpha() != 0.f) {
            mov(reg_tmp, float2int(pd_->alpha()));
            mov(ptr[rsp + stack_off_relu_alpha], reg_tmp);
        }
#undef PARAM_OFF
    }

    void prepare_tail_mask_avx512_common() {
        if (!is_c_padded()) return;

        const int tail = pd_->C() % (int)(vlen / sizeof(float));
        const int mask = (1 << tail) - 1;

        Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
    }

    void prepare_tail_mask_avx2_common() {
        if (!is_c_padded()) return;

        const int tail = pd_->C() % (int)(vlen / sizeof(float));
        static const uint32_t mask[16] = {0xffffffff, 0xffffffff, 0xffffffff,
                0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0,
                0, 0, 0, 0, 0, 0, 0};

        mov(reg_tmp, reinterpret_cast<size_t>(&mask[8 - tail]));
        vmovups(vtail_mask, ptr[reg_tmp]);
    }

    void prepare_relu() {
        with_relu = pd_->is_fwd() ? pd_->with_relu_post_op(pd_->is_training())
                        || pd_->fuse_norm_relu()
                                  : pd_->fuse_norm_relu();
        with_relu_inf_only = with_relu && pd_->is_fwd()
                && !(pd_->fuse_norm_relu() && pd_->is_training());

        vzero = pd_->is_fwd() ? vdiff_beta : vbeta;
        if (with_relu) {
            uni_vpxor(vzero, vzero, vzero);
            if (!pd_->is_fwd() && isa == avx2) prepare_l_relu_mask_avx2();
        }
    }

    void prepare_l_relu_mask_avx2() {
        Label l_mask_after;
        jmp(l_mask_after);
        align(32);
        L(l_relu_mask_avx2); /* [0x80 0x40 0x20 0x10 0x08 0x04 0x02 0x01] */
        for (int i = 0; i < 8; ++i)
            dd(1 << i);
        L(l_mask_after);
    }

    void fwd_process_relu_avx2(Vmm vdst, int offt) {
        Reg64 reg_store_mask = reg_diff_scale;
        Reg64 reg_soff_loc = jbp_->is_nspc_ ? reg_soff_nspc : reg_soff;
        shr(reg_soff_loc, bit_shift());
        vcmpps(vtmp, vzero, vdst, _cmp_lt_os);
        vmovmskps(reg_store_mask, vtmp);
        mov(ptr[reg_ws + reg_soff_loc + offt / (1 << bit_shift())],
                reg_store_mask.cvt8());
        vblendvps(vdst, vzero, vdst, vtmp);
        shl(reg_soff_loc, bit_shift());
    }

    void fwd_process_relu_avx512_common(Vmm vdst, int offt = 0) {
        Reg64 reg_soff_loc = jbp_->is_nspc_ ? reg_soff_nspc : reg_soff;
        shr(reg_soff_loc, bit_shift());
        vcmpps(kstore_mask, vzero, vdst, _cmp_lt_os);
        kmovw(ptr[reg_ws + reg_soff_loc + offt / (1 << bit_shift())],
                kstore_mask);
        vblendmps(vdst | kstore_mask, vzero, vdst);
        shl(reg_soff_loc, bit_shift());
    }

    void fwd_process_relu_alpha(Vmm vmm_dst) {
        if (isa == avx512_core)
            fwd_process_relu_alpha_avx512_common(vmm_dst);
        else {
            assert(utils::one_of(isa, avx2, sse41));
            if (vmm_dst.getIdx() == 0) {
                uni_vmovups(vdst_aux, vmm_dst);
                fwd_process_relu_alpha_avx2(vdst_aux);
                uni_vmovups(Vmm(0), vdst_aux);
            } else
                fwd_process_relu_alpha_avx2(vmm_dst);
        }
    }
    void fwd_process_relu_alpha_avx512_common(Vmm vmm_dst) {
        const Xmm xmm_tmp = Xmm(vtmp.getIdx());
        vmovq(xmm_tmp, ptr[rsp + stack_off_relu_alpha]);
        vbroadcastss(vtmp, xmm_tmp);
        vcmpps(kstore_mask, vzero, vmm_dst, _cmp_lt_os);
        vmulps(vtmp, vmm_dst, vtmp);
        vblendmps(vmm_dst | kstore_mask, vtmp, vmm_dst);
    }

    void fwd_process_relu_alpha_avx2(Vmm vmm_dst) {
        const Xmm xmm_tmp = Xmm(vtmp.getIdx());
        uni_vpxor(vmask, vmask, vmask);
        if (isa == sse41) {
            mov(reg_tmp_alpha, ptr[rsp + stack_off_relu_alpha]);
            uni_vmovq(xmm_tmp, reg_tmp_alpha);
        } else
            vmovq(xmm_tmp, ptr[rsp + stack_off_relu_alpha]);
        uni_vbroadcastss(vtmp, xmm_tmp);
        uni_vcmpps(vmask, vmm_dst, vzero, _cmp_lt_os);
        uni_vmulps(vtmp, vtmp, vmm_dst);
        uni_vblendvps(vmm_dst, vmm_dst, vtmp, vmask);
    }

    void bwd_process_relu_avx2(Vmm vdiff_dst, int offt) {
        shr(reg_soff, bit_shift());
        vpbroadcastb(vtmp, ptr[reg_ws + reg_soff + offt / (1 << bit_shift())]);
        vpand(vtmp, vtmp, ptr[rip + l_relu_mask_avx2]);
        vpcmpeqd(vtmp, vtmp, ptr[rip + l_relu_mask_avx2]);
        vblendvps(vdiff_dst, vzero, vdiff_dst, vtmp);
        shl(reg_soff, bit_shift());
    }

    void bwd_process_relu_avx512_common(Vmm vdiff_dst, int offt = 0) {
        shr(jbp_->is_nspc_ ? reg_soff_nspc : reg_soff, bit_shift());
        kmovw(kstore_mask,
                ptr[reg_ws + (jbp_->is_nspc_ ? reg_soff_nspc : reg_soff)
                        + offt / (1 << bit_shift())]);
        vmovups(vdiff_dst | kstore_mask | T_z, vdiff_dst);
        shl(jbp_->is_nspc_ ? reg_soff_nspc : reg_soff, bit_shift());
    }

    void merge_interleaved_to_plain(
            const Vmm &vmm_even, const Vmm &vmm_odd, const Vmm &vmm_aux0) {
        Ymm ymm_even = Ymm(vmm_even.getIdx());
        Ymm ymm_odd = Ymm(vmm_odd.getIdx());
        Ymm ymm_aux0 = Ymm(vmm_aux0.getIdx());
        Ymm ymm_aux1 = ymm_odd;

        vpunpckldq(ymm_aux0, ymm_even, ymm_odd);
        vpunpckhdq(ymm_aux1, ymm_even, ymm_odd);
        vperm2i128(ymm_even, ymm_aux0, ymm_aux1, 0x20);
        vperm2i128(ymm_odd, ymm_aux0, ymm_aux1, 0x31);
    }
    void uni_vmovups_spat_data(
            const Vmm &vmm_even, const Vmm &vmm_odd, const Address &addr) {
        // load two simd_w data from addr into two registers
        if (is_bf16_) {
            // convert bf16 input to f32
            vcvtneebf162ps(vmm_even, addr);
            vcvtneobf162ps(vmm_odd, addr);
        } else if (is_f16_) {
            vcvtneeph2ps(vmm_even, addr);
            vcvtneoph2ps(vmm_odd, addr);
        } else
            assert(!"unsupported data type!");
    }

    void uni_vmovups_spat_data(
            const Operand &dst, const Operand &src, bool is_nt_store = false) {
        if (dst.isMEM()) {
            if (is_bf16_) {
                constexpr bool isAvx2 = isa == avx2;
                const typename std::conditional<isAvx2, Xmm, Ymm>::type
                        dst_reg {src.getIdx()};
                const typename std::conditional<isAvx2, Ymm, Zmm>::type
                        src_reg {src.getIdx()};

                // convert f32 output to bf16
                if (!use_bf16_emulation())
                    vcvtneps2bf16(dst_reg, src_reg,
                            mayiuse(avx512_core) ? Xbyak::EvexEncoding
                                                 : Xbyak::VexEncoding);
                else
                    bf16_emu_->vcvtneps2bf16(dst_reg, src_reg);

                // store to memory
                if (is_nt_store)
                    uni_vmovntps(dst.getAddress(), dst_reg);
                else
                    uni_vmovups(dst.getAddress(), dst_reg);
            } else if (is_f16_) {
                auto src_reg = Vmm(src.getIdx());
                auto dst_reg =
                        typename vreg_traits<Vmm>::Vmm_lower_t(src.getIdx());
                if (is_nt_store) {
                    if (mayiuse(avx512_core_fp16))
                        vcvtps2phx(dst_reg, src_reg);
                    else
                        vcvtps2ph(dst_reg, src_reg, _op_mxcsr);
                    uni_vmovntps(dst.getAddress(), dst_reg);
                } else {
                    vcvtps2ph(dst.getAddress(), src_reg, _op_mxcsr);
                }
            } else {
                if (is_nt_store)
                    uni_vmovntps(dst.getAddress(), Vmm(src.getIdx()));
                else
                    uni_vmovups(dst.getAddress(), Vmm(src.getIdx()));
            }
        } else {
            if (is_bf16_) {
                // convert bf16 input to f32
                vpmovzxwd(Vmm(dst.getIdx()), src.getAddress());
                vpslld(Vmm(dst.getIdx()), Vmm(dst.getIdx()), 0x10);
            } else if (is_f16_) {
                if (mayiuse(avx512_core_fp16))
                    vcvtph2psx(Vmm(dst.getIdx()), src.getAddress());
                else
                    vcvtph2ps(Vmm(dst.getIdx()), src.getAddress());
            } else {
                uni_vmovups(Vmm(dst.getIdx()), src.getAddress());
            }
        }
    }

    void uni_vmovups_tail_avx2_common(
            const Operand &dst, const Operand &src, Label &l_ret) {
        if (dst.isMEM()) {
            vmaskmovps(dst.getAddress(), vtail_mask, Vmm(src.getIdx()));
        } else {
            vmaskmovps(Vmm(dst.getIdx()), vtail_mask, src.getAddress());
        }
        jmp(l_ret);
    }

    void uni_vmovups_tail_avx512_common(
            const Operand &dst, const Operand &src, Label &l_ret) {
        if (dst.isMEM())
            uni_vmovups(dst.getAddress() | ktail_mask | T_z, Vmm(src.getIdx()));
        else
            uni_vmovups(Vmm(dst.getIdx()) | ktail_mask | T_z, src.getAddress());

        jmp(l_ret);
    }

    void uni_vmovups_maybe_tail(const Operand &dst, const Operand &src) {
        Label l_no_mask, l_ret;

        if (is_c_padded()) {
            mov(reg_tmp, ptr[rsp + stack_off_is_cblk_tail]);
            cmp(reg_tmp, 0);
            jz(l_no_mask);

            lea(reg_tmp, ptr[reg_coff + vlen]);
            cmp(reg_tmp, reg_coff_max);
            jl(l_no_mask);
            assert(isa == avx512_core || isa == avx2);
            if (isa == avx512_core)
                uni_vmovups_tail_avx512_common(dst, src, l_ret);
            else if (isa == avx2)
                uni_vmovups_tail_avx2_common(dst, src, l_ret);
        }
        L(l_no_mask);
        if (dst.isMEM())
            uni_vmovups(dst.getAddress(), Vmm(src.getIdx()));
        else
            uni_vmovups(Vmm(dst.getIdx()), src.getAddress());

        L(l_ret);
    }

    void barrier() {
        mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
        mov(reg_bar, ptr[rsp + stack_off_barrier]);
        simple_barrier::generate(*this, reg_bar, reg_nnthr);
    }

    Address mean_ptr(size_t offt = 0) {
        return vmmword[reg_mean + reg_coff + offt];
    }

    Address var_ptr(size_t offt = 0) {
        return vmmword[reg_var + reg_coff + offt];
    }

    Address diff_gamma_ptr(size_t offt = 0) {
        return vmmword[reg_diff_scale + reg_coff + offt];
    }

    Address diff_beta_ptr(size_t offt = 0) {
        return vmmword[reg_diff_shift + reg_coff + offt];
    }

    Address gamma_ptr(size_t offt = 0) {
        return vmmword[reg_scale + reg_coff + offt];
    }

    Address beta_ptr(size_t offt = 0) {
        return vmmword[reg_shift + reg_coff + offt];
    }

    template <typename init_t, typename body_t, typename fini_t>
    void spat_loop(size_t len, size_t blocks, size_t regs, init_t init,
            body_t body, fini_t fini) {
        size_t factor = regs * blocks;
        size_t loop_unroll = len / factor * factor;
        size_t loop_tail = len - loop_unroll;
        size_t num_active_regs = (len < regs) ? len : regs;
        for (size_t i = 0; i < num_active_regs; i++)
            init(i);
        if (loop_unroll) {
            if (jbp_->is_spatial_thr_) {
                mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
                add(reg_soff, ptr[rsp + stack_off_s_s]);
            } else {
                mov(reg_ctr, loop_unroll);
            }
            Label label;
            L(label);
            {
                for (size_t i = 0; i < factor; i++) {
                    size_t base_reg = i % regs;
                    body(base_reg, i);
                }
                add(reg_soff, factor * spat_step);
                sub(reg_ctr, factor);
                jnz(label);
            }
            if (jbp_->is_spatial_thr_) {
                add(reg_soff, ptr[rsp + stack_off_s_tail]);
            }
        }

        for (size_t i = 0; i < loop_tail; i++) {
            size_t base_reg = i % regs;
            body(base_reg, i);
        }
        if (loop_tail) add(reg_soff, loop_tail * spat_step);

        for (size_t i = 0; i < num_active_regs; i++)
            fini(i);
    }

    void mean_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_vmovups(Vmm(0), vmmword[reg_rbuf1 + reg_coff]);
            spat_loop(
                    spat_size, unroll_blocks, unroll_regs,
                    [this](size_t base_reg) {
                        Vmm v = Vmm(base_reg * 2);
                        if (base_reg) uni_vpxor(v, v, v);
                    },
                    [this](size_t base_reg, size_t i) {
                        Vmm v0 = Vmm(base_reg * 2 + 0);
                        Vmm v1 = Vmm(base_reg * 2 + 1);
                        size_t offt = i * vlen_spat_data_;
                        uni_vmovups_spat_data(
                                v1, vmmword[reg_src + reg_soff + offt]);
                        uni_vaddps(v0, v0, v1);
                    },
                    [this](size_t base_reg) {
                        Vmm b = Vmm(0);
                        Vmm v = Vmm(base_reg * 2);
                        if (base_reg) uni_vaddps(b, b, v);
                    });
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));

            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(ch_label);
        }
    }

    void mean_variance_nspc(
            const int num_ch_blks, int num_spat_pts, bool compute_mean) {

        auto mean_compute_avx2_ne_xf16 = [this](int num_ch_blks,
                                                 int num_spat_pts) {
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                for (int ch_idx = 0; ch_idx < num_ch_blks; ch_idx += 2) {
                    const int offt = ch_idx * vlen_spat_data_;
                    const bool is_ch_blks_tail = num_ch_blks - ch_idx < 2;
                    const Vmm vsrc_even = vtmp;
                    const Vmm vsrc_odd = vsrc_aux;
                    if (is_ch_blks_tail)
                        uni_vmovups_spat_data(vsrc_even,
                                vmmword[reg_src + reg_soff_nspc + offt]);
                    else
                        uni_vmovups_spat_data(vsrc_even, vsrc_odd,
                                vmmword[reg_src + reg_soff_nspc + offt]);

                    uni_vaddps(Vmm(ch_idx), Vmm(ch_idx), vsrc_even);
                    if (!is_ch_blks_tail)
                        uni_vaddps(Vmm(ch_idx + 1), Vmm(ch_idx + 1), vsrc_odd);
                }
                add(reg_soff_nspc, spat_step);
            }
        };

        auto variance_compute_avx2_ne_xf16 = [this](int num_ch_blks,
                                                     int num_spat_pts) {
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                for (int ch_idx = 0; ch_idx < num_ch_blks; ch_idx += 2) {
                    const int offt = ch_idx * vlen_spat_data_;
                    const bool is_ch_blks_tail = num_ch_blks - ch_idx < 2;
                    const Vmm vsrc_even = vtmp;
                    const Vmm vsrc_odd = vsrc_aux;
                    const Vmm vmean_ch_even = Vmm(ch_idx + num_ch_blks);
                    const Vmm vmean_ch_odd = Vmm(ch_idx + 1 + num_ch_blks);
                    if (is_ch_blks_tail)
                        uni_vmovups_spat_data(vsrc_even,
                                vmmword[reg_src + reg_soff_nspc + offt]);
                    else
                        uni_vmovups_spat_data(vsrc_even, vsrc_odd,
                                vmmword[reg_src + reg_soff_nspc + offt]);
                    uni_vsubps(vsrc_even, vsrc_even, vmean_ch_even);
                    uni_vfmadd231ps(Vmm(ch_idx), vsrc_even, vsrc_even);
                    if (!is_ch_blks_tail) {
                        uni_vsubps(vsrc_odd, vsrc_odd, vmean_ch_odd);
                        uni_vfmadd231ps(Vmm(ch_idx + 1), vsrc_odd, vsrc_odd);
                    }
                }
                add(reg_soff_nspc, spat_step);
            }
        };

        auto mean_compute = [this](int num_ch_blks, int num_spat_pts) {
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    const int offt = ch_idx * vlen_spat_data_;
                    const Vmm vsrc = vtmp;
                    uni_vmovups_spat_data(
                            vsrc, vmmword[reg_src + reg_soff_nspc + offt]);
                    uni_vaddps(Vmm(ch_idx), Vmm(ch_idx), vsrc);
                }
                add(reg_soff_nspc, spat_step);
            }
        };

        auto variance_compute = [this](int num_ch_blks, int num_spat_pts) {
            for (int spat_pt = 0; spat_pt < num_spat_pts; ++spat_pt) {
                for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                    const int offt = ch_idx * vlen_spat_data_;
                    const Vmm vsrc = vtmp;
                    const Vmm vmean_ch = Vmm(ch_idx + num_ch_blks);
                    uni_vmovups_spat_data(
                            vsrc, vmmword[reg_src + reg_soff_nspc + offt]);
                    uni_vsubps(vsrc, vsrc, vmean_ch);
                    uni_vfmadd231ps(Vmm(ch_idx), vsrc, vsrc);
                }
                add(reg_soff_nspc, spat_step);
            }
        };

        for (int idx = 0; idx < num_ch_blks; ++idx) {
            const int coff = idx * vlen;
            uni_vmovups(Vmm(idx), vmmword[reg_rbuf1 + reg_coff + coff]);
            if (!compute_mean) {
                // pre-load mean to avoid extra data movement during variance
                const Vmm vmean_ch = Vmm(idx + num_ch_blks);
                uni_vmovups_maybe_tail(vmean_ch, mean_ptr(coff));
            }
        }

        xor_(reg_soff_nspc, reg_soff_nspc);

        if (jbp_->is_spatial_thr_) {
            mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
            add(reg_soff_nspc, ptr[rsp + stack_off_s_s]);
            // TODO: need a better heuristic for num_spat_pts
            num_spat_pts = 1;
        } else {
            mov(reg_ctr, spat_size);
            num_spat_pts = nstl::min((size_t)num_spat_pts, spat_size);
            // TODO: unroll by spatial
            if (spat_size % num_spat_pts != 0) num_spat_pts = 1;
        }

        Label spatial;
        L(spatial);
        {
            if (is_avx2_ne_xf16_)
                compute_mean
                        ? mean_compute_avx2_ne_xf16(num_ch_blks, num_spat_pts)
                        : variance_compute_avx2_ne_xf16(
                                num_ch_blks, num_spat_pts);
            else
                compute_mean ? mean_compute(num_ch_blks, num_spat_pts)
                             : variance_compute(num_ch_blks, num_spat_pts);
            sub(reg_ctr, num_spat_pts);
            jnz(spatial, T_NEAR);
        }

        for (int idx = 0; idx < num_ch_blks; ++idx) {
            const int coff = idx * vlen;
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff + coff], Vmm(idx));
        }
    }

    void forward_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [this, num_ch_blks](bool stream_store_allowed) {
            // Overwritten during mean and variance computation
            uni_vpxor(vzero, vzero, vzero);

            xor_(reg_soff_nspc, reg_soff_nspc);

            if (jbp_->is_spatial_thr_) {
                mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
                add(reg_soff_nspc, ptr[rsp + stack_off_s_s]);
            } else {
                mov(reg_ctr, spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            // pre-compute scale for each channel to avoid costly div and sqrt
            // merge variances in interleaved to plain layout if needed
            for (int idx = 0; idx < num_ch_blks; idx += 2) {
                const int coff_base = idx * vlen;
                const bool is_ch_blks_tail = num_ch_blks - idx < 2;
                const Vmm vvar_even = Vmm(idx);
                const Vmm vvar_odd = Vmm(idx + 1);
                if (!is_ch_blks_tail) {
                    uni_vmovups_maybe_tail(vvar_even, var_ptr(coff_base));
                    uni_vmovups_maybe_tail(vvar_odd, var_ptr(coff_base + vlen));
                    if (is_avx2_ne_xf16_ && !pd_->stats_is_src())
                        merge_interleaved_to_plain(vvar_even, vvar_odd, vtmp);
                } else
                    uni_vmovups_maybe_tail(vvar_even, var_ptr(coff_base));

                for (int i_odd = 0; i_odd < 2 && idx + i_odd < num_ch_blks;
                        ++i_odd) {
                    const int coff = coff_base + i_odd * vlen;
                    const Vmm vscale = Vmm(idx + i_odd + num_ch_blks);
                    const Vmm vvar = i_odd ? vvar_odd : vvar_even;
                    uni_vmovups(vsqrtvar, vvar);
                    uni_vaddps(vsqrtvar, vsqrtvar, veps);
                    uni_vsqrtps(vsqrtvar, vsqrtvar);

                    if (pd_->use_scale()) {
                        uni_vmovups_maybe_tail(vgamma, gamma_ptr(coff));
                        uni_vdivps(vscale, vgamma, vsqrtvar, vtmp);
                    } else {
                        uni_vdivps(vscale, vone, vsqrtvar, vtmp);
                    }
                }
            }

            Label spatial;
            L(spatial);
            {
                if (is_avx2_ne_xf16_) {
                    for (int idx = 0; idx < num_ch_blks; idx += 2) {
                        const int offt = idx * vlen_spat_data_;
                        const int coff = idx * vlen;
                        const bool is_ch_blks_tail = num_ch_blks - idx < 2;
                        Vmm vdata_even = Vmm(idx);
                        Vmm vdata_odd = Vmm(idx + 1);
                        if (is_ch_blks_tail) {
                            uni_vmovups_spat_data(vdata_even,
                                    vmmword[reg_src + reg_soff_nspc + offt]);
                            if (!pd_->stats_is_src())
                                uni_vsubps(
                                        vdata_even, vdata_even, mean_ptr(coff));
                        } else {
                            uni_vmovups_spat_data(vdata_even, vdata_odd,
                                    vmmword[reg_src + reg_soff_nspc + offt]);
                            // apply mean in interleave to data in interleave
                            // before merge them to plain layout when needed
                            if (!pd_->stats_is_src()) {
                                uni_vsubps(
                                        vdata_even, vdata_even, mean_ptr(coff));
                                uni_vsubps(vdata_odd, vdata_odd,
                                        mean_ptr(coff + vlen));
                            }
                            merge_interleaved_to_plain(
                                    vdata_even, vdata_odd, vtmp);
                        }
                    }
                }

                for (int idx = 0; idx < num_ch_blks; ++idx) {
                    const int coff = idx * vlen;
                    const int offt = idx * vlen_spat_data_;
                    const Vmm vdata = Vmm(idx);
                    const Vmm vscale = Vmm(idx + num_ch_blks);
                    uni_vmovups_maybe_tail(vmean, mean_ptr(coff));

                    if (pd_->use_shift()) {
                        uni_vmovups_maybe_tail(vbeta, beta_ptr(coff));
                    }

                    if (!is_avx2_ne_xf16_)
                        uni_vmovups_spat_data(
                                vdata, vmmword[reg_src + reg_soff_nspc + offt]);
                    if (IMPLICATION(is_avx2_ne_xf16_, pd_->stats_is_src()))
                        uni_vsubps(vdata, vdata, vmean);

                    if (pd_->use_shift()) {
                        // --flags=S,CH,H
                        uni_vfmadd213ps(vdata, vscale, vbeta);
                    } else {
                        // --flags=,C
                        uni_vmulps(vdata, vdata, vscale);
                    }

                    if (with_relu_inf_only) { // --attr=post_ops='relu'
                        if (pd_->alpha() != 0.f)
                            fwd_process_relu_alpha(vdata);
                        else
                            uni_vmaxps(vdata, vdata, vzero);
                    } else if (with_relu) { // --flags=R
                        if (isa == avx512_core)
                            fwd_process_relu_avx512_common(vdata, offt);
                        else if (isa == avx2)
                            fwd_process_relu_avx2(vdata, offt);
                        else
                            assert(false);
                    }
                    uni_vmovups_spat_data(
                            vmmword[reg_dst + reg_soff_nspc + offt], vdata,
                            stream_store_allowed);
                }
                add(reg_soff_nspc, spat_step);
                sub(reg_ctr, num_spat_pts);
                jnz(spatial, T_NEAR);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;
            test(reg_dst, vlen_spat_data_ - 1);
            jnz(normal_store, T_NEAR);
            compute(true);
            jmp(end_store, T_NEAR);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // disabled for bf16 when data fits in cache
        }
    }

    void compute_mean_variance_nspc(bool compute_mean = true) {
        xor_(reg_coff, reg_coff);
        mov(reg_coff_max_fwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll = isa == avx512_core ? 4 : 2;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll, sp_idx = 1; ch_idx > 0;
                --ch_idx, ++sp_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                jl(ch_unroll_label[ch_idx - 1], T_NEAR);

                const int spat_blk_size = (1 << sp_idx);
                mean_variance_nspc(ch_blk_size, spat_blk_size, compute_mean);

                add(reg_src, vlen_spat_data_ * ch_blk_size);
                add(reg_coff, vlen * ch_blk_size);

                sub(reg_coff_max, vlen * ch_blk_size);
                jmp(ch_unroll_label[ch_idx], T_NEAR);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_fwd_copy);

        if (is_xf16()) shr(reg_coff_max, 1);
        sub(reg_src, reg_coff_max);
        if (is_xf16()) shl(reg_coff_max, 1);
    }

    void var_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_vmovups_maybe_tail(vmean, mean_ptr());
            uni_vmovups(Vmm(0), vmmword[reg_rbuf1 + reg_coff]);
            spat_loop(
                    spat_size, unroll_blocks, unroll_regs,
                    [this](size_t base_reg) {
                        Vmm v = Vmm(base_reg * 3);
                        if (base_reg > 0) uni_vpxor(v, v, v);
                    },
                    [this](size_t base_reg, size_t i) {
                        Vmm v = Vmm(3 * base_reg);
                        Vmm vtmp0 = Vmm(3 * base_reg + 1);
                        Vmm vtmp1 = Vmm(3 * base_reg + 2);
                        size_t offt = i * vlen_spat_data_;
                        uni_vmovups_spat_data(
                                vtmp0, vmmword[reg_src + reg_soff + offt]);
                        if (isa == sse41) {
                            movups(vtmp1, vmean);
                            subps(vtmp1, vtmp0);
                        } else {
                            vsubps(vtmp1, vmean, vtmp0);
                        }
                        uni_vfmadd231ps(v, vtmp1, vtmp1);
                    },
                    [this](size_t base_reg) {
                        Vmm b = Vmm(0);
                        Vmm v = Vmm(base_reg * 3);
                        if (base_reg) uni_vaddps(b, b, v);
                    });
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(ch_label);
        }
    }

    void compute_mean_variance() {
        uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
        xor_(reg_coff, reg_coff);
        Label zero_rbuf;
        L(zero_rbuf);
        {
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            add(reg_coff, isa == sse41 ? vlen / 2 : vlen);
            cmp(reg_coff, reg_coff_max);
            jne(zero_rbuf);
        }

        mov(reg_src, ptr[rsp + stack_off_src]);

        xor_(reg_soff, reg_soff);
        Label mean_spatial;
        L(mean_spatial);
        {
            xor_(reg_coff, reg_coff);

            if (isa == sse41) mov(reg_tmp_off, reg_soff);

            jbp_->is_nspc_ ? compute_mean_variance_nspc() : mean_channels();

            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);

                mean_channels();

                sub(reg_src, vlen / 2);
            }

            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add(reg_src, mb_offt);
                add(reg_soff, mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }

            cmp(reg_soff, reg_soff_max);
            jl(mean_spatial);
        }

        if (jbp_->is_nspc_) mov(reg_src, ptr[rsp + stack_off_src]); // comeback

        Label no_mean_reduction;
        barrier();
        {
            mov(reg_tmp, ptr[rsp + stack_off_N_ithr]);
            cmp(reg_tmp, 0);
            jne(no_mean_reduction);
            mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
            xor_(reg_coff, reg_coff);
            Label mean_reduction_channels;
            L(mean_reduction_channels);
            {
                mov(reg_roff, reg_coff);
                uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
                uni_vpxor(Vmm(1), Vmm(1), Vmm(1));
                mov(reg_ctr, reg_nnthr);
                Label mean_reduction_thrs;
                L(mean_reduction_thrs);
                {
                    uni_vaddps(Vmm(1), Vmm(1), vmmword[reg_rbuf1 + reg_roff]);
                    uni_vmovups(vmmword[reg_rbuf1 + reg_roff], Vmm(0));
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(mean_reduction_thrs);
                }
                uni_vdivps(Vmm(1), Vmm(1), vchan_size);
                uni_vmovups_maybe_tail(mean_ptr(), Vmm(1));

                add(reg_coff, isa == sse41 ? vlen / 2 : vlen);

                cmp(reg_coff, reg_coff_max);
                jl(mean_reduction_channels);
            }
        }
        L(no_mean_reduction);
        barrier();

        xor_(reg_soff, reg_soff);
        Label var_spatial;
        L(var_spatial);
        {
            xor_(reg_coff, reg_coff);

            if (isa == sse41) mov(reg_tmp_off, reg_soff);

            jbp_->is_nspc_ ? compute_mean_variance_nspc(false) : var_channels();

            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);

                var_channels();

                sub(reg_src, vlen / 2);
            }

            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add(reg_src, mb_offt);
                add(reg_soff, mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }

            cmp(reg_soff, reg_soff_max);
            jl(var_spatial);
        }

        if (jbp_->is_nspc_) mov(reg_src, ptr[rsp + stack_off_src]); // comeback

        Label no_var_reduction;
        barrier();
        {
            mov(reg_tmp, ptr[rsp + stack_off_N_ithr]);
            cmp(reg_tmp, 0);
            jne(no_var_reduction);

            mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
            xor_(reg_coff, reg_coff);
            Label var_reduction_channels;
            L(var_reduction_channels);
            {
                mov(reg_roff, reg_coff);
                uni_vpxor(Vmm(1), Vmm(1), Vmm(1));
                mov(reg_ctr, reg_nnthr);
                Label var_reduction_thrs;
                L(var_reduction_thrs);
                { // TODO: unroll (?)
                    uni_vaddps(Vmm(1), Vmm(1), vmmword[reg_rbuf1 + reg_roff]);
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(var_reduction_thrs);
                }
                uni_vdivps(Vmm(1), Vmm(1), vchan_size);
                uni_vmovups_maybe_tail(var_ptr(), Vmm(1));
                add(reg_coff, isa == sse41 ? vlen / 2 : vlen);

                cmp(reg_coff, reg_coff_max);
                jne(var_reduction_channels);
            }
        }
        L(no_var_reduction);
        barrier();
    }

    void forward_channels() {
        Label ch_label;
        L(ch_label);
        {
            uni_vmovups_maybe_tail(vmean, mean_ptr());
            uni_vmovups_maybe_tail(vsqrtvar, var_ptr());
            uni_vaddps(vsqrtvar, vsqrtvar, veps);
            uni_vsqrtps(vsqrtvar, vsqrtvar);

            if (pd_->use_scale()) {
                uni_vmovups_maybe_tail(vgamma, gamma_ptr());
            }
            if (pd_->use_shift()) { uni_vmovups_maybe_tail(vbeta, beta_ptr()); }

            Vmm vscale = (pd_->use_scale()) ? vgamma : vone;
            Vmm vdiv = (pd_->use_scale()) ? vgamma : vsqrtvar;

            if (isa == sse41) {
                movups(vtmp, vscale);
                divps(vtmp, vsqrtvar);
                movups(vdiv, vtmp);
            } else {
                vdivps(vdiv, vscale, vsqrtvar);
            }

            const auto spat_loop_init_fin
                    = [](size_t base_reg) { UNUSED(base_reg); };

            const auto spat_loop_body = [this](size_t base_reg, size_t i,
                                                bool stream_store_allowed) {
                const Vmm v = Vmm(base_reg);
                const size_t offt = i * vlen_spat_data_;
                uni_vmovups_spat_data(v, vmmword[reg_src + reg_soff + offt]);
                uni_vsubps(v, v, vmean);
                if ((pd_->use_scale() && pd_->use_shift())) {
                    // --flags=CH
                    uni_vfmadd213ps(v, vgamma, vbeta);
                } else if (pd_->use_scale()) {
                    // --flags=C
                    uni_vmulps(v, v, vgamma);
                } else if (pd_->use_shift()) {
                    // --flags=H
                    uni_vfmadd213ps(v, vsqrtvar, vbeta);
                } else {
                    uni_vmulps(v, v, vsqrtvar);
                }
                if (with_relu_inf_only) { // --attr=post_ops='relu'
                    if (pd_->alpha() != 0.f) {
                        fwd_process_relu_alpha(v);
                    } else
                        uni_vmaxps(v, v, vzero);
                } else if (with_relu) { // --flags=R
                    if (isa == avx512_core)
                        fwd_process_relu_avx512_common(v, offt);
                    else
                        fwd_process_relu_avx2(v, offt);
                }
                if (stream_store_allowed) {
                    uni_vmovntps(vmmword[reg_dst + reg_soff + offt], v);
                } else {
                    uni_vmovups_spat_data(
                            vmmword[reg_dst + reg_soff + offt], v);
                }
            };

            const auto compute = [this, spat_loop_init_fin, spat_loop_body](
                                         bool stream_store_allowed) {
                using namespace std::placeholders;
                spat_loop(spat_size, unroll_blocks, unroll_regs,
                        spat_loop_init_fin,
                        std::bind(spat_loop_body, _1, _2, stream_store_allowed),
                        spat_loop_init_fin);
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                test(reg_dst, vlen - 1);
                jnz(normal_store, T_NEAR);
                compute(true);
                jmp(end_store, T_NEAR);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(ch_label);
        }
    }

    void forward_channels_nspc() {
        xor_(reg_coff, reg_coff);
        mov(reg_coff_max_fwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll
                = isa == avx512_core ? 4 - use_bf16_emulation() : 2;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                jl(ch_unroll_label[ch_idx - 1], T_NEAR);

                forward_channels_nspc_compute(ch_blk_size);

                add(reg_src, vlen_spat_data_ * ch_blk_size);
                add(reg_dst, vlen_spat_data_ * ch_blk_size);

                // advance mean_ptr() and var_ptr()
                add(reg_coff, vlen * ch_blk_size);

                add(reg_ws, (vlen / 32) * ch_blk_size);

                sub(reg_coff_max, vlen * ch_blk_size);
                jmp(ch_unroll_label[ch_idx], T_NEAR);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_fwd_copy);

        if (is_xf16()) shr(reg_coff_max, 1);
        sub(reg_src, reg_coff_max);
        sub(reg_dst, reg_coff_max);
        if (is_xf16()) shl(reg_coff_max, 1);

        shr(reg_coff_max, 5);
        sub(reg_ws, reg_coff_max);
        shl(reg_coff_max, 5);
    }

    void forward() {
        mov(reg_src, ptr[rsp + stack_off_src]);
        mov(reg_dst, ptr[rsp + stack_off_dst]);
        mov(reg_ws, ptr[rsp + stack_off_ws]);
        mov(reg_shift, ptr[rsp + stack_off_shift]);

        xor_(reg_soff, reg_soff);
        Label dst_spatial;
        L(dst_spatial);
        {
            xor_(reg_coff, reg_coff);
            if (isa == sse41) mov(reg_tmp_off, reg_soff);

            jbp_->is_nspc_ ? forward_channels_nspc() : forward_channels();

            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_src, vlen / 2);
                add(reg_dst, vlen / 2);
                mov(reg_coff, vlen / 2);

                forward_channels();

                sub(reg_src, vlen / 2);
                sub(reg_dst, vlen / 2);
            }

            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add(reg_src, mb_offt);
                add(reg_dst, mb_offt);
                add(reg_soff, mb_offt);
                add(reg_ws, ws_mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }

            cmp(reg_soff, reg_soff_max);
            jl(dst_spatial);
        }

        if (jbp_->is_nspc_) {
            // comeback
            mov(reg_src, ptr[rsp + stack_off_src]);
            mov(reg_dst, ptr[rsp + stack_off_dst]);
            mov(reg_ws, ptr[rsp + stack_off_ws]);
        }
    }

    void backward_sh_channels() {
        Label sh_channels;
        L(sh_channels);
        {
            uni_vmovups_maybe_tail(vmean, mean_ptr());
            uni_vmovups(Vmm(0), vmmword[reg_rbuf1 + reg_coff]);
            uni_vmovups(Vmm(1), vmmword[reg_rbuf2 + reg_coff]);
            spat_loop(
                    spat_size, 1, 1,
                    [this](size_t base_reg) {
                        if (base_reg > 0) {
                            for (int i = 0; i < 2; i++) {
                                Vmm v(base_reg * 5 + i);
                                uni_vpxor(v, v, v);
                            }
                        }
                    },
                    [this](size_t base_reg, size_t i) {
                        // TODO: use single set of tmp regs and let ROB handle the rest
                        Vmm o0 = Vmm(base_reg * 5 + 0);
                        Vmm o1 = Vmm(base_reg * 5 + 1);
                        Vmm t1 = Vmm(base_reg * 5 + 2);
                        Vmm t2 = Vmm(base_reg * 5 + 3);
                        Vmm t3 = Vmm(base_reg * 5 + 4);
                        size_t offt = i * vlen_spat_data_;
                        uni_vmovups_spat_data(
                                t1, vmmword[reg_src + reg_soff + offt]);
                        uni_vmovups_spat_data(
                                t2, vmmword[reg_diff_dst + reg_soff + offt]);
                        if (with_relu) {
                            if (isa == avx512_core)
                                bwd_process_relu_avx512_common(t2, offt);
                            else if (isa == avx2)
                                bwd_process_relu_avx2(t2, offt);
                            else
                                assert(false);
                        }
                        uni_vsubps(t3, vmean, t1, t3);
                        if (isa == sse41) {
                            mulps(t3, t2);
                            subps(o0, t3);
                        } else {
                            vfnmadd231ps(o0, t3, t2);
                        }
                        uni_vaddps(o1, o1, t2);
                    },
                    [this](size_t base_reg) {
                        Vmm b0 = Vmm(0);
                        Vmm b1 = Vmm(1);
                        if (base_reg) {
                            uni_vaddps(b0, b0, Vmm(base_reg * 5 + 0));
                            uni_vaddps(b1, b1, Vmm(base_reg * 5 + 1));
                        }
                    });
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            uni_vmovups(vmmword[reg_rbuf2 + reg_coff], Vmm(1));
            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(sh_channels);
        }
    }

    void backward_sh_channels_nspc_compute(const int num_ch_blks) {
        for (int idx = 0; idx < num_ch_blks; ++idx) {
            const int offt = idx * vlen;
            const Vmm vdiff_gamma_ch = Vmm(idx);
            const Vmm vdiff_beta_ch = Vmm(idx + num_ch_blks);
            uni_vmovups(vdiff_gamma_ch, vmmword[reg_rbuf1 + reg_coff + offt]);
            uni_vmovups(vdiff_beta_ch, vmmword[reg_rbuf2 + reg_coff + offt]);
        }

        xor_(reg_soff_nspc, reg_soff_nspc);

        if (jbp_->is_spatial_thr_) {
            mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
            add(reg_soff_nspc, ptr[rsp + stack_off_s_s]);
        } else {
            mov(reg_ctr, spat_size);
        }

        // TODO: spatial blocking
        const int num_spat_pts = 1;

        Label spatial;
        L(spatial);
        {
            for (int ch_idx = 0; ch_idx < num_ch_blks; ++ch_idx) {
                const int coff = ch_idx * vlen;
                const int offt = ch_idx * vlen_spat_data_;
                const Vmm vdiff_gamma_ch = Vmm(ch_idx);
                const Vmm vdiff_beta_ch = Vmm(ch_idx + num_ch_blks);
                // vdiff_beta and vdiff_gamma are free registers for nspc
                const Vmm vsrc = vdiff_gamma;
                const Vmm vdiff_dst = vdiff_beta;
                uni_vmovups_maybe_tail(vmean, mean_ptr(coff));

                uni_vmovups_spat_data(
                        vsrc, vmmword[reg_src + reg_soff_nspc + offt]);
                uni_vmovups_spat_data(vdiff_dst,
                        vmmword[reg_diff_dst + reg_soff_nspc + offt]);

                if (with_relu) {
                    if (isa == avx512_core)
                        bwd_process_relu_avx512_common(vdiff_dst, offt);
                    else
                        assert(false);
                }

                uni_vsubps(vsrc, vsrc, vmean);
                uni_vfmadd231ps(vdiff_gamma_ch, vsrc, vdiff_dst);
                uni_vaddps(vdiff_beta_ch, vdiff_beta_ch, vdiff_dst);
            }
            add(reg_soff_nspc, spat_step);
            sub(reg_ctr, num_spat_pts);
            jnz(spatial, T_NEAR);
        }

        for (int idx = 0; idx < num_ch_blks; ++idx) {
            const Vmm vdiff_gamma_ch = Vmm(idx);
            const Vmm vdiff_beta_ch = Vmm(idx + num_ch_blks);
            const int offt = idx * vlen;
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff + offt], vdiff_gamma_ch);
            uni_vmovups(vmmword[reg_rbuf2 + reg_coff + offt], vdiff_beta_ch);
        }
    }

    void backward_sh_channels_nspc() {
        xor_(reg_coff, reg_coff);
        mov(reg_coff_max_bwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll = 4;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 8, 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                jl(ch_unroll_label[ch_idx - 1], T_NEAR);

                backward_sh_channels_nspc_compute(ch_blk_size);

                add(reg_src, vlen_spat_data_ * ch_blk_size);
                add(reg_diff_dst, vlen_spat_data_ * ch_blk_size);

                // advance mean_ptr() and var_ptr()
                add(reg_coff, vlen * ch_blk_size);

                add(reg_ws, 2 * ch_blk_size);

                sub(reg_coff_max, vlen * ch_blk_size);
                jmp(ch_unroll_label[ch_idx], T_NEAR);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_bwd_copy);
        mov(reg_diff_scale, ptr[rsp + stack_off_diff_scale]);

        if (is_xf16()) shr(reg_coff_max, 1);
        sub(reg_src, reg_coff_max);
        sub(reg_diff_dst, reg_coff_max);
        if (is_xf16()) shl(reg_coff_max, 1);

        if (with_relu) {
            shr(reg_coff_max, 5);
            sub(reg_ws, reg_coff_max);
            shl(reg_coff_max, 5);
        }
    }

    void backward_diff_channels() {
        Label diff_channels;
        L(diff_channels);
        {
            uni_vmovups_maybe_tail(vmean, mean_ptr());
            uni_vmovups_maybe_tail(vsqrtvar, var_ptr());
            uni_vaddps(vsqrtvar, vsqrtvar, veps);
            uni_vsqrtps(vsqrtvar, vsqrtvar);
            uni_vdivps(vsqrtvar, vone, vsqrtvar, vtmp);
            if (pd_->use_scale()) uni_vmovups_maybe_tail(vgamma, gamma_ptr());
            uni_vmovups_maybe_tail(vdiff_gamma, diff_gamma_ptr());
            uni_vmovups_maybe_tail(vdiff_beta, diff_beta_ptr());
            uni_vmulps(vdiff_gamma, vdiff_gamma, vsqrtvar);
            uni_vdivps(vdiff_beta, vdiff_beta, vchan_size);
            uni_vdivps(vdiff_gamma, vdiff_gamma, vchan_size);

            const auto spat_loop_init_fin
                    = [](size_t base_reg) { UNUSED(base_reg); };
            const auto spat_loop_body = [this](size_t base_reg, size_t i,
                                                bool stream_store_allowed) {
                const Vmm v(base_reg * 2 + 0);
                const Vmm t(base_reg * 2 + 1);
                const Vmm t1(base_reg * 2 + 2);
                const size_t offt = i * vlen_spat_data_;
                uni_vmovups_spat_data(
                        v, vmmword[reg_diff_dst + reg_soff + offt]);
                if (with_relu) {
                    if (isa == avx512_core)
                        bwd_process_relu_avx512_common(v, offt);
                    else if (isa == avx2)
                        bwd_process_relu_avx2(v, offt);
                    else
                        assert(false);
                }
                if (!pd_->use_global_stats()) {
                    uni_vsubps(v, v, vdiff_beta);
                    uni_vmovups_spat_data(
                            t, vmmword[reg_src + reg_soff + offt]);
                    uni_vsubps(t, vmean, t, t1);
                    uni_vmulps(t, t, vdiff_gamma);
                    uni_vaddps(v, v, t);
                }
                uni_vmulps(v, v, vsqrtvar);
                if (pd_->use_scale()) { uni_vmulps(v, v, vgamma); }
                if (stream_store_allowed) {
                    uni_vmovntps(vmmword[reg_diff_src + reg_soff + offt], v);
                } else {
                    uni_vmovups_spat_data(
                            vmmword[reg_diff_src + reg_soff + offt], v);
                }
            };

            const auto compute = [this, spat_loop_init_fin, spat_loop_body](
                                         bool stream_store_allowed) {
                using namespace std::placeholders;
                spat_loop(spat_size, unroll_blocks, unroll_regs,
                        spat_loop_init_fin,
                        std::bind(spat_loop_body, _1, _2, stream_store_allowed),
                        spat_loop_init_fin);
            };

            if (stream_store_supported()) {
                Label normal_store, end_store;
                test(reg_diff_src, vlen - 1);
                jnz(normal_store, T_NEAR);
                compute(true);
                jmp(end_store, T_NEAR);
                L(normal_store);
                { compute(false); }
                L(end_store);
            } else {
                compute(false); // no NT store for BF16
            }

            add(reg_coff, vlen);
            cmp(reg_coff, reg_coff_max);
            jl(diff_channels);
        }
    }

    void backward_diff_channels_nspc_compute(const int num_ch_blks) {
        auto compute = [this, num_ch_blks](bool stream_store_allowed) {
            xor_(reg_soff_nspc, reg_soff_nspc);
            if (jbp_->is_spatial_thr_) {
                mov(reg_ctr, ptr[rsp + stack_off_spat_size_loc]);
                add(reg_soff_nspc, ptr[rsp + stack_off_s_s]);
            } else {
                mov(reg_ctr, spat_size);
            }

            // TODO: spatial blocking
            const int num_spat_pts = 1;

            // pre-compute scale for each channel to avoid costly div and sqrt
            if (!pd_->use_global_stats()) {
                mov(ptr[rsp + stack_off_ws_off_copy], reg_ws);
                mov(reg_ws, ptr[rsp + stack_off_diff_scale]);
            }
            for (int idx = 0; idx < num_ch_blks; ++idx) {
                const int coff = idx * vlen;
                const Vmm vsqrtvar_ch = Vmm(idx);
                uni_vmovups_maybe_tail(vsqrtvar_ch, var_ptr(coff));
                uni_vaddps(vsqrtvar_ch, vsqrtvar_ch, veps);
                uni_vsqrtps(vsqrtvar_ch, vsqrtvar_ch);
                uni_vdivps(vsqrtvar_ch, vone, vsqrtvar_ch, vtmp);
                if (!pd_->use_global_stats()) {
                    const Vmm vdiff_beta_ch = Vmm(idx + num_ch_blks);
                    const Vmm vdiff_gamma_ch = Vmm(idx + 2 * num_ch_blks);
                    uni_vmovups_maybe_tail(vdiff_beta_ch,
                            vmmword[reg_diff_shift + reg_coff + coff]);
                    uni_vmovups_maybe_tail(
                            vdiff_gamma_ch, vmmword[reg_ws + reg_coff + coff]);
                    uni_vdivps(vdiff_beta_ch, vdiff_beta_ch, vchan_size);
                    uni_vmulps(vdiff_gamma_ch, vdiff_gamma_ch, vsqrtvar_ch);
                    uni_vdivps(vdiff_gamma_ch, vdiff_gamma_ch, vchan_size);
                }
            }
            if (!pd_->use_global_stats()) {
                mov(reg_ws, ptr[rsp + stack_off_ws_off_copy]);
            }

            Label spatial;
            L(spatial);
            {
                for (int idx = 0; idx < num_ch_blks; ++idx) {
                    const int coff = idx * vlen;
                    const int offt = idx * vlen_spat_data_;
                    // vdiff_beta and vdiff_gamma are free registers for nspc
                    const Vmm vdiff_data = vdiff_beta;
                    const Vmm vdata = vdiff_gamma;
                    const Vmm vsqrtvar_ch = Vmm(idx);
                    uni_vmovups_maybe_tail(vmean, mean_ptr(coff));

                    if (pd_->use_scale())
                        uni_vmovups_maybe_tail(vgamma, gamma_ptr(coff));

                    uni_vmovups_spat_data(vdiff_data,
                            vmmword[reg_diff_dst + reg_soff_nspc + offt]);

                    if (with_relu) {
                        if (isa == avx512_core)
                            bwd_process_relu_avx512_common(vdiff_data, offt);
                        else
                            assert(false);
                    }

                    if (!pd_->use_global_stats()) {
                        const Vmm vdiff_beta_ch = Vmm(idx + num_ch_blks);
                        const Vmm vdiff_gamma_ch = Vmm(idx + 2 * num_ch_blks);
                        uni_vsubps(vdiff_data, vdiff_data, vdiff_beta_ch);
                        uni_vmovups_spat_data(
                                vdata, vmmword[reg_src + reg_soff_nspc + offt]);
                        uni_vsubps(vdata, vmean, vdata, vtmp);
                        uni_vmulps(vdata, vdata, vdiff_gamma_ch);
                        uni_vaddps(vdiff_data, vdiff_data, vdata);
                    }

                    uni_vmulps(vdiff_data, vdiff_data, vsqrtvar_ch);

                    if (pd_->use_scale()) {
                        uni_vmulps(vdiff_data, vdiff_data, vgamma);
                    }

                    uni_vmovups_spat_data(
                            vmmword[reg_diff_src + reg_soff_nspc + offt],
                            vdiff_data, stream_store_allowed);
                }
                add(reg_soff_nspc, spat_step);
                sub(reg_ctr, num_spat_pts);
                jnz(spatial, T_NEAR);
            }
        };

        if (stream_store_supported()) {
            Label normal_store, end_store;
            test(reg_diff_src, vlen - 1);
            jnz(normal_store, T_NEAR);
            compute(true);
            jmp(end_store, T_NEAR);
            L(normal_store);
            { compute(false); }
            L(end_store);
        } else {
            compute(false); // disabled for bf16 when data fits in cache
        }
    }

    void backward_diff_channels_nspc() {
        xor_(reg_coff, reg_coff);
        mov(reg_coff_max_bwd_copy, reg_coff_max);

        Label ch_unroll_label[5];
        const int max_ch_unroll = 3;

        // TODO: Spatial and channel unrolling decisions should be made during
        // initialization depending on the problem size
        for (int ch_idx = max_ch_unroll; ch_idx > 0; --ch_idx) {
            L(ch_unroll_label[ch_idx]);
            {
                const int ch_blk_size = (1 << (ch_idx - 1)); // 4, 2, 1
                cmp(reg_coff_max, vlen * ch_blk_size);
                jl(ch_unroll_label[ch_idx - 1], T_NEAR);

                backward_diff_channels_nspc_compute(ch_blk_size);

                add(reg_diff_dst, vlen_spat_data_ * ch_blk_size);
                if (!pd_->use_global_stats())
                    add(reg_src, vlen_spat_data_ * ch_blk_size);
                add(reg_diff_src, vlen_spat_data_ * ch_blk_size);

                // advance mean_ptr() and var_ptr()
                add(reg_coff, vlen * ch_blk_size);

                add(reg_ws, 2 * ch_blk_size);

                sub(reg_coff_max, vlen * ch_blk_size);
                jmp(ch_unroll_label[ch_idx], T_NEAR);
            }
        }
        L(ch_unroll_label[0]);

        // comeback
        mov(reg_coff_max, reg_coff_max_bwd_copy);
        mov(reg_diff_scale, ptr[rsp + stack_off_diff_scale]);

        if (is_xf16()) shr(reg_coff_max, 1);
        sub(reg_diff_dst, reg_coff_max);
        if (!pd_->use_global_stats()) sub(reg_src, reg_coff_max);
        sub(reg_diff_src, reg_coff_max);
        if (is_xf16()) shl(reg_coff_max, 1);

        shr(reg_coff_max, 5);
        sub(reg_ws, reg_coff_max);
        shl(reg_coff_max, 5);
    }

    void backward() {
        uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
        xor_(reg_coff, reg_coff);
        Label zero_rbuf, sh_spatial;

        L(zero_rbuf);
        {
            uni_vmovups(vmmword[reg_rbuf1 + reg_coff], Vmm(0));
            uni_vmovups(vmmword[reg_rbuf2 + reg_coff], Vmm(0));
            add(reg_coff, isa == sse41 ? vlen / 2 : vlen);
            cmp(reg_coff, reg_coff_max);
            jne(zero_rbuf);
        }

        mov(reg_src, ptr[rsp + stack_off_src]);
        mov(reg_diff_dst, ptr[rsp + stack_off_diff_dst]);
        if (with_relu) {
            assert(isa == avx2 || isa == avx512_core);
            mov(reg_ws, ptr[rsp + stack_off_ws]);
        }

        xor_(reg_soff, reg_soff);
        L(sh_spatial);
        {
            xor_(reg_coff, reg_coff);
            if (isa == sse41) { mov(reg_tmp_off, reg_soff); }
            jbp_->is_nspc_ ? backward_sh_channels_nspc()
                           : backward_sh_channels();
            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_diff_dst, vlen / 2);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);
                backward_sh_channels();
                sub(reg_diff_dst, vlen / 2);
                sub(reg_src, vlen / 2);
            }
            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                add(reg_src, mb_offt);
                add(reg_diff_dst, mb_offt);
                add(reg_soff, mb_offt);
                add(reg_ws, ws_mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }
            cmp(reg_soff, reg_soff_max);
            jl(sh_spatial);
        }

        if (jbp_->is_nspc_) {
            // comeback
            mov(reg_src, ptr[rsp + stack_off_src]);
            mov(reg_diff_dst, ptr[rsp + stack_off_diff_dst]);
        }

        mov(reg_diff_scale, ptr[rsp + stack_off_diff_scale]);
        mov(reg_diff_shift, ptr[rsp + stack_off_diff_shift]);

        Label no_sh_reduction;
        barrier();
        {
            mov(reg_tmp, ptr[rsp + stack_off_N_ithr]);
            cmp(reg_tmp, 0);
            Label sh_reduction_channels;
            jne(no_sh_reduction, T_NEAR);

            mov(reg_nnthr, ptr[rsp + stack_off_N_nthr]);
            xor_(reg_coff, reg_coff);
            L(sh_reduction_channels);
            {
                mov(reg_roff, reg_coff);
                uni_vpxor(Vmm(0), Vmm(0), Vmm(0));
                uni_vpxor(Vmm(1), Vmm(1), Vmm(1));
                uni_vmovups_maybe_tail(vsqrtvar, var_ptr());
                uni_vaddps(vsqrtvar, vsqrtvar, veps);
                uni_vsqrtps(vsqrtvar, vsqrtvar);
                uni_vdivps(vsqrtvar, vone, vsqrtvar, vtmp);
                mov(reg_ctr, reg_nnthr);
                Label sh_reduction_thrs;
                L(sh_reduction_thrs);
                { // TODO: unroll (?)
                    uni_vaddps(Vmm(0), Vmm(0), vmmword[reg_rbuf1 + reg_roff]);
                    uni_vaddps(Vmm(1), Vmm(1), vmmword[reg_rbuf2 + reg_roff]);
                    add(reg_roff, reg_coff_max);
                    sub(reg_ctr, 1);
                    jnz(sh_reduction_thrs);
                }
                uni_vmulps(Vmm(0), Vmm(0), vsqrtvar);
                uni_vmovups_maybe_tail(diff_gamma_ptr(), Vmm(0));
                uni_vmovups_maybe_tail(diff_beta_ptr(), Vmm(1));
                add(reg_coff, isa == sse41 ? vlen / 2 : vlen);
                cmp(reg_coff, reg_coff_max);
                jne(sh_reduction_channels);
            }
        }
        L(no_sh_reduction);
        barrier();

        mov(reg_diff_src, ptr[rsp + stack_off_diff_src]);
        if (with_relu) {
            assert(isa == avx2 || isa == avx512_core);
            mov(reg_ws, ptr[rsp + stack_off_ws]);
        }

        xor_(reg_soff, reg_soff);
        Label diff_spatial;
        L(diff_spatial);
        {
            xor_(reg_coff, reg_coff);
            // diff_shift is shared with soff_max.
            mov(reg_diff_shift, ptr[rsp + stack_off_diff_shift]);
            if (isa == sse41) { mov(reg_tmp_off, reg_soff); }
            jbp_->is_nspc_ ? backward_diff_channels_nspc()
                           : backward_diff_channels();
            if (isa == sse41) {
                mov(reg_soff, reg_tmp_off);
                add(reg_diff_dst, vlen / 2);
                add(reg_diff_src, vlen / 2);
                add(reg_src, vlen / 2);
                mov(reg_coff, vlen / 2);
                backward_diff_channels();
                sub(reg_diff_dst, vlen / 2);
                sub(reg_diff_src, vlen / 2);
                sub(reg_src, vlen / 2);
            }
            // Process next image
            if (jbp_->is_nspc_) {
                // Can use static offset since we comeback after spatial loop
                if (!pd_->use_global_stats()) add(reg_src, mb_offt);
                add(reg_diff_dst, mb_offt);
                add(reg_diff_src, mb_offt);
                add(reg_soff, mb_offt);
                add(reg_ws, ws_mb_offt);
            } else {
                add(reg_soff, reg_mb_stride_Bc);
            }

            // comeback soff_max. Shared with diff_shift.
            mov(reg_soff_max, ptr[rsp + stack_off_soff_max]);
            cmp(reg_soff, reg_soff_max);
            jl(diff_spatial);
        }
        if (jbp_->is_nspc_) {
            // comeback
            if (!pd_->use_global_stats())
                mov(reg_src, ptr[rsp + stack_off_src]);
            mov(reg_diff_dst, ptr[rsp + stack_off_diff_dst]);
            mov(reg_diff_src, ptr[rsp + stack_off_diff_src]);
            if (with_relu) mov(reg_ws, ptr[rsp + stack_off_ws]);
        }
    }

    jit_bnorm_t(const batch_normalization_pd_t *pd, const jit_bnorm_conf_t *jbp)
        : jit_generator(jit_name()), pd_(pd), jbp_(jbp) {
        static_assert(isa == sse41 || isa == avx2 || isa == avx512_core,
                "unsupported isa");

        is_bf16_ = pd_->src_md()->data_type == data_type::bf16;
        is_f16_ = pd_->src_md()->data_type == data_type::f16;
        is_avx2_ne_xf16_
                = isa == avx2 && mayiuse(avx2_vnni_2) && (is_bf16_ || is_f16_);
        vlen_spat_data_ = vlen / (1 + is_xf16()); // 32B of xF16 -> 64B of FP32

        unroll_blocks = isa == avx512_core && !jbp_->is_spatial_thr_ ? 4 : 1;
        unroll_regs = isa == avx512_core && !jbp_->is_spatial_thr_ ? 4 : 1;
    }

    void generate() override {
        preamble();

        if (use_bf16_emulation()) {
            // init emulation of bfloat16 operations
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserved_1,
                    bf16_emu_reserved_2, bf16_emu_reserved_3, reg_bf16_tmp,
                    bf16_emu_reserved_4, bf16_emu_reserved_4);
            bf16_emu_->init_vcvtneps2bf16();
        }

        if (isa == avx512_core)
            prepare_tail_mask_avx512_common();
        else if (isa == avx2)
            prepare_tail_mask_avx2_common();

        compute_static_strides();

        prepare_relu();

        sub(rsp, stack_size_required);
        load_common_params();

        if (pd_->is_fwd()) {
            if (!pd_->stats_is_src()) { compute_mean_variance(); }
            forward();
        } else {
            backward();
        }
        add(rsp, stack_size_required);
        postamble();
    }

    void operator()(const call_params_t *p) { jit_generator::operator()(p); }

    ~jit_bnorm_t() override { delete bf16_emu_; }
};

namespace bnorm_impl {

template <cpu_isa_t isa>
struct driver_t : public c_compatible {
    driver_t(const batch_normalization_pd_t *pd, int nthr)
        : pd_(pd), jbp_(pd_, nthr, simd_w), ker_(pd_, &jbp_) {}

    ~driver_t() = default;

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const batch_normalization_pd_t *pd, int nthr) {
        dim_t C_PADDED = get_c_padded(pd);

        auto sbuf_sz = use_tmp_stats(pd) * 2 * C_PADDED;
        auto pbuf_sz
                = (use_tmp_diff_scale(pd) + use_tmp_diff_shift(pd)) * C_PADDED;
        auto rbuf_sz = (pd->is_fwd() ? 1 : 2) * C_PADDED * nthr;

        scratchpad.book<acc_data_t>(key_bnorm_tmp_stats, sbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_tmp_diff_ss, pbuf_sz);
        scratchpad.book<acc_data_t>(key_bnorm_reduction, rbuf_sz);

        if (dnnl_thr_syncable()) {
            auto n_barriers = C_PADDED / simd_w;
            scratchpad.book<barrier::ctx_64_t>(key_barrier, n_barriers);
        }
    }

    // given nthr, shape of problem, and thread partition,
    // balance work among the threads
    void thread_balance(int ithr, int nthr, dim_t N, dim_t C_blks, dim_t SP,
            int &C_ithr, int C_nthr, dim_t &C_blk_s, dim_t &C_blk_e,
            int &N_ithr, int N_nthr, dim_t &N_s, dim_t &N_e, int &S_ithr,
            int S_nthr, dim_t &S_s, dim_t &S_e) {
        if (ithr < C_nthr * N_nthr * S_nthr) {
            utils::nd_iterator_init(
                    ithr, C_ithr, C_nthr, N_ithr, N_nthr, S_ithr, S_nthr);
            balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
            balance211(N, N_nthr, N_ithr, N_s, N_e);
            balance211(SP, S_nthr, S_ithr, S_s, S_e);
        } else {
            S_ithr = N_ithr = C_ithr = -ithr;
            S_s = S_e = N_s = N_e = C_blk_s = C_blk_e = -1;
        }
    }

    void exec(int ithr, int nthr, const void *src, void *diff_src, void *dst,
            const void *diff_dst, const acc_data_t *scale,
            acc_data_t *diff_scale, const acc_data_t *shift,
            acc_data_t *diff_shift, const acc_data_t *mean,
            const acc_data_t *var, const uint8_t *ws,
            const memory_tracking::grantor_t &scratchpad) {
        auto sbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_stats);
        auto pbuf = scratchpad.get<acc_data_t>(key_bnorm_tmp_diff_ss);
        auto rbuf = scratchpad.get<acc_data_t>(key_bnorm_reduction);
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);

        dim_t N = pd_->MB();
        dim_t C = pd_->C();
        dim_t C_PADDED = get_c_padded(pd_);
        dim_t D = pd_->D();
        dim_t H = pd_->H();
        dim_t W = pd_->W();
        dim_t SP = D * H * W;
        dim_t img_size = C_PADDED * SP;
        const int vlen_spat_data = ker_.spat_step;

        typename jit_bnorm_t<isa>::call_params_t p;

        p.eps = pd_->desc()->batch_norm_epsilon;
        p.one = 1.0f;
        p.spat_size = SP;
        p.chan_size = 1.0f * N * p.spat_size;

        int C_ithr {0}, N_ithr {0}, S_ithr {0};
        dim_t C_blk_s {0}, C_blk_e {0}, N_s {0}, N_e {0}, S_s {0}, S_e {0};

        this->thread_balance(ithr, nthr, N, jbp_.C_blks_per_iter_, SP, C_ithr,
                jbp_.C_nthr_, C_blk_s, C_blk_e, N_ithr, jbp_.N_nthr_, N_s, N_e,
                S_ithr, jbp_.S_nthr_, S_s, S_e);

        int SP_N_ithr = N_ithr * jbp_.S_nthr_ + S_ithr;
        int SP_N_nthr = jbp_.N_nthr_ * jbp_.S_nthr_;
        assert(IMPLICATION(!dnnl_thr_syncable(), SP_N_nthr == 1));

        p.N_ithr = SP_N_ithr;
        p.N_nthr = SP_N_nthr;

        int global_C_blk_s;
        int global_barriers_per_iter = jbp_.C_nthr_;

        for (int64_t it = 0; it < jbp_.iters_; it++) {
            if (it == jbp_.iters_ - 1 && jbp_.iters_ > 1) {
                C_blk_s = C_blk_e = N_s = N_e = 0;
                this->thread_balance(ithr, nthr, N, jbp_.C_blks_last_iter_, SP,
                        C_ithr, jbp_.C_nthr_last_iter_, C_blk_s, C_blk_e,
                        N_ithr, jbp_.N_nthr_last_iter_, N_s, N_e, S_ithr,
                        jbp_.S_nthr_last_iter_, S_s, S_e);

                // Update call parameters for JIT, last iteration
                p.N_ithr = N_ithr * jbp_.S_nthr_last_iter_ + S_ithr;
                p.N_nthr = jbp_.N_nthr_last_iter_ * jbp_.S_nthr_last_iter_;
            }

            global_C_blk_s = jbp_.do_blocking_ ? (C_blk_s == -1)
                            ? -1
                            : it * jbp_.C_blks_per_iter_ + C_blk_s
                                               : C_blk_s;

            int C_blks_thr = C_blk_e - C_blk_s;
            int N_thr = N_e - N_s;

            if (C_blks_thr == 0 || N_thr == 0) continue;

            size_t coff_base = global_C_blk_s * simd_w;
            size_t soff_base = jbp_.is_nspc_
                    ? coff_base + N_s * img_size
                    : global_C_blk_s * p.spat_size * simd_w + N_s * img_size;
            size_t shift_off = use_tmp_diff_scale(pd_) ? pd_->C() : 0;

            p.spat_size_loc = S_e - S_s;
            p.S_s = S_s * vlen_spat_data;
            p.S_tail = (p.spat_size - S_e) * vlen_spat_data;
            p.coff_max = C_blks_thr * simd_w;
            const auto tmp_mean = use_tmp_stats(pd_) ? sbuf : mean;
            if (tmp_mean != nullptr) p.mean = tmp_mean + coff_base;
            const auto tmp_var = use_tmp_stats(pd_) ? sbuf + C_PADDED : var;
            if (tmp_var != nullptr) p.var = tmp_var + coff_base;
            if (scale != nullptr) p.scale = scale + coff_base;
            if (shift != nullptr) p.shift = shift + coff_base;
            const auto tmp_diff_scale
                    = use_tmp_diff_scale(pd_) ? pbuf : diff_scale;
            if (tmp_diff_scale != nullptr)
                p.diff_scale = tmp_diff_scale + coff_base;
            const auto tmp_diff_shift
                    = use_tmp_diff_shift(pd_) ? &pbuf[shift_off] : diff_shift;
            if (tmp_diff_shift != nullptr)
                p.diff_shift = tmp_diff_shift + coff_base;

            p.soff_max = jbp_.dt_size_ * N_thr * img_size;
            if (src != nullptr)
                p.src = (void *)((char *)src + soff_base * jbp_.dt_size_);
            if (dst != nullptr)
                p.dst = (void *)((char *)dst + soff_base * jbp_.dt_size_);
            if (diff_src != nullptr)
                p.diff_src = (void *)((char *)diff_src
                        + soff_base * jbp_.dt_size_);
            if (diff_dst != nullptr)
                p.diff_dst = (void *)((char *)diff_dst
                        + soff_base * jbp_.dt_size_);
            if (ws != nullptr) p.ws = ws + soff_base / 8;

            p.mb_stride_Bc
                    = jbp_.dt_size_ * (img_size - p.coff_max * p.spat_size);

            // use SP_N_nthr which is the same as p.N_nthr except maybe for
            // the last iteration.
            p.rbuf1 = rbuf
                    + ((it * jbp_.C_blks_per_iter_) * SP_N_nthr
                              + C_blk_s * p.N_nthr + p.N_ithr * C_blks_thr)
                            * simd_w;
            // rbuf1 and rbuf2 have to be disjoint
            p.rbuf2 = p.rbuf1 + C_PADDED * nthr;
            p.is_cblk_tail
                    = (it * jbp_.C_blks_per_iter_ + C_blk_e) * simd_w > C;

            size_t iter_barriers
                    = jbp_.do_blocking_ ? it * global_barriers_per_iter : 0;
            p.barrier = barriers + C_ithr + iter_barriers;
            if (p.soff_max != 0 && p.coff_max != 0) ker_(&p);
        }
    }

    void init_barriers(const memory_tracking::grantor_t &scratchpad) {
        auto barriers = scratchpad.get<barrier::ctx_64_t>(key_barrier);
        if (barriers) {
            const int n_barriers = get_c_padded(pd_) / simd_w;
            for (int i = 0; i < n_barriers; ++i)
                barrier::ctx_init(&barriers[i]);
        }
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    enum {
        simd_w = isa == sse41 ? 8
                              : cpu_isa_traits<isa>::vlen
                        / sizeof(acc_data_t) // BF16 will expand to FP32
    };

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

    const batch_normalization_pd_t *pd_;
    jit_bnorm_conf_t jbp_;
    jit_bnorm_t<isa> ker_;
};
} // namespace bnorm_impl

using namespace data_type;
using namespace format_tag;
using namespace utils;

/* fwd */

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::pd_t::init(engine_t *engine) {
    bool ok = is_fwd() && mayiuse(isa)
            && !has_zero_dim_memory()
            // Algorithm requires barriers for best performance.
            // TBB utilizes jit_uni_tbb_batch_normalization implementation.
            && dnnl_thr_syncable()
            && one_of(src_md()->data_type, f32, bf16, f16)
            && src_md()->data_type == dst_md()->data_type
            && IMPLICATION(src_md()->data_type == bf16,
                    is_superset(isa, avx512_core)
                            || (isa == avx2 && mayiuse(avx2_vnni_2)))
            // Note: re-using avx512_core/avx2 implementation for f16.
            // This is okay as currently, we do not support binary post-ops
            // for this primitive.
            && IMPLICATION(src_md()->data_type == f16,
                    (is_superset(isa, avx512_core) && mayiuse(avx512_core_fp16))
                            || (isa == avx2 && mayiuse(avx2_vnni_2)))
            && check_scale_shift_data_type()
            && (attr()->has_default_values()
                    || with_relu_post_op(is_training()))
            && set_default_formats_common()
            && memory_desc_wrapper(src_md()) == memory_desc_wrapper(dst_md());
    if (!ok) return status::unimplemented;

    // BN+Add+Relu fusion is not currently implemented
    if (fuse_norm_add_relu()) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md());
    if (isa == avx512_core) {
        if (!src_d.matches_one_of_tag(
                    nCw16c, nChw16c, nCdhw16c, nc, nwc, nhwc, ndhwc))
            return status::unimplemented;
    } else if (isa == avx2 && one_of(src_md()->data_type, bf16, f16)) {
        // no support for training or blocked layouts for avx2_vnni_2
        if (is_training() || !src_d.matches_one_of_tag(nc, nwc, nhwc, ndhwc))
            return status::unimplemented;
    } else if (isa == avx2) {
        // full support
        if (!src_d.matches_one_of_tag(
                    nCw8c, nChw8c, nCdhw8c, nc, nwc, nhwc, ndhwc))
            return status::unimplemented;
    } else {
        if (!src_d.matches_one_of_tag(nCw8c, nChw8c, nCdhw8c))
            return status::unimplemented;
    }

    const bool isa_supports_avx2 = is_superset(isa, avx2);
    if (is_training() && fuse_norm_relu()) {
        if (!isa_supports_avx2) return status::unimplemented;
        init_default_ws(1);
    }

    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C()
            && !isa_supports_avx2)
        return status::unimplemented;

    // Only IC % simd_w == 0 is supported for now
    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(acc_data_t);
    if (src_d.matches_one_of_tag(nc, nwc, nhwc, ndhwc)
            && src_d.padded_dims()[1] % simd_w != 0) {
        return status::unimplemented;
    }

    nthr_ = dnnl_get_max_threads();
    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this, nthr_);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::jit_uni_batch_normalization_fwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            bnorm_driver_, new bnorm_impl::driver_t<isa>(pd(), pd()->nthr_)));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_fwd_t<isa>::execute(
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

    bnorm_driver_->init_barriers(scratchpad);
    const int nthr = pd()->nthr_;

    parallel(nthr, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, nullptr, dst, nullptr, scale,
                nullptr, shift, nullptr, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_fwd_t<isa>::~jit_uni_batch_normalization_fwd_t() {
    delete bnorm_driver_;
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::pd_t::init(engine_t *engine) {
    bool ok = !is_fwd() && mayiuse(isa)
            && !has_zero_dim_memory()
            // Algorithm requires barriers for best performance.
            // TBB utilizes jit_uni_tbb_batch_normalization implementation.
            && dnnl_thr_syncable()
            && one_of(src_md()->data_type, f32, bf16, f16)
            && src_md()->data_type == diff_src_md()->data_type
            && diff_src_md()->data_type == diff_dst_md()->data_type
            && IMPLICATION(
                    src_md()->data_type == bf16, is_superset(isa, avx512_core))
            // Note: re-using avx512_core implementation for f16. This is okay
            // as currently, we do not support binary post-ops for this
            // primitive.
            && IMPLICATION(src_md()->data_type == f16,
                    is_superset(isa, avx512_core) && mayiuse(avx512_core_fp16))
            && check_scale_shift_data_type() && attr()->has_default_values()
            && set_default_formats_common()
            && memory_desc_wrapper(diff_src_md())
                    == memory_desc_wrapper(diff_dst_md());
    if (!ok) return status::unimplemented;

    // BN+Add+Relu fusion is not currently implemented
    if (fuse_norm_add_relu()) return status::unimplemented;

    const memory_desc_wrapper src_d(src_md());
    const memory_desc_wrapper diff_src_d(diff_src_md());

    format_tag_t src_tag, diff_src_tag;
    if (isa == avx512_core) {
        src_tag = src_d.matches_one_of_tag(
                nc, nwc, nCw16c, nhwc, nChw16c, ndhwc, nCdhw16c);
        diff_src_tag = diff_src_d.matches_one_of_tag(
                nc, nwc, nCw16c, nhwc, nChw16c, ndhwc, nCdhw16c);
    } else {
        src_tag = src_d.matches_one_of_tag(nCw8c, nChw8c, nCdhw8c);
        diff_src_tag = diff_src_d.matches_one_of_tag(nCw8c, nChw8c, nCdhw8c);
    }
    ok = (src_tag != format_tag::undef && diff_src_tag != format_tag::undef
            && src_tag == diff_src_tag);
    if (!ok) return status::unimplemented;

    const bool isa_supports_avx2 = is_superset(isa, avx2);
    if (memory_desc_wrapper(src_md()).padded_dims()[1] != C()
            && !isa_supports_avx2)
        return status::unimplemented;

    // Only IC % 16 == 0 is supported for now
    if (src_d.matches_one_of_tag(nc, nwc, nhwc, ndhwc)
            && src_d.padded_dims()[1] % 16 != 0) {
        return status::unimplemented;
    }

    if (fuse_norm_relu()) {
        if (!isa_supports_avx2) return status::unimplemented;
        init_default_ws(1);
        if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
    }

    /* TODO: extra checks required */

    nthr_ = dnnl_get_max_threads();
    auto scratchpad = scratchpad_registry().registrar();
    bnorm_impl::driver_t<isa>::init_scratchpad(scratchpad, this, nthr_);

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::jit_uni_batch_normalization_bwd_t(
        const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(
            bnorm_driver_, new bnorm_impl::driver_t<isa>(pd(), pd()->nthr_)));
    return bnorm_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_batch_normalization_bwd_t<isa>::execute(
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

    bnorm_driver_->init_barriers(scratchpad);
    const int nthr = pd()->nthr_;

    parallel(nthr, [&](const int ithr, const int nthr) {
        bnorm_driver_->exec(ithr, nthr, src, diff_src, nullptr, diff_dst, scale,
                diff_scale, nullptr, diff_shift, mean, var, ws, scratchpad);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_batch_normalization_bwd_t<isa>::~jit_uni_batch_normalization_bwd_t() {
    delete bnorm_driver_;
}

/* struct instantiation */
template struct jit_uni_batch_normalization_fwd_t<sse41>;
template struct jit_uni_batch_normalization_bwd_t<sse41>;
template struct jit_uni_batch_normalization_fwd_t<avx2>;
template struct jit_uni_batch_normalization_bwd_t<avx2>;
template struct jit_uni_batch_normalization_fwd_t<avx512_core>;
template struct jit_uni_batch_normalization_bwd_t<avx512_core>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
