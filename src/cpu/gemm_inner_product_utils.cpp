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

#include "gemm_inner_product_utils.hpp"
#include "dnnl_thread.hpp"
#include "jit_uni_eltwise_injector.hpp"
#include "math_utils.hpp"
#include "simple_q10n.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace inner_product_utils {

using namespace alg_kind;
using namespace math;

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type>::pp_kernel_t(size_t OC, size_t MB,
        const primitive_attr_t *attr, data_type_t bias_dt, bool skip_sum)
    : ker_(nullptr)
    , eltwise_injector_(nullptr)
    , ref_eltwise_(nullptr)
    , bf16_emu_(nullptr)
    , OC_(OC)
    , MB_(MB)
    , bias_data_type_(bias_dt)
    , bias_data_type_size_(0)
    , do_scale_(false)
    , scale_idx_mult_(0)
    , do_eltwise_(false)
    , do_sum_(false)
    , do_dst_zero_points_(false)
    , sum_scale_(0)
    , isa_(isa_any)
    , max_OC_loop_unroll_(13)
    , idx_compute_vreg_start_(0)
    , idx_compute_vreg_max_(31)
    , compute_vregs_per_iter_(1)
    , compute_vreg_bias_shift_(0)
    , compute_vreg_prev_dst_shift_(0) {
    using namespace types;
    using namespace Xbyak;

    do_scale_ = !attr->output_scales_.has_default_values();
    if (do_scale_) {
        scale_idx_mult_ = (attr->output_scales_.mask_ == (1 << 1));
        vreg_scale = Zmm(idx_compute_vreg_start_++);
    }
    if (dst_type == data_type::u8) vreg_zero = Zmm(idx_compute_vreg_start_++);

    auto &p = attr->post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    do_eltwise_ = eltwise_ind != -1;
    if (do_eltwise_) eltwise_ = p.entry_[eltwise_ind].eltwise;

    const int sum_ind = p.find(primitive_kind::sum);
    do_sum_ = sum_ind != -1 && !skip_sum;
    if (do_sum_) {
        sum_scale_ = p.entry_[sum_ind].sum.scale;
        vreg_sum_scale = Zmm(idx_compute_vreg_start_++);
        compute_vreg_prev_dst_shift_ = compute_vregs_per_iter_++;
    }

    if (do_bias()) {
        bias_data_type_size_ = data_type_size(bias_data_type_);
        compute_vreg_bias_shift_ = compute_vregs_per_iter_++;
    }

    if (!attr->zero_points_.has_default_values(DNNL_ARG_DST)) {
        do_dst_zero_points_ = true;
        vreg_dst_zero_points = Zmm(idx_compute_vreg_start_++);
    }

    if (!mayiuse(avx512_core)) {
        // use fallback code for older CPUs since they do not have optimized
        // x8s8s32 GEMM anyways. The configuration variables above are used by
        // the fallback code.
        if (do_eltwise_)
            ref_eltwise_ = new ref_eltwise_scalar_fwd_t(eltwise_.alg,
                    eltwise_.alpha, eltwise_.beta, eltwise_.scale);
        return;
    } else {
        isa_ = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                         : bf16_emulation_t::get_isa();
        if (dst_type == data_type::bf16 && isa_ != avx512_core_bf16) {
            idx_compute_vreg_max_ = 27;
            bf16_emu_ = new bf16_emulation_t(this, bf16_emu_reserv_1,
                    bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                    bf16_emu_reserv_5);
        }

        int max_unroll = (idx_compute_vreg_max_ - idx_compute_vreg_start_ + 1)
                / compute_vregs_per_iter_;
        max_OC_loop_unroll_ = nstl::min(max_OC_loop_unroll_, max_unroll);

        if (do_eltwise_)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_core>(
                    this, eltwise_, true, eltwise_reserved_1_,
                    eltwise_reserved_2_);
        generate();
    }
}

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type>::pp_kernel_t(
        const cpu_inner_product_fwd_pd_t *pd, bool skip_sum)
    : pp_kernel_t(pd->OC(), pd->MB(), pd->attr(),
            pd->desc()->bias_desc.data_type, skip_sum) {}

template <data_type_t acc_type, data_type_t dst_type>
void pp_kernel_t<acc_type, dst_type>::compute_oc_channel_blk() {
    using namespace Xbyak;
    using namespace utils;

    // Load accumulated value, convert to float, apply bias (if any), scaling,
    // and eltwise (if any); then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];
        if (dst_type == data_type::bf16 && isa_ != avx512_core_bf16)
            bf16_emu_->init_vcvtneps2bf16();

        if (do_scale_ && scale_idx_mult_ == 1) {
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_msk_ = vreg_scale;
            if (apply_mask) vreg_scale_msk_ = vreg_scale_msk_ | kreg_rem_mask;
            vmovups(vreg_scale_msk_, scale_addr);
        }

        auto vreg_dst_ = vreg_dst(idx);
        auto vreg_dst_msk_ = apply_mask ? vreg_dst_ | kreg_rem_mask : vreg_dst_;

        switch (acc_type) {
            case data_type::s32: vcvtdq2ps(vreg_dst_msk_, acc_addr); break;
            case data_type::f32: vmovups(vreg_dst_msk_, acc_addr); break;
        }

        if (do_bias()) {
            auto bias_addr = ptr[reg_bias + offset * bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            auto vreg_bias_msk_
                    = apply_mask ? vreg_bias_ | kreg_rem_mask : vreg_bias_;

            switch (bias_data_type_) {
                case data_type::s8: vpmovsxbd(vreg_bias_msk_, bias_addr); break;
                case data_type::u8: vpmovzxbd(vreg_bias_msk_, bias_addr); break;
                case data_type::s32:
                case data_type::f32: vmovups(vreg_bias_msk_, bias_addr); break;
                case data_type::bf16:
                    vpmovzxwd(vreg_bias_msk_, bias_addr);
                    vpslld(vreg_bias_, vreg_bias_, 0x10);
                    break;
                default: assert(!"unimplemented");
            }
            if (utils::one_of(bias_data_type_, data_type::u8, data_type::s8,
                        data_type::s32))
                vcvtdq2ps(vreg_bias_, vreg_bias_);
            vaddps(vreg_dst_, vreg_dst_, vreg_bias_);
        }

        if (do_scale_) vmulps(vreg_dst_, vreg_dst_, vreg_scale);

        auto dst_addr = ptr[reg_dst + offset * sizeof(dst_data_t)];
        if (do_sum_) {
            auto vreg_prev_dst_ = vreg_prev_dst(idx);
            auto vreg_prev_dst_msk_ = apply_mask
                    ? vreg_prev_dst_ | kreg_rem_mask
                    : vreg_prev_dst_;

            switch (dst_type) {
                case data_type::f32:
                case data_type::s32:
                    vmovups(vreg_prev_dst_msk_, dst_addr);
                    break;
                case data_type::s8:
                    vpmovsxbd(vreg_prev_dst_msk_, dst_addr);
                    break;
                case data_type::u8:
                    vpmovzxbd(vreg_prev_dst_msk_, dst_addr);
                    break;
                case data_type::bf16:
                    vpmovzxwd(vreg_prev_dst_msk_, dst_addr);
                    vpslld(vreg_prev_dst_, vreg_prev_dst_, 0x10);
                    break;
                default: assert(!"unsupported data type");
            }
            if (utils::one_of(
                        dst_type, data_type::u8, data_type::s8, data_type::s32))
                vcvtdq2ps(vreg_prev_dst_, vreg_prev_dst_);

            vfmadd231ps(vreg_dst_, vreg_prev_dst_, vreg_sum_scale);
        }

        if (do_eltwise_) eltwise_injector_->compute_vector(vreg_dst_.getIdx());

        if (do_dst_zero_points_)
            vaddps(vreg_dst_, vreg_dst_, vreg_dst_zero_points);

        if (dst_type == data_type::u8) vmaxps(vreg_dst_, vreg_dst_, vreg_zero);

        if (utils::one_of(
                    dst_type, data_type::s8, data_type::u8, data_type::s32)) {
            vcvtps2dq(vreg_dst_, vreg_dst_);
        } else if (dst_type == data_type::bf16) {
            if (isa_ == avx512_core_bf16)
                vcvtneps2bf16(Ymm(vreg_dst_.getIdx()), vreg_dst_);
            else
                bf16_emu_->vcvtneps2bf16(Ymm(vreg_dst_.getIdx()), vreg_dst_);
        }

        switch (dst_type) {
            case data_type::s8: vpmovsdb(dst_addr, vreg_dst_msk_); break;
            case data_type::u8: vpmovusdb(dst_addr, vreg_dst_msk_); break;
            case data_type::f32:
            case data_type::s32: vmovups(dst_addr, vreg_dst_msk_); break;
            case data_type::bf16:
                vmovdqu16(dst_addr,
                        apply_mask ? Ymm(vreg_dst_.getIdx()) | kreg_rem_mask
                                   : Ymm(vreg_dst_.getIdx()));
                break;
            default: assert(!"unimplemented");
        }
    };

    // Advance all pointers by an immediate
    auto advance_ptrs_imm = [&](size_t offset) {
        add(reg_dst, offset * sizeof(dst_data_t));
        add(reg_acc, offset * sizeof(acc_data_t));
        if (do_scale_ && scale_idx_mult_ == 1)
            add(reg_scales, offset * sizeof(float));
        if (do_bias()) add(reg_bias, offset * bias_data_type_size_);
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](Reg64 offset) {
        lea(reg_dst, ptr[reg_dst + offset * sizeof(dst_data_t)]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (do_scale_ && scale_idx_mult_ == 1)
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        if (do_bias())
            lea(reg_bias, ptr[reg_bias + offset * bias_data_type_size_]);
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        neg(reg_oc);
        if (do_bias())
            lea(reg_bias, ptr[reg_bias + reg_oc * bias_data_type_size_]);
        if (do_scale_ && scale_idx_mult_ == 1)
            lea(reg_scales, ptr[reg_scales + reg_oc * sizeof(float)]);
        neg(reg_oc);
    };

    // Process one row of reg_tmp elements
    auto process_runtime_oc = [&]() {
        Label l_loop, l_loop_tail, l_loop_end;
        cmp(reg_tmp, vlen);
        jle(l_loop_tail, T_NEAR); // Skips for reg_tmp == 16 too (?)

        L(l_loop);
        {
            compute(0, 0, false);
            advance_ptrs_imm(vlen);
            sub(reg_tmp, vlen);
            cmp(reg_tmp, vlen);
            jge(l_loop, T_NEAR);
        }

        L(l_loop_tail);
        mov(reg_rem_mask, 1);
        shl(reg_rem_mask, cl); // cl == reg_tmp because reg_tmp <= vlen here
        sub(reg_rem_mask, 1);
        jz(l_loop_end, T_NEAR);

        kmovq(kreg_rem_mask, reg_rem_mask);
        compute(0, 0, true);
        advance_ptrs_reg(reg_tmp);

        L(l_loop_end);
    };

    //      <-------------------- OC ------------------------------->
    //
    // ^    +....................+----------------------------------+
    // |    :   not accessed     |          Prologue loop           |
    // |    +--------------------+----------------------------------+
    //      |                                                       |
    // M    |                 Main loop (unrolled)                  |
    // B    |                                                       |
    //      +--------------------------------+----------------------+
    // |    |       Epilogue loop            |      not accessed    :
    // v    +--------------------------------+......................+

    // Prologue loop
    Label l_prologue_end;
    cmp(reg_oc_offset, 0);
    je(l_prologue_end, T_NEAR);
    {
        mov(reg_tmp, reg_oc);
        sub(reg_tmp, reg_oc_offset);
        cmp(reg_tmp, reg_len);
        cmovg(reg_tmp, reg_len);
        sub(reg_len, reg_tmp);

        process_runtime_oc();
        rewind_ptrs();
    }
    L(l_prologue_end);

    // Main loop
    Label l_main_loop_end;
    cmp(reg_len, reg_oc);
    jle(l_main_loop_end, T_NEAR);
    if (runtime_oc()) {
        Label l_main_loop;
        L(l_main_loop);
        {
            mov(reg_tmp, reg_oc);

            process_runtime_oc();
            rewind_ptrs();

            sub(reg_len, reg_oc);
            cmp(reg_len, reg_oc);
            jge(l_main_loop, T_NEAR);
        }
    } else {
        Label l_main_loop;
        L(l_main_loop);
        {
            size_t OC_loop, OC_tail;
            if (OC_ < max_OC_loop_unroll_ * vlen) {
                // Fully unroll small loops
                OC_loop = 0;
                OC_tail = OC_;
            } else {
                OC_loop = vlen * default_OC_loop_unroll_;
                OC_tail = OC_ % OC_loop;
            }

            assert(!!OC_loop || !!OC_tail);

            if (OC_tail % vlen) {
                int vlen_tail = OC_tail % vlen;
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask, reg_tmp);
            }

            if (OC_loop) {
                mov(reg_tmp, rnd_dn(OC_, OC_loop));
                Label l_oc_loop;
                L(l_oc_loop);
                {
                    for (size_t offset = 0; offset < OC_loop; offset += vlen)
                        compute(offset, offset / vlen, false);
                    advance_ptrs_imm(OC_loop);
                    sub(reg_tmp, OC_loop);
                    jnz(l_oc_loop);
                }
            }

            if (OC_tail) {
                for (size_t offset = 0; offset < OC_tail; offset += vlen) {
                    bool use_mask = (offset + vlen) > OC_tail;
                    compute(offset, offset / vlen, use_mask);
                }
                advance_ptrs_imm(OC_tail);
            }

            rewind_ptrs();
            sub(reg_len, reg_oc);
            cmp(reg_len, reg_oc);
            jge(l_main_loop, T_NEAR);
        }
    }
    L(l_main_loop_end);

    // Epilogue loop
    Label l_epilogue_end;
    cmp(reg_len, 0);
    je(l_epilogue_end, T_NEAR);
    {
        mov(reg_tmp, reg_len);
        process_runtime_oc();
    }
    L(l_epilogue_end);
}

template <data_type_t acc_type, data_type_t dst_type>
void pp_kernel_t<acc_type, dst_type>::generate() {
    using namespace Xbyak;
    using namespace utils;

    preamble();

#define PARAM_OFF(x) offsetof(ker_args, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    if (do_scale_) mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    if (do_dst_zero_points_) {
        // use reg_oc as a temporary one (alas, reg_tmp = reg_param on Windows)
        mov(reg_oc, ptr[reg_param + PARAM_OFF(dst_zero_points)]);
        vbroadcastss(vreg_dst_zero_points, ptr[reg_oc]);
    }
    if (runtime_oc())
        mov(reg_oc, ptr[reg_param + PARAM_OFF(oc)]);
    else
        mov(reg_oc, OC_);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    if (do_scale_ && scale_idx_mult_ == 0)
        vbroadcastss(vreg_scale, dword[reg_scales]);
#undef PARAM_OFF

    if (do_sum_) {
        mov(reg_tmp, float2int(sum_scale_));
        auto xreg_sum_scale = Xmm(vreg_sum_scale.getIdx());
        vmovq(xreg_sum_scale, reg_tmp);
        vbroadcastss(vreg_sum_scale, xreg_sum_scale);
    }

    if (dst_type == data_type::u8) vxorps(vreg_zero, vreg_zero, vreg_zero);

    compute_oc_channel_blk();

    postamble();

    if (do_eltwise_) eltwise_injector_->prepare_table();

    ker_ = getCode<decltype(ker_)>();
}

template <data_type_t acc_type, data_type_t dst_type>
void pp_kernel_t<acc_type, dst_type>::operator()(dst_data_t *dst,
        const acc_data_t *acc, const char *bias, const float *scales,
        size_t start, size_t end, size_t runtime_oc,
        const float *dst_zero_points) {
    using math::get_bias;

    if (end <= start) return;

    const size_t OC = this->runtime_oc() ? runtime_oc : OC_;

    if (ker_) {
        // JIT
        ker_args args;
        size_t oc_offset = start % OC;
        args.dst = dst + start;
        args.acc = acc + start;
        args.bias = bias + oc_offset * bias_data_type_size_;
        args.scales = scales + scale_idx_mult_ * oc_offset;
        args.dst_zero_points = dst_zero_points;
        args.oc = OC;
        args.len = end - start;
        args.oc_offset = oc_offset;
        ker_(&args);
    } else {
        // Fallback
        size_t oc = start % OC;
        for (size_t i = start; i < end; i++) {
            float d = (float)acc[i];
            if (do_bias()) d += get_bias(bias, oc, bias_data_type_);
            if (do_scale_) d *= scales[oc * scale_idx_mult_];
            if (do_sum_) d += sum_scale_ * dst[i];
            if (do_eltwise_) d = ref_eltwise_->compute_scalar(d);
            if (do_dst_zero_points_) d += dst_zero_points[0];
            dst[i] = qz_a1b0<float, dst_data_t>()(d);
            oc = (oc == OC - 1) ? 0 : oc + 1;
        }
    }
};

using namespace data_type;
template class pp_kernel_t<f32, f32>;
template class pp_kernel_t<s32, f32>;
template class pp_kernel_t<s32, s32>;
template class pp_kernel_t<s32, s8>;
template class pp_kernel_t<s32, u8>;
template class pp_kernel_t<f32, bf16>;
} // namespace inner_product_utils

} // namespace cpu
} // namespace impl
} // namespace dnnl
