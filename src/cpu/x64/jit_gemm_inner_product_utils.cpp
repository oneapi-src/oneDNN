/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <memory>

#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "cpu/simple_q10n.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_gemm_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace inner_product_utils {

using namespace dnnl::impl::cpu::inner_product_utils;
using namespace Xbyak;

template <data_type_t acc_type, data_type_t dst_type>
struct jit_pp_kernel_t : public pp_kernel_t<acc_type, dst_type>,
                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(inner_product_utils::jit_pp_kernel_t);

    jit_pp_kernel_t(size_t OC, size_t MB, dim_t dst_mb_stride,
            const primitive_attr_t *attr, data_type_t bias_dt,
            const memory_desc_t *dst_md, bool skip_sum);

    using acc_data_t = typename prec_traits<acc_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    void operator()(dst_data_t *dst, const acc_data_t *acc, const char *bias,
            const float *scales, size_t start, size_t end, size_t runtime_oc,
            dim_t dst_mb_stride, const float *dst_zero_points,
            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
            const exec_ctx_t &ctx, const memory_desc_t &dst_md) const override;

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    void apply_postops(const bool apply_mask, const int vmm_idx);
    void generate() override;
    void compute_oc_channel_blk();
    void compute_mb_blk(); // vectorize across minibatch
    template <typename T>
    void advance_binary_postops_off(const T &offset);
    void zero_binary_postops_off();

    struct ker_args_t {
        dst_data_t *dst = nullptr;
        const acc_data_t *acc = nullptr;
        const char *bias = nullptr;
        const float *scales = nullptr;
        const float *dst_zero_points = nullptr;
        float nslope = 0;
        size_t oc = 0;
        size_t len = 0;
        size_t oc_offset = 0;
        dim_t dst_mb_stride = 0;
        const void *post_ops_binary_rhs_arg_vec = nullptr;
        const void *dst_orig = nullptr;
    };

    enum { default_OC_loop_unroll_ = 4 };

    std::unique_ptr<injector::jit_uni_postops_injector_t<avx512_core>>
            postops_injector_;

    std::unique_ptr<bf16_emulation_t> bf16_emu_;

#ifdef _WIN32
    const Xbyak::Reg64 reg_binary_inj_param_ = abi_not_param1;
#else
    const Xbyak::Reg64 reg_binary_inj_param_ = abi_param1;
#endif

    Xbyak::Reg64 reg_param = abi_param1;
    Xbyak::Reg64 reg_dst = rdx;
    Xbyak::Reg64 reg_acc = rax;
    Xbyak::Reg64 reg_bias = rbx;
    Xbyak::Reg64 reg_scales = rsi;

    Xbyak::Reg64 reg_oc = r13;
    Xbyak::Reg64 reg_len = r8;
    Xbyak::Reg64 reg_tmp = rcx; // intentional for shifting purposes
    Xbyak::Reg64 reg_oc_offset = r9;
    Xbyak::Reg64 reg_rem_mask = r10;
    Xbyak::Opmask kreg_rem_mask = k1;
    const Xbyak::Opmask &opmask_binary = k3;
    // register used for temp computation, needs not to be preserved
    Xbyak::Reg64 reg_tmp_comp = r15;

    // *mb_stride used only in matmul_pp_kernel && compute_oc_channel_blk()
    Xbyak::Reg64 reg_dst_mb_stride = r12;
    Xbyak::Reg64 reg_acc_mb_stride = r14;

    // Will be assigned in constructor
    Xbyak::Zmm vreg_zero, vreg_saturation_ubound, vreg_scale, vreg_sum_scale,
            vreg_dst_zero_points;

    Xbyak::Reg64 eltwise_reserved_gpr_ = r11;
    const Xbyak::Opmask &eltwise_reserved_opmask_ = k2;

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(28);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(30);
    Xbyak::Reg64 bf16_emu_reserv_4 = reg_tmp_comp;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(31);

    cpu_isa_t isa_ = isa_any;
    int max_OC_loop_unroll_ = 13;
    int idx_compute_vreg_start_ = 0;
    int idx_compute_vreg_max_ = 31;
    int compute_vregs_per_iter_ = 1;
    int compute_vreg_bias_shift_ = 0;
    int compute_vreg_prev_dst_shift_ = 0;

    const size_t vlen = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    constexpr static int reg64_size = sizeof(int64_t);
    constexpr static int reg_binary_post_op_oc_off = 0;
    constexpr static int stack_space_needed = 1 * reg64_size;

    int vreg_dst_idx(const int iter) const {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }

    Xbyak::Zmm vreg_dst(int iter) { return Xbyak::Zmm(vreg_dst_idx(iter)); }

    Xbyak::Zmm vreg_prev_dst(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_
                + compute_vreg_prev_dst_shift_;
        assert(idx <= idx_compute_vreg_max_);
        return Xbyak::Zmm(idx);
    }

    Xbyak::Zmm vreg_bias(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_
                + compute_vreg_bias_shift_;
        assert(idx <= idx_compute_vreg_max_);
        return Xbyak::Zmm(idx);
    }
};

template <data_type_t acc_type, data_type_t dst_type>
jit_pp_kernel_t<acc_type, dst_type>::jit_pp_kernel_t(size_t OC, size_t MB,
        dim_t dst_mb_stride, const primitive_attr_t *attr, data_type_t bias_dt,
        const memory_desc_t *dst_md, bool skip_sum)
    : pp_kernel_t<acc_type, dst_type>(
            OC, MB, dst_mb_stride, attr, bias_dt, skip_sum) {
    assert(mayiuse(avx512_core));

    if (this->do_scale_) vreg_scale = Zmm(idx_compute_vreg_start_++);

    if (dst_type == data_type::u8) vreg_zero = Zmm(idx_compute_vreg_start_++);
    if (utils::one_of(dst_type, data_type::u8, data_type::s8, data_type::s32))
        vreg_saturation_ubound = Zmm(idx_compute_vreg_start_++);

    if (this->do_sum_) {
        vreg_sum_scale = Zmm(idx_compute_vreg_start_++);
        compute_vreg_prev_dst_shift_ = compute_vregs_per_iter_++;
    }

    if (this->do_bias()) compute_vreg_bias_shift_ = compute_vregs_per_iter_++;

    if (!attr->zero_points_.has_default_values(DNNL_ARG_DST)) {
        this->do_dst_zero_points_ = true;
        vreg_dst_zero_points = Zmm(idx_compute_vreg_start_++);
    }

    isa_ = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                     : bf16_emulation_t::get_isa();

    if (dst_type == data_type::bf16 && isa_ != avx512_core_bf16) {
        idx_compute_vreg_max_ = 27;
        bf16_emu_.reset(new bf16_emulation_t(this, bf16_emu_reserv_1,
                bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                bf16_emu_reserv_5));
    }

    int max_unroll = (idx_compute_vreg_max_ - idx_compute_vreg_start_ + 1)
            / compute_vregs_per_iter_;
    max_OC_loop_unroll_ = nstl::min(max_OC_loop_unroll_, max_unroll);

    if (this->do_eltwise_ || this->do_binary_) {
#define PARAM_OFF(field) offsetof(ker_args_t, field)
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = true;
        static constexpr size_t helper_vmm_idx = 31;
        static constexpr size_t tail_size = 1;
        static constexpr bool use_exact_tail_scalar_bcast = false;
        const binary_injector::rhs_arg_static_params_t rhs_arg_static_params {
                helper_vmm_idx, eltwise_reserved_gpr_, r14, preserve_gpr,
                preserve_vmm, PARAM_OFF(post_ops_binary_rhs_arg_vec),
                memory_desc_wrapper(*dst_md), tail_size, kreg_rem_mask,
                use_exact_tail_scalar_bcast};

        const binary_injector::static_params_t binary_static_params {
                reg_binary_inj_param_, rhs_arg_static_params};
        static constexpr bool save_state = true;
        const eltwise_injector::static_params_t eltwise_static_params {
                save_state, reg_tmp_comp, eltwise_reserved_opmask_};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx512_core>>(this,
                this->post_ops_, binary_static_params, eltwise_static_params);
    }
#undef PARAM_OFF
}

template <data_type_t acc_type, data_type_t dst_type>
template <typename T>
void jit_pp_kernel_t<acc_type, dst_type>::advance_binary_postops_off(
        const T &offset) {
    const auto binary_post_op_oc_off_reg = reg_tmp_comp;
    const auto binary_post_op_oc_off_on_stack
            = ptr[rsp + reg_binary_post_op_oc_off];
    mov(binary_post_op_oc_off_reg, binary_post_op_oc_off_on_stack);
    add(binary_post_op_oc_off_reg, offset);

    Xbyak::Label end;
    cmp(binary_post_op_oc_off_reg, this->OC_);
    jl(end, T_NEAR);
    xor_(binary_post_op_oc_off_reg, binary_post_op_oc_off_reg);
    L(end);

    mov(binary_post_op_oc_off_on_stack, binary_post_op_oc_off_reg);
}

template <data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<acc_type, dst_type>::zero_binary_postops_off() {
    mov(EVEX_compress_addr(rsp, reg_binary_post_op_oc_off), 0);
}

template <data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<acc_type, dst_type>::apply_postops(
        const bool apply_mask, const int vmm_idx) {
#define PARAM_OFF(x) offsetof(ker_args_t, x)
    if (this->do_eltwise_ || this->do_binary_) {
        if (this->do_binary_) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
            const auto oc_off_oprnd = reg_tmp_comp;
            rhs_arg_params.vmm_idx_to_oc_off_oprnd.emplace(
                    vmm_idx, oc_off_oprnd);
            rhs_arg_params.vmm_idx_to_out_off_oprnd.emplace(
                    vmm_idx, oc_off_oprnd);
            if (apply_mask) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);

            mov(oc_off_oprnd, ptr[rsp + reg_binary_post_op_oc_off]);

            postops_injector_->compute_vector(vmm_idx, rhs_arg_params);
        } else
            postops_injector_->compute_vector(vmm_idx);
    }
#undef PARAM_OFF
}

template <data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<acc_type, dst_type>::compute_oc_channel_blk() {
    // Load accumulated value, convert to float, apply bias (if any), scaling,
    // and eltwise (if any); then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];
        if (dst_type == data_type::bf16 && isa_ != avx512_core_bf16)
            bf16_emu_->init_vcvtneps2bf16();

        if (this->do_scale_ && this->scale_idx_mult_ == 1) {
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_msk_ = vreg_scale;
            if (apply_mask) vreg_scale_msk_ = vreg_scale_msk_ | kreg_rem_mask;
            vmovups(vreg_scale_msk_, scale_addr);
        }

        if (this->do_binary_) {
            if (offset) advance_binary_postops_off(vlen);
            if (apply_mask) kmovq(opmask_binary, kreg_rem_mask);
        }

        const int dst_idx = vreg_dst_idx(idx);
        auto vreg_dst_ = Zmm(dst_idx);
        auto vreg_dst_msk_ = apply_mask ? vreg_dst_ | kreg_rem_mask : vreg_dst_;

        switch (acc_type) {
            case data_type::s32: vcvtdq2ps(vreg_dst_msk_, acc_addr); break;
            case data_type::f32: vmovups(vreg_dst_msk_, acc_addr); break;
        }

        if (this->do_bias()) {
            auto bias_addr
                    = ptr[reg_bias + offset * this->bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            auto vreg_bias_msk_
                    = apply_mask ? vreg_bias_ | kreg_rem_mask : vreg_bias_;

            switch (this->bias_data_type_) {
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
            if (utils::one_of(this->bias_data_type_, data_type::u8,
                        data_type::s8, data_type::s32))
                vcvtdq2ps(vreg_bias_, vreg_bias_);
            vaddps(vreg_dst_, vreg_dst_, vreg_bias_);
        }

        if (this->do_scale_) vmulps(vreg_dst_, vreg_dst_, vreg_scale);

        auto dst_addr = ptr[reg_dst + offset * sizeof(dst_data_t)];
        if (this->do_sum_) {
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

        apply_postops(apply_mask, dst_idx);

        if (this->do_dst_zero_points_)
            vaddps(vreg_dst_, vreg_dst_, vreg_dst_zero_points);

        if (utils::one_of(
                    dst_type, data_type::u8, data_type::s8, data_type::s32)) {
            saturate_f32(
                    vreg_dst_, vreg_zero, vreg_saturation_ubound, dst_type);
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
        if (this->do_scale_ && this->scale_idx_mult_ == 1)
            add(reg_scales, offset * sizeof(float));
        if (this->do_bias()) add(reg_bias, offset * this->bias_data_type_size_);
        if (this->do_binary_) advance_binary_postops_off(vlen);
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](const Reg64 &offset) {
        lea(reg_dst, ptr[reg_dst + offset * sizeof(dst_data_t)]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (this->do_scale_ && this->scale_idx_mult_ == 1)
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        if (this->do_bias())
            lea(reg_bias, ptr[reg_bias + offset * this->bias_data_type_size_]);
        if (this->do_binary_) advance_binary_postops_off(offset);
    };

    // incase of non-trivial dst_mb_strides, fixup the reg_dst and reg_acc
    auto maybe_advance_mb_stride = [&]() {
        if (!this->has_trivial_mb_stride()) {
            lea(reg_dst, ptr[reg_dst + reg_dst_mb_stride * sizeof(dst_data_t)]);
            lea(reg_acc, ptr[reg_acc + reg_acc_mb_stride * sizeof(acc_data_t)]);
        }
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        neg(reg_oc);
        if (this->do_bias())
            lea(reg_bias, ptr[reg_bias + reg_oc * this->bias_data_type_size_]);
        if (this->do_scale_ && this->scale_idx_mult_ == 1)
            lea(reg_scales, ptr[reg_scales + reg_oc * sizeof(float)]);
        if (this->do_binary_) zero_binary_postops_off();
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
        maybe_advance_mb_stride();
        process_runtime_oc();
        rewind_ptrs();
    }
    L(l_prologue_end);

    // Main loop
    Label l_main_loop_end;
    cmp(reg_len, reg_oc);
    jle(l_main_loop_end, T_NEAR);
    if (this->runtime_oc()) {
        Label l_main_loop;
        L(l_main_loop);
        {
            mov(reg_tmp, reg_oc);

            process_runtime_oc();
            rewind_ptrs();

            sub(reg_len, reg_oc);
            maybe_advance_mb_stride();
            cmp(reg_len, reg_oc);
            jge(l_main_loop, T_NEAR);
        }
    } else {
        Label l_main_loop;
        L(l_main_loop);
        {
            size_t OC_loop, OC_tail;
            if (this->OC_ < max_OC_loop_unroll_ * vlen) {
                // Fully unroll small loops
                OC_loop = 0;
                OC_tail = this->OC_;
            } else {
                OC_loop = vlen * default_OC_loop_unroll_;
                OC_tail = this->OC_ % OC_loop;
            }

            assert(!!OC_loop || !!OC_tail);

            if (OC_tail % vlen) {
                int vlen_tail = OC_tail % vlen;
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask, reg_tmp);
            }

            if (OC_loop) {
                mov(reg_tmp, utils::rnd_dn(this->OC_, OC_loop));
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
            maybe_advance_mb_stride();
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
void jit_pp_kernel_t<acc_type, dst_type>::compute_mb_blk() {
    auto compute = [&](size_t mb_step, bool apply_mask) {
        auto zmm_bias = vreg_bias(0);
        auto zmm_bias_msk = apply_mask ? zmm_bias | kreg_rem_mask : zmm_bias;
        auto zmm_dst = vreg_dst(0);
        auto zmm_dst_msk = apply_mask ? zmm_dst | kreg_rem_mask : zmm_dst;

        switch (acc_type) {
            case data_type::s32: vcvtdq2ps(zmm_dst_msk, ptr[reg_acc]); break;
            case data_type::f32: vmovups(zmm_dst_msk, ptr[reg_acc]); break;
            default: assert(!"unimplemented");
        }

        vaddps(zmm_dst, zmm_dst, zmm_bias_msk);

        switch (dst_type) {
            case data_type::f32: break;
            case data_type::u8:
            case data_type::s8:
            case data_type::s32:
                saturate_f32(
                        zmm_dst, vreg_zero, vreg_saturation_ubound, dst_type);
                vcvtps2dq(zmm_dst, zmm_dst);
                break;
            case data_type::bf16:
                if (isa_ == avx512_core_bf16)
                    vcvtneps2bf16(Ymm(zmm_dst.getIdx()), zmm_dst);
                else
                    bf16_emu_->vcvtneps2bf16(Ymm(zmm_dst.getIdx()), zmm_dst);
                break;
            default: assert(!"unimplemented");
        }

        switch (dst_type) {
            case data_type::s8: vpmovsdb(ptr[reg_dst], zmm_dst_msk); break;
            case data_type::u8: vpmovusdb(ptr[reg_dst], zmm_dst_msk); break;
            case data_type::f32:
            case data_type::s32: vmovups(ptr[reg_dst], zmm_dst_msk); break;
            case data_type::bf16:
                vmovdqu16(ptr[reg_dst],
                        apply_mask ? Ymm(zmm_dst.getIdx()) | kreg_rem_mask
                                   : Ymm(zmm_dst.getIdx()));
                break;
            default: assert(!"unimplemented");
        }
    };

    Label mb_main_loop, end_main_loop;

    bool expl_broadcast = this->OC_ == 1
            && utils::one_of(
                    this->bias_data_type_, data_type::s32, data_type::f32);
    size_t mb_step = vlen / this->OC_;
    size_t mb_tail = this->MB_ % mb_step;
    size_t mb_oc_blk = mb_step * this->OC_;

    auto zmm_bias = vreg_bias(0);

    if (dst_type == data_type::bf16 && isa_ != avx512_core_bf16)
        bf16_emu_->init_vcvtneps2bf16();

    if (expl_broadcast) {
        // when OC == 1 bias can be loaded directly into simd
        switch (this->bias_data_type_) {
            case data_type::s32: vpbroadcastd(zmm_bias, ptr[reg_bias]); break;
            case data_type::f32: vbroadcastss(zmm_bias, ptr[reg_bias]); break;
            // TODO: enable broadcast for other data types
            default: assert(!"unimplemented");
        }
    } else {
        // prepare bias data for simd computation
        mov(reg_tmp, (1 << this->OC_) - 1);
        kmovq(kreg_rem_mask, reg_tmp);
        auto zmm_bias_msk = zmm_bias | kreg_rem_mask;

        switch (this->bias_data_type_) {
            case data_type::s8: vpmovsxbd(zmm_bias_msk, ptr[reg_bias]); break;
            case data_type::u8: vpmovzxbd(zmm_bias_msk, ptr[reg_bias]); break;
            case data_type::s32:
            case data_type::f32: vmovups(zmm_bias_msk, ptr[reg_bias]); break;
            case data_type::bf16:
                vpmovzxwd(zmm_bias_msk, ptr[reg_bias]);
                vpslld(zmm_bias_msk, zmm_bias_msk, 0x10);
                break;
            default: assert(!"unimplemented");
        }

        // write repeated MB*OC entries into stack
        sub(rsp, mb_oc_blk * sizeof(uint32_t));
        for (size_t i = 0; i < mb_step; ++i) {
            vmovups(ptr[rsp + i * this->OC_ * sizeof(uint32_t)], zmm_bias_msk);
        }

        // load into simd
        mov(reg_tmp, (1 << mb_oc_blk) - 1);
        kmovq(kreg_rem_mask, reg_tmp);
        vmovups(zmm_bias | kreg_rem_mask, ptr[rsp]);
    }
    if (utils::one_of(this->bias_data_type_, data_type::u8, data_type::s8,
                data_type::s32))
        vcvtdq2ps(zmm_bias, zmm_bias);

    L(mb_main_loop);
    {
        cmp(reg_len, mb_oc_blk);
        jl(end_main_loop, T_NEAR);

        compute(mb_step, !expl_broadcast);
        add(reg_dst, mb_oc_blk * sizeof(dst_data_t));
        add(reg_acc, mb_oc_blk * sizeof(acc_data_t));
        sub(reg_len, mb_oc_blk);
        jmp(mb_main_loop, T_NEAR);
    }
    L(end_main_loop);

    if (mb_tail > 0) {
        Label mb_tail_loop, end_tail_loop;

        mov(reg_tmp, (1 << (mb_tail * this->OC_)) - 1);
        kmovq(kreg_rem_mask, reg_tmp);

        L(mb_tail_loop);
        {
            cmp(reg_len, 0);
            jle(end_tail_loop, T_NEAR);
            compute(mb_tail, true);
            sub(reg_len, mb_tail * this->OC_);
            jmp(mb_tail_loop, T_NEAR);
        }
        L(end_tail_loop);
    }

    if (!expl_broadcast) add(rsp, mb_oc_blk * sizeof(uint32_t));
}

template <data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<acc_type, dst_type>::generate() {
    preamble();

#ifdef _WIN32
    // binary postops injector needs params held (in case of WIN32)
    // in rcx register that is also used as a temp reg, so the pointer to
    // params needs to be stored in extra reg
    if (this->do_binary_) mov(reg_binary_inj_param_, param1);
#endif

#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    if (this->do_scale_) mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    if (this->do_dst_zero_points_) {
        // use reg_oc as a temporary one (alas, reg_tmp = reg_param on Windows)
        mov(reg_oc, ptr[reg_param + PARAM_OFF(dst_zero_points)]);
        vbroadcastss(vreg_dst_zero_points, ptr[reg_oc]);
    }
    if (this->runtime_oc())
        mov(reg_oc, ptr[reg_param + PARAM_OFF(oc)]);
    else
        mov(reg_oc, this->OC_);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    if (this->do_binary_) {
        // zero initialize binary post_ops offset accumulator (store on stack)
        sub(rsp, stack_space_needed);
        mov(ptr[rsp + reg_binary_post_op_oc_off], reg_oc_offset);
    }
    if (this->do_scale_ && this->scale_idx_mult_ == 0)
        vbroadcastss(vreg_scale, dword[reg_scales]);
    if (!this->has_trivial_mb_stride()) {
        mov(reg_dst_mb_stride, ptr[reg_param + PARAM_OFF(dst_mb_stride)]);
        sub(reg_dst_mb_stride, reg_oc);
        // if dst and acc point to same address (in-place), then strides must be
        // similar, else assume acc buffer is dense.
        xor_(reg_acc_mb_stride, reg_acc_mb_stride);
        cmp(reg_dst, reg_acc);
        cmove(reg_acc_mb_stride, reg_dst_mb_stride);
    }
#undef PARAM_OFF

    if (this->do_sum_) {
        mov(reg_tmp, float2int(this->sum_scale_));
        auto xreg_sum_scale = Xmm(vreg_sum_scale.getIdx());
        vmovq(xreg_sum_scale, reg_tmp);
        vbroadcastss(vreg_sum_scale, xreg_sum_scale);
    }

    init_saturate_f32(vreg_zero, vreg_saturation_ubound, reg_tmp_comp,
            data_type::f32, dst_type);

    // at least 2 blocks of mb within vlen
    bool dim_restrict = !this->runtime_oc() && !this->runtime_mb()
            && (this->OC_ <= vlen / 2) && (this->MB_ >= vlen);
    bool supported_postops = this->do_scale_ || this->do_eltwise_
            || this->do_binary_ || this->do_sum_ || this->do_dst_zero_points_;

    if (this->do_bias() && !supported_postops && dim_restrict
            && this->has_trivial_mb_stride()) {
        this->mb_blk_kernel_ = true;
        compute_mb_blk();
    } else {
        compute_oc_channel_blk();
    }

    if (this->do_binary_) add(rsp, stack_space_needed);
    postamble();

    if (this->do_eltwise_) postops_injector_->prepare_table();
}

template <data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<acc_type, dst_type>::operator()(dst_data_t *dst,
        const acc_data_t *acc, const char *bias, const float *scales,
        size_t start, size_t end, size_t runtime_oc, dim_t dst_mb_stride,
        const float *dst_zero_points, const void *post_ops_binary_rhs_arg_vec,
        const void *dst_orig, const exec_ctx_t & /* ctx */,
        const memory_desc_t & /* dst_md */) const {

    if (end <= start) return;

    const size_t OC = this->runtime_oc() ? runtime_oc : this->OC_;

    ker_args_t args;
    size_t oc_offset = start % OC;
    if (this->has_trivial_mb_stride()) {
        args.dst = dst + start;
        args.acc = acc + start;
    } else {
        const dim_t offt = (start / OC) * dst_mb_stride + oc_offset;
        args.dst = dst + offt;
        // if dst and acc point to same address (inplace), then strides
        // must be similar, else assume acc buffer is dense.
        if (dst == (dst_data_t *)acc)
            args.acc = acc + offt;
        else
            args.acc = acc + start;
    }
    args.bias = bias + oc_offset * this->bias_data_type_size_;
    args.scales = scales + this->scale_idx_mult_ * oc_offset;
    args.dst_zero_points = dst_zero_points;
    args.oc = OC;
    args.len = end - start;
    args.oc_offset = oc_offset;
    args.dst_mb_stride = dst_mb_stride;

    args.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;
    args.dst_orig = dst_orig;
    jit_generator::operator()(&args);
}

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type> *jit_pp_kernel_create(size_t OC, size_t MB,
        dim_t dst_mb_stride, const primitive_attr_t *attr, data_type_t bias_dt,
        const memory_desc_t *dst_md, bool skip_sum) {
    if (!mayiuse(avx512_core)) return nullptr;
    return new jit_pp_kernel_t<acc_type, dst_type>(
            OC, MB, dst_mb_stride, attr, bias_dt, dst_md, skip_sum);
}

#define INST(acc_type, dst_type) \
    template pp_kernel_t<acc_type, dst_type> * \
    jit_pp_kernel_create<acc_type, dst_type>(size_t OC, size_t MB, \
            dim_t dst_mb_stride, const primitive_attr_t *attr, \
            data_type_t bias_dt, const memory_desc_t *dst_md, bool skip_sum);

using namespace data_type;
INST(f32, f32);
INST(s32, f32);
INST(s32, s32);
INST(s32, s8);
INST(s32, u8);
INST(f32, bf16);

#undef INST

} // namespace inner_product_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
