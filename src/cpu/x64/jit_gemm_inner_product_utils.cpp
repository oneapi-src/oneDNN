/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#include "cpu/x64/jit_avx512_core_bf16cvt.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_uni_depthwise_injector.hpp"

#include "cpu/x64/jit_gemm_inner_product_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace inner_product_utils {

using namespace dnnl::impl::cpu::inner_product_utils;
using namespace Xbyak;

template <cpu_isa_t isa, data_type_t acc_type, data_type_t dst_type>
struct jit_pp_kernel_t : public pp_kernel_t<acc_type, dst_type>,
                         public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(inner_product_utils::jit_pp_kernel_t);

    jit_pp_kernel_t(size_t OC, size_t MB, const primitive_attr_t *attr,
            data_type_t bias_dt, bool skip_sum);
    ~jit_pp_kernel_t() {
        for (auto inj : jit_eltwise_injectors_)
            delete inj;
        jit_eltwise_injectors_.clear();
        for (auto inj : jit_depthwise_injectors_)
            delete inj;
        jit_depthwise_injectors_.clear();
    }

    using acc_data_t = typename prec_traits<acc_type>::type;
    using dst_data_t = typename prec_traits<dst_type>::type;

    void operator()(dst_data_t *dst, const acc_data_t *acc, const char *bias,
            const float *scales, size_t start, size_t end, size_t runtime_oc,
            const float *dst_zero_points) const override;

    status_t create_kernel() override { return jit_generator::create_kernel(); }

private:
    void generate() override;
    void compute_oc_channel_blk();
    void compute_mb_blk(); // vectorize across minibatch

    struct ker_args_t {
        dst_data_t *dst;
        const acc_data_t *acc;
        const char *bias;
        const float *scales;
        size_t oc;
        size_t len;
        size_t oc_offset;
    };

    enum { default_OC_loop_unroll_ = 4 };

    nstl::vector<jit_uni_eltwise_injector_f32<isa == avx512_core_bf16 ? avx512_common : isa> *> jit_eltwise_injectors_;
    nstl::vector<jit_uni_depthwise_injector_f32<isa == avx512_core_bf16 ? avx512_common : isa> *> jit_depthwise_injectors_;

    std::unique_ptr<bf16_emulation_t> bf16_emu_;

    using Vmm = typename cpu_isa_traits<isa == avx512_core_bf16 ? avx512_common : isa>::Vmm;
    const size_t vlen = cpu_isa_traits<isa == avx512_core_bf16 ? avx512_common : isa>::vlen / sizeof(float);

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

    // Will be assigned in constructor
    Vmm vreg_zero, vreg_scale;

    Xbyak::Zmm bf16_emu_reserv_1 = Xbyak::Zmm(28);
    Xbyak::Zmm bf16_emu_reserv_2 = Xbyak::Zmm(29);
    Xbyak::Zmm bf16_emu_reserv_3 = Xbyak::Zmm(30);
    Xbyak::Reg64 bf16_emu_reserv_4 = r12;
    Xbyak::Zmm bf16_emu_reserv_5 = Xbyak::Zmm(31);

    //  sse42/avx2
    Xbyak::Reg64 reg_ptr_maskmovdqu_dst = rdi; // sse41: store destination - must be rdi
    Xbyak::Reg8 reg_tmp_8 = r11b;
    Xbyak::Reg32 reg_tmp_32 = r11d;
    Xbyak::Reg64 reg_tmp_64 = r11;
    Xbyak::Label l_table;
    Xbyak::Reg64 reg_table = r12;
    Xbyak::Reg64 reg_shift_table = r13;
    Vmm vreg_mask = Vmm(0); //  sse41: mask for blendvps must be in xmm0
    Vmm vreg_store_mask = Vmm(1);

    //  post_ops
    Xbyak::Reg64 eltwise_reserved_1_ = r11;
    Xbyak::Opmask eltwise_reserved_2_ = k2;
    Xbyak::Reg64 reg_d_weights = r14;
    Xbyak::Reg64 reg_d_bias = r15; // todo: reg_tmp_comp = r15
    Vmm vreg_d_weights, vreg_d_bias;

    size_t bias_data_type_size_;
    int max_OC_loop_unroll_ = 13;
    int idx_compute_vreg_start_ = 0;
    int idx_compute_vreg_max_ = 31;
    int compute_vregs_per_iter_ = 1;

    int idx_vreg_dst(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 0;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }
    int idx_vreg_bias(int iter) {
        int idx = idx_compute_vreg_start_ + iter * compute_vregs_per_iter_ + 1;
        assert(idx <= idx_compute_vreg_max_);
        return idx;
    }

    Vmm vreg_dst(int iter) { return Vmm(idx_vreg_dst(iter)); };
    Xbyak::Zmm zmm_dst(int iter) { return Xbyak::Zmm(idx_vreg_dst(iter)); };
    Xbyak::Ymm ymm_dst(int iter) { return Xbyak::Ymm(idx_vreg_dst(iter)); };
    Xbyak::Xmm xmm_dst(int iter) { return Xbyak::Xmm(idx_vreg_dst(iter)); };
    Vmm vreg_bias(int iter) { return Vmm(idx_vreg_bias(iter)); };
};

template <cpu_isa_t isa, data_type_t acc_type, data_type_t dst_type>
jit_pp_kernel_t<isa, acc_type, dst_type>::jit_pp_kernel_t(size_t OC, size_t MB,
        const primitive_attr_t *attr, data_type_t bias_dt, bool skip_sum)
    : pp_kernel_t<acc_type, dst_type>(OC, MB, attr, bias_dt, skip_sum)
    , bf16_emu_(nullptr)
    , bias_data_type_size_(0)
    , max_OC_loop_unroll_(utils::one_of(isa, avx512_core_bf16, avx512_common) ? 13 : 6)
    , idx_compute_vreg_start_(0)
    , idx_compute_vreg_max_(utils::one_of(isa, avx512_core_bf16, avx512_common) ? 31 : 15)
    , compute_vregs_per_iter_(1) {

    if (utils::one_of(isa, avx2, sse41)) {
        idx_compute_vreg_start_ += 2;   //  Vmm(0), Vmm(1) - for masks
    }

    if (this->do_scale_) vreg_scale = Zmm(idx_compute_vreg_start_++);

    bool only_eltwise = true;
    for (int i = 0; i < this->post_ops_.len(); i++) {
        auto &post_op = this->post_ops_.entry_[i];
        if (post_op.is_eltwise()) {
            jit_eltwise_injectors_.push_back(new jit_uni_eltwise_injector_f32<isa == avx512_core_bf16 ? avx512_common : isa>(
                    this, post_op.eltwise, true, eltwise_reserved_1_, eltwise_reserved_2_));
        } else if (post_op.is_depthwise()) {
            only_eltwise = false;
            jit_depthwise_injectors_.push_back(new jit_uni_depthwise_injector_f32<isa == avx512_core_bf16 ? avx512_common : isa>(
                    this, post_op.depthwise.alg, eltwise_reserved_2_));
        } else {
            only_eltwise = false;
        }
    }
    if (this->post_ops_.len() > 0 && !only_eltwise) {
        vreg_d_weights = Vmm(idx_compute_vreg_max_--);
        vreg_d_bias = Vmm(idx_compute_vreg_max_--);
    }
    if (dst_type == data_type::u8 || utils::one_of(isa, avx2, sse41))
        vreg_zero = Vmm(idx_compute_vreg_start_++);

    if (this->do_bias_) {
        compute_vregs_per_iter_++;
        bias_data_type_size_ = types::data_type_size(this->bias_data_type_);
    }

    if (dst_type == data_type::bf16 && isa != avx512_core_bf16) {
        idx_compute_vreg_max_ = 27;
        bf16_emu_.reset(new bf16_emulation_t(this, bf16_emu_reserv_1,
                bf16_emu_reserv_2, bf16_emu_reserv_3, bf16_emu_reserv_4,
                bf16_emu_reserv_5));
    }

    int max_unroll = (idx_compute_vreg_max_ - idx_compute_vreg_start_ + 1)
            / compute_vregs_per_iter_;
    max_OC_loop_unroll_ = nstl::min(max_OC_loop_unroll_, max_unroll);
}

template <cpu_isa_t isa, data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<isa, acc_type, dst_type>::compute_oc_channel_blk() {
    bool do_post_ops = this->post_ops_.len() > 0;

    auto apply_post_ops = [&](size_t offset, int idx) {
        int eltwise_inj_idx = 0;
        int depthwise_inj_idx = 0;
        for (int i = 0; i < this->post_ops_.len(); i++) {
            auto& post_op = this->post_ops_.entry_[i];
            if (post_op.is_eltwise()) {
                jit_eltwise_injectors_[eltwise_inj_idx]->compute_vector(vreg_dst(idx).getIdx());
                eltwise_inj_idx++;
            } else if (post_op.is_depthwise()) {
                mov(reg_d_weights, reinterpret_cast<size_t>(post_op.depthwise.weights_data + offset));
                mov(reg_d_bias, reinterpret_cast<size_t>(post_op.depthwise.biases_data + offset));
                lea(reg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                lea(reg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                jit_depthwise_injectors_[depthwise_inj_idx]->compute_vector_range(vreg_dst(idx).getIdx(), vreg_dst(idx).getIdx() + 1, reg_d_weights, reg_d_bias);
                depthwise_inj_idx++;
            } else if (post_op.is_quantization()) {
                bool do_dequantization = post_op.quantization.alg == alg_kind::quantization_quantize_dequantize;
                bool do_rounding = do_dequantization || dst_type == dnnl_f32 || i != this->post_ops_.len() - 1;

                if (post_op.quantization.crop_low_data->count_ != 1) {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data->shifts_ + offset));
                    uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.crop_low_data->shifts_));
                    uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights]);
                }

                if (post_op.quantization.crop_high_data->count_ != 1) {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data->shifts_ + offset));
                    uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.crop_high_data->shifts_));
                    uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias]);
                }

                uni_vmaxps(vreg_dst(idx), vreg_dst(idx), vreg_d_weights);
                uni_vminps(vreg_dst(idx), vreg_dst(idx), vreg_d_bias);

                if (post_op.quantization.input_scale_data->count_ != 1) {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data->scales_ + offset));
                    uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.input_scale_data->scales_));
                    uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights]);
                }

                if (post_op.quantization.input_shift_data->count_ != 1) {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data->shifts_ + offset));
                    uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                } else {
                    mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.input_shift_data->shifts_));
                    uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias]);
                }

                uni_vfmadd213ps(vreg_dst(idx), vreg_d_weights, vreg_d_bias);

                if (do_rounding)
                    uni_vroundps(vreg_dst(idx), vreg_dst(idx), 0);

                if (do_dequantization) {
                    if (post_op.quantization.output_scale_data->count_ != 1) {
                        mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data->scales_ + offset));
                        uni_vmovups(vreg_d_weights, ptr[reg_d_weights + reg_oc_offset * sizeof(float)]);
                    } else {
                        mov(reg_d_weights, reinterpret_cast<size_t>(post_op.quantization.output_scale_data->scales_));
                        uni_vbroadcastss(vreg_d_weights, ptr[reg_d_weights]);
                    }

                    if (post_op.quantization.output_shift_data->count_ != 1) {
                        mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data->shifts_ + offset));
                        uni_vmovups(vreg_d_bias, ptr[reg_d_bias + reg_oc_offset * sizeof(float)]);
                    } else {
                        mov(reg_d_bias, reinterpret_cast<size_t>(post_op.quantization.output_shift_data->shifts_));
                        uni_vbroadcastss(vreg_d_bias, ptr[reg_d_bias]);
                    }

                    uni_vfmadd213ps(vreg_dst(idx), vreg_d_weights, vreg_d_bias);
                }
            }
        }
    };

    // Load accumulated value, convert to float, apply bias (if any), scaling,
    // and eltwise (if any); then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];
        if (dst_type == data_type::bf16 && isa != avx512_core_bf16)
            bf16_emu_->init_vcvtneps2bf16();

        if (this->do_scale_ && this->scale_idx_mult_ == 1) {
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_msk_ = vreg_scale;
            if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                if (apply_mask)
                    vreg_scale_msk_ = vreg_scale_msk_ | kreg_rem_mask;
                uni_vmovups(vreg_scale_msk_, scale_addr);
            } else {
                if (apply_mask)
                    if (isa != sse41) {
                        uni_vblendvps(vreg_scale, vreg_zero, scale_addr, vreg_mask);
                    } else {
                        uni_vmovups(vreg_scale, vreg_zero);
                        uni_vblendvps(vreg_scale, vreg_scale, scale_addr, vreg_mask);
                    }
                else
                    uni_vmovups(vreg_scale, scale_addr);
            }
        }

        auto vreg_dst_ = vreg_dst(idx);
        if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
            if (apply_mask)
                vreg_dst_ = vreg_dst_ | kreg_rem_mask;

            switch (acc_type) {
                case data_type::s32: uni_vcvtdq2ps(vreg_dst_, acc_addr); break;
                case data_type::f32: uni_vmovups(vreg_dst_, acc_addr); break;
            }
        } else {
            if (apply_mask) {
                if (isa != sse41) {
                    uni_vblendvps(vreg_dst_, vreg_zero, acc_addr, vreg_mask);
                } else {
                    uni_vmovups(vreg_dst_, acc_addr);
                }
                if (acc_type == data_type::s32)
                    uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
            } else {
                if (isa == sse41) {
                    uni_vmovups(vreg_dst_, acc_addr);
                    if (acc_type == data_type::s32) uni_vcvtdq2ps(vreg_dst_, vreg_dst_);
                } else {
                    switch (acc_type) {
                        case data_type::s32: uni_vcvtdq2ps(vreg_dst_, acc_addr); break;
                        case data_type::f32: uni_vmovups(vreg_dst_, acc_addr); break;
                    }
                }
            }
        }

        if (this->do_bias_) {
            auto bias_addr = ptr[reg_bias + offset * this->bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            if (utils::one_of(isa, avx512_core_bf16, avx512_common) && apply_mask)
                vreg_bias_ = vreg_bias_ | kreg_rem_mask;

            switch (this->bias_data_type_) {
                case data_type::s8: uni_vpmovsxbd(vreg_bias_, bias_addr); break;
                case data_type::u8: uni_vpmovzxbd(vreg_bias_, bias_addr); break;
                case data_type::s32:
                case data_type::f32: uni_vmovups(vreg_bias_, bias_addr); break;
                case data_type::bf16:
                    vpmovzxwd(vreg_bias_, bias_addr);
                    vpslld(vreg_bias_, vreg_bias_, 0x10);
                    break;
                default: assert(!"unimplemented");
            }
            if (utils::one_of(this->bias_data_type_, data_type::u8,
                        data_type::s8, data_type::s32))
                uni_vcvtdq2ps(vreg_bias_, vreg_bias_);
            uni_vaddps(vreg_dst_, vreg_dst_, vreg_bias_);
        }

        if (this->do_scale_) uni_vmulps(vreg_dst_, vreg_dst_, vreg_scale);

        apply_post_ops(offset, idx);

        if (dst_type == data_type::u8)
            uni_vmaxps(vreg_dst(idx), vreg_dst(idx), vreg_zero);

        if (utils::one_of(dst_type, data_type::s8, data_type::u8, data_type::s32)) {
            if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                auto rmode_control = T_rn_sae;
                vcvtps2dq(vreg_dst(idx) | rmode_control, vreg_dst(idx));
            } else {
                uni_vcvtps2dq(vreg_dst(idx), vreg_dst(idx));
            }
        } else if (dst_type == data_type::bf16) {
            if (isa == avx512_core_bf16)
                vcvtneps2bf16(ymm_dst(idx), vreg_dst(idx));
            else
                bf16_emu_->vcvtneps2bf16(ymm_dst(idx), zmm_dst(idx));
        }

        auto dst_addr = ptr[reg_dst + offset * sizeof(dst_data_t)];
        switch (dst_type) {
            case data_type::s8:
                if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                    vpmovsdb(dst_addr, vreg_dst_);
                } else {
                    uni_vpackssdw(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (isa != sse41)
                        vpermq(ymm_dst(idx), ymm_dst(idx), 0x08);
                    uni_vpacksswb(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (apply_mask) {
                        lea(reg_ptr_maskmovdqu_dst, dst_addr);
                        maskmovdqu(vreg_dst_, vreg_store_mask);
                    } else {
                        if (isa != sse41) {
                            vmovq(dst_addr, xmm_dst(idx));
                        } else {
                            movd(dst_addr, xmm_dst(idx));
                        }
                    }
                }
                break;
            case data_type::u8:
                if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                    vpmovusdb(dst_addr, vreg_dst_);
                } else {
                    uni_vpackusdw(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (isa != sse41)
                        vpermq(ymm_dst(idx), ymm_dst(idx), 0x08);
                    uni_vpackuswb(vreg_dst_, vreg_dst_, vreg_dst_);
                    if (apply_mask) {
                        lea(reg_ptr_maskmovdqu_dst, dst_addr);
                        maskmovdqu(vreg_dst_, vreg_store_mask);
                    } else {
                        if (isa != sse41) {
                            vmovq(dst_addr, xmm_dst(idx));
                        } else {
                            movd(dst_addr, xmm_dst(idx));
                        }
                    }
                }
                break;
            case data_type::f32:
            case data_type::s32:
                if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                    uni_vmovups(dst_addr, vreg_dst_);
                } else {
                    if (apply_mask) {
                        if (isa != sse41) {
                            vmaskmovps(dst_addr, vreg_mask, vreg_dst_);
                        } else {
                            lea(reg_ptr_maskmovdqu_dst, dst_addr);
                            maskmovdqu(vreg_dst_, vreg_mask);
                        }
                    } else {
                        uni_vmovups(dst_addr, vreg_dst_);
                    }
                }
                break;
            case data_type::bf16:
                vmovdqu16(dst_addr,
                          apply_mask
                          ? ymm_dst(idx) | kreg_rem_mask
                          : ymm_dst(idx));
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
        if (this->do_bias())
            add(reg_bias, offset * this->bias_data_type_size_);
        if (do_post_ops)
            add(reg_oc_offset, offset);
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](Reg64 offset) {
        lea(reg_dst, ptr[reg_dst + offset * sizeof(dst_data_t)]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (this->do_scale_ && this->scale_idx_mult_ == 1)
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        if (this->do_bias())
            lea(reg_bias, ptr[reg_bias + offset * this->bias_data_type_size_]);
        if (do_post_ops)
            lea(reg_oc_offset, ptr[reg_oc_offset + offset]);
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        neg(reg_oc);
        if (do_post_ops)
            lea(reg_oc_offset, ptr[reg_oc_offset + reg_oc]);
        if (this->do_bias_)
            lea(reg_bias, ptr[reg_bias + reg_oc * this->bias_data_type_size_]);
        if (this->do_scale_ && this->scale_idx_mult_ == 1)
            lea(reg_scales, ptr[reg_scales + reg_oc * sizeof(float)]);
        neg(reg_oc);
    };

    // Process one row of reg_tmp elements
    auto process_runtime_oc = [&]() {
        Label l_loop, l_loop_tail, l_loop_end;
        cmp(reg_tmp, vlen);
        jl(l_loop_tail, T_NEAR); // Skips for reg_tmp == 16 too (?)

        L(l_loop);
        {
            compute(0, 0, false);
            advance_ptrs_imm(vlen);
            sub(reg_tmp, vlen);
            cmp(reg_tmp, vlen);
            jge(l_loop, T_NEAR);
        }

        L(l_loop_tail);
        if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
            mov(reg_rem_mask, 1);
            shl(reg_rem_mask, cl); // cl == reg_tmp because reg_tmp <= vlen here
            sub(reg_rem_mask, 1);
            jz(l_loop_end, T_NEAR);

            kmovq(kreg_rem_mask, reg_rem_mask);
        } else {
            push(reg_oc);
            mov(reg_shift_table, vlen);
            sub(reg_shift_table, reg_tmp);
            uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
            if (dst_type == data_type::s8 || dst_type == data_type::u8) {
                mov(reg_shift_table, vlen * sizeof(float));
                sub(reg_shift_table, reg_tmp);
                uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
            }
            pop(reg_oc);
        }
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
    jl(l_main_loop_end, T_NEAR);
    if (this->runtime_oc()) { // todo: antonvor: do we support this case?
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
            if (utils::one_of(isa, avx512_core_bf16, avx512_common)) {
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask, reg_tmp);
            } else {
                push(reg_oc);
                mov(reg_shift_table, vlen - vlen_tail);
                uni_vmovups(vreg_mask, ptr[reg_table + reg_shift_table * sizeof(float)]);
                if (dst_type == data_type::s8 || dst_type == data_type::u8) {
                    mov(reg_shift_table, vlen * sizeof(float));
                    sub(reg_shift_table, vlen_tail);
                    uni_vmovups(vreg_store_mask, ptr[reg_table + reg_shift_table]);
                }
                pop(reg_oc);
            }
        }

        Label l_main_loop;
        L(l_main_loop);
        {
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

template <cpu_isa_t isa, data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<isa, acc_type, dst_type>::compute_mb_blk() {
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
                vcvtps2dq(zmm_dst, zmm_dst);
                break;
            case data_type::bf16:
                if (isa == avx512_core_bf16)
                    vcvtneps2bf16(Ymm(zmm_dst.getIdx()), zmm_dst);
                else
                    bf16_emu_->vcvtneps2bf16(Ymm(zmm_dst.getIdx()), Zmm(zmm_dst.getIdx()));
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

    if (dst_type == data_type::bf16 && isa != avx512_core_bf16)
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

template <cpu_isa_t isa, data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<isa, acc_type, dst_type>::generate() {
    preamble();

#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    if (this->do_scale_) mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    if (this->runtime_oc()) // todo: antonvor: do we support this case?
        mov(reg_oc, ptr[reg_param + PARAM_OFF(oc)]);
    else
        mov(reg_oc, this->OC_);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    if (this->do_scale_ && this->scale_idx_mult_ == 0)
        uni_vbroadcastss(vreg_scale, dword[reg_scales]);
#undef PARAM_OFF

    if (dst_type == data_type::u8 || utils::one_of(isa, avx2, sse41))
        uni_vpxor(vreg_zero, vreg_zero, vreg_zero);

    if (utils::one_of(isa, avx2, sse41))
        mov(reg_table, l_table);

    // at least 2 blocks of mb within vlen
    bool dim_restrict = !this->runtime_oc() && !this->runtime_mb()
            && (this->OC_ <= vlen / 2) && (this->MB_ >= vlen);
    bool supported_postops = this->do_scale_ || this->do_eltwise_
            || this->do_sum_;

    if (this->do_bias() && !supported_postops && dim_restrict) {
        this->mb_blk_kernel_ = true;
        compute_mb_blk();
    } else {
        compute_oc_channel_blk();
    }

    postamble();

    for (auto& inj : jit_eltwise_injectors_)
        inj->prepare_table();

    if (utils::one_of(isa, avx2, sse41)) {
        align(64);
        L(l_table);
        for (size_t i = 0; i < vlen; i++) dd(0xFFFFFFFF);
        for (size_t i = 0; i < vlen; i++) dd(0x00000000);
    }
}

template <cpu_isa_t isa, data_type_t acc_type, data_type_t dst_type>
void jit_pp_kernel_t<isa, acc_type, dst_type>::operator()(dst_data_t *dst,
        const acc_data_t *acc, const char *bias, const float *scales,
        size_t start, size_t end, size_t runtime_oc,
        const float *dst_zero_points) const {

    if (end <= start) return;

    const size_t OC = this->runtime_oc() ? runtime_oc : this->OC_;

    ker_args_t args;
    size_t oc_offset = start % OC;
    args.dst = dst + start;
    args.acc = acc + start;
    args.bias = bias + oc_offset * this->bias_data_type_size_;
    args.scales = scales + this->scale_idx_mult_ * oc_offset;
    args.oc = OC;
    args.len = end - start;
    args.oc_offset = oc_offset;
    jit_generator::operator()(&args);
}

template <data_type_t acc_type, data_type_t dst_type>
pp_kernel_t<acc_type, dst_type> *jit_pp_kernel_create(size_t OC, size_t MB,
        const primitive_attr_t *attr, data_type_t bias_dt, bool skip_sum) {
    if (mayiuse(avx512_common)) {
        return new jit_pp_kernel_t<avx512_common, acc_type, dst_type>(OC, MB, attr, bias_dt, skip_sum);
    } else if (mayiuse(avx2)) {
        return new jit_pp_kernel_t<avx2, acc_type, dst_type>(OC, MB, attr, bias_dt, skip_sum);
    } else if (mayiuse(sse41)) {
        return new jit_pp_kernel_t<sse41, acc_type, dst_type>(OC, MB, attr, bias_dt, skip_sum);
    }
    return nullptr;
}

#define INST(acc_type, dst_type) \
    template pp_kernel_t<acc_type, dst_type> * \
    jit_pp_kernel_create<acc_type, dst_type>(size_t OC, size_t MB, \
            const primitive_attr_t *attr, data_type_t bias_dt, bool skip_sum);

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
