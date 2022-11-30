/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "common/type_helpers.hpp"

#include "jit_uni_reduction_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace Xbyak;
#define GET_OFF(field) offsetof(jit_reduction_call_s, field)

static const bcast_set_t &get_supported_postops_bcast_strategies() {
    static const bcast_set_t supported_strategies
            = {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
                    broadcasting_strategy_t::per_oc_spatial,
                    broadcasting_strategy_t::no_broadcast};
    return supported_strategies;
}

template <cpu_isa_t isa, typename Vmm>
jit_uni_reduction_kernel_t<isa, Vmm>::jit_uni_reduction_kernel_t(
        const jit_reduction_conf_t &conf, const memory_desc_t *dst_md)
    : jit_uni_reduction_kernel_base_t(conf)
    , load_tail_size_(conf.reduce_size % simd_w_)
    , io_load_(this, isa, conf_.src_type, {false},
              io::io_tail_conf_t {simd_w_, load_tail_size_, k_tail_load_mask_,
                      vmm_tail_load_mask_.getIdx(), reg_tmp_},
              io::io_emu_bf16_conf_t {vmm_bf16_emu_1_, vmm_bf16_emu_2_,
                      vmm_bf16_emu_3_, reg_tmp_, vmm_bf16_emu_4_},
              io::io_saturation_conf_t {vmm_zero_saturation_.getIdx(),
                      vmm_saturation_ubound_.getIdx(), reg_tmp_})
    , io_store_(this, isa, conf_.dst_type, {false},
              io::io_tail_conf_t {simd_w_, store_tail_size_, k_tail_store_mask_,
                      vmm_tail_store_mask_.getIdx(), reg_tmp_},
              io::io_emu_bf16_conf_t {vmm_bf16_emu_1_, vmm_bf16_emu_2_,
                      vmm_bf16_emu_3_, reg_tmp_, vmm_bf16_emu_4_},
              io::io_saturation_conf_t {vmm_zero_saturation_.getIdx(),
                      vmm_saturation_ubound_.getIdx(), reg_tmp_}) {
    init_compute_op();
    init_compute_scalar_op();
    if (conf_.with_postops) init_post_ops_injector(dst_md);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::init_acc() {
    using namespace alg_kind;
    using namespace nstl;

    const Xmm xmm_tmp_(vmm_tmp1_.getIdx());
    float starting_val = 0;

    switch (conf_.alg) {
        case reduction_max:
            starting_val = numeric_limits<float>::lowest();
            break;
        case reduction_min: starting_val = numeric_limits<float>::max(); break;
        case reduction_mean:
        case reduction_sum: starting_val = 0.f; break;
        case reduction_mul: starting_val = 1.f; break;
        default: assert(!"unknown alg");
    }

    mov(reg_tmp_.cvt32(), float2int(starting_val));
    uni_vmovd(xmm_tmp_, reg_tmp_.cvt32());
    uni_vbroadcastss(vmm_acc_, xmm_tmp_);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::init_compute_op() {
    using namespace alg_kind;
    switch (conf_.alg) {
        case reduction_max:
            compute_op_ = [&](const Xbyak::Xmm &acc, const Xbyak::Xmm &to_acc) {
                uni_vmaxps(acc, acc, to_acc);
            };
            break;
        case reduction_min:
            compute_op_ = [&](const Xbyak::Xmm &acc, const Xbyak::Xmm &to_acc) {
                uni_vminps(acc, acc, to_acc);
            };
            break;
        case reduction_mean:
        case reduction_sum:
            compute_op_ = [&](const Xbyak::Xmm &acc, const Xbyak::Xmm &to_acc) {
                uni_vaddps(acc, acc, to_acc);
            };
            break;
        case reduction_mul:
            compute_op_ = [&](const Xbyak::Xmm &acc, const Xbyak::Xmm &to_acc) {
                uni_vmulps(acc, acc, to_acc);
            };
            break;
        default: assert(!"unsupported alg.");
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::init_compute_scalar_op() {
    using namespace alg_kind;

    switch (conf_.alg) {
        case reduction_max:
            compute_scalar_op_
                    = [&](const Xbyak::Xmm &acc, const Xbyak::Xmm &to_acc) {
                          maxss(acc, to_acc);
                      };
            break;
        case reduction_min:
            compute_scalar_op_
                    = [&](const Xbyak::Xmm &acc, const Xbyak::Xmm &to_acc) {
                          minss(acc, to_acc);
                      };
            break;
        case reduction_mean:
        case reduction_sum:
            compute_scalar_op_
                    = [&](const Xbyak::Xmm &acc, const Xbyak::Xmm &to_acc) {
                          addss(acc, to_acc);
                      };
            break;
        case reduction_mul:
            compute_scalar_op_
                    = [&](const Xbyak::Xmm &acc, const Xbyak::Xmm &to_acc) {
                          mulss(acc, to_acc);
                      };
            break;
        default: assert(!"unsupported alg.");
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::init_post_ops_injector(
        const memory_desc_t *dst_md) {
    const memory_desc_wrapper dst_d(*dst_md);

    const eltwise_injector::static_params_t esp(true /*save_state*/,
            reg_po_injector_helper_1_, elt_inj_opmask_, true /*is_fwd*/,
            false /*use_dst*/);
    const binary_injector::rhs_arg_static_params_t rhs_arg_bsp {
            static_cast<size_t>(rhs_dt_helper_vmm_.getIdx()),
            reg_po_injector_helper_1_, reg_po_injector_helper_2_,
            reg_po_injector_helper_3_, true /*preserve gpr*/,
            true /*preserve vmm*/, GET_OFF(post_ops_binary_rhs_arg_vec),
            GET_OFF(dst_orig), dst_d, store_tail_size_, k_tail_store_mask_,
            false /*use_exact_tail_scalar_bcast*/};
    const binary_injector::static_params_t bsp(
            reg_param_, get_supported_postops_bcast_strategies(), rhs_arg_bsp);

    postops_injector_ = utils::make_unique<
            injector::jit_uni_postops_injector_t<inject_isa_, Vmm>>(
            this, conf_.post_ops, bsp, esp);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::reduce_zmm_to_ymm(
        const Xmm &acc, const Xmm &tmp) {
    const Zmm zmm_acc(acc.getIdx());
    const Ymm ymm_acc(acc.getIdx());
    const Ymm ymm_to_acc(tmp.getIdx());
    vextractf64x4(ymm_to_acc, zmm_acc, 1);
    compute_op_(ymm_acc, ymm_to_acc);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::reduce_ymm_to_xmm(
        const Xmm &acc, const Xmm &tmp) {
    const Ymm ymm_acc(acc.getIdx());
    const Xmm xmm_acc(acc.getIdx());
    const Xmm xmm_to_acc(tmp.getIdx());
    vextractf128(xmm_to_acc, ymm_acc, 1);
    compute_op_(xmm_acc, xmm_to_acc);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::reduce_xmm_to_scalar(const Xmm &acc,
        const Xmm &tmp, const std::size_t number_of_values_to_reduce) {
    assert(number_of_values_to_reduce <= number_of_f32_in_xmm_);

    const Xmm xmm_acc(acc.getIdx());
    const Xmm ymm_to_acc(tmp.getIdx());

    static constexpr int number_of_f32_to_move = number_of_f32_in_xmm_ - 1;
    static constexpr uint8_t insertps_configuration[number_of_f32_to_move]
            = {0b01001110, 0b10001110, 0b11001110};

    for (std::size_t i = 0; i < number_of_values_to_reduce - 1; i++) {
        insertps(ymm_to_acc, xmm_acc, insertps_configuration[i]);
        compute_scalar_op_(xmm_acc, ymm_to_acc);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::reduce_ymm_to_scalar(
        const Xbyak::Xmm &acc, const Xbyak::Xmm &tmp1, const Xbyak::Xmm &tmp2,
        const std::size_t number_of_values_to_reduce) {
    assert(number_of_values_to_reduce <= number_of_f32_in_ymm_);

    const Ymm ymm_acc(acc.getIdx());
    const Xmm xmm_acc(acc.getIdx());
    const Xmm xmm_tmp(tmp1.getIdx());
    const Xmm xmm_acc_upper_half(tmp2.getIdx());

    if (number_of_values_to_reduce == number_of_f32_in_ymm_) {
        reduce_ymm_to_xmm(ymm_acc, xmm_tmp);
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp);
    } else if (number_of_values_to_reduce > number_of_f32_in_xmm_) {
        vextractf128(xmm_acc_upper_half, ymm_acc, 1);
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp);
        reduce_xmm_to_scalar(xmm_acc_upper_half, xmm_tmp,
                number_of_values_to_reduce - number_of_f32_in_xmm_);
        compute_scalar_op_(xmm_acc, xmm_acc_upper_half);
    } else if (number_of_values_to_reduce <= number_of_f32_in_xmm_) {
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp, number_of_values_to_reduce);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::reduce_vmm_to_scalar(
        const Xbyak::Xmm &acc, const Xbyak::Xmm &tmp1, const Xbyak::Xmm &tmp2,
        const Xbyak::Xmm &tmp3, const std::size_t number_of_values_to_reduce) {
    assert(number_of_values_to_reduce <= number_of_f32_in_zmm_);

    const Zmm zmm_acc(acc.getIdx());
    const Ymm ymm_acc(acc.getIdx());
    const Xmm xmm_acc(acc.getIdx());
    const Ymm ymm_acc_upper_half(tmp1.getIdx());
    const Xmm xmm_acc_upper_half(tmp1.getIdx());
    const Ymm ymm_tmp(tmp2.getIdx());
    const Xmm xmm_tmp1(tmp2.getIdx());
    const Xmm xmm_tmp2(tmp3.getIdx());

    if (number_of_values_to_reduce == number_of_f32_in_zmm_) {
        reduce_zmm_to_ymm(zmm_acc, ymm_tmp);
        reduce_ymm_to_xmm(ymm_acc, xmm_tmp1);
        reduce_xmm_to_scalar(xmm_acc, xmm_tmp1);
    } else if (number_of_values_to_reduce > number_of_f32_in_ymm_) {
        vextractf64x4(ymm_acc_upper_half, zmm_acc, 1);
        reduce_ymm_to_scalar(ymm_acc, xmm_tmp1, xmm_tmp2);
        reduce_ymm_to_scalar(ymm_acc_upper_half, xmm_tmp1, xmm_tmp2,
                number_of_values_to_reduce - number_of_f32_in_ymm_);
        compute_scalar_op_(xmm_acc, xmm_acc_upper_half);
    } else if (number_of_values_to_reduce <= number_of_f32_in_ymm_) {
        reduce_ymm_to_scalar(
                ymm_acc, xmm_tmp1, xmm_tmp2, number_of_values_to_reduce);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::reduce_ne_convert_xf16() {
    Label label_work_begin, label_work_tail_begin, label_work_tail_end;

    L(label_work_begin);
    {
        cmp(reg_work_, 2);
        jl(label_work_tail_begin);
        io_load_.load_two_simdw_xf16(ptr[reg_src_], vmm_tmp1_, vmm_tmp2_);

        compute_op_(vmm_acc_, vmm_tmp1_);
        compute_op_(vmm_acc_, vmm_tmp2_);

        add(reg_src_, 2 * simd_w_ * conf_.src_dt_size);

        sub(reg_work_, 2);
        jmp(label_work_begin);
    }

    L(label_work_tail_begin);
    {
        cmp(reg_work_, 0);
        je(label_work_tail_end);
        io_load_.load(ptr[reg_src_], vmm_tmp1_, false);
        compute_op_(vmm_acc_, vmm_tmp1_);

        add(reg_src_, simd_w_ * conf_.src_dt_size);

        dec(reg_work_);
        jmp(label_work_tail_begin);
    }
    L(label_work_tail_end);

    if (load_tail_size_) {
        io_load_.load(ptr[reg_src_], vmm_tmp1_, true);
        reduce_vmm_to_scalar(
                vmm_tmp1_, vmm_tmp2_, vmm_tmp3_, vmm_tmp4_, load_tail_size_);
        compute_scalar_op_(Xmm(vmm_acc_.getIdx()), Xmm(vmm_tmp1_.getIdx()));
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::reduce_base() {
    Label label_work_begin, label_work_end;

    L(label_work_begin);
    {
        cmp(reg_work_, 0);
        je(label_work_end);
        io_load_.load(ptr[reg_src_], vmm_tmp1_, false);
        compute_op_(vmm_acc_, vmm_tmp1_);

        add(reg_src_, simd_w_ * conf_.src_dt_size);

        dec(reg_work_);
        jmp(label_work_begin);
    }
    L(label_work_end);

    if (load_tail_size_) {
        io_load_.load(ptr[reg_src_], vmm_tmp1_, true);
        reduce_vmm_to_scalar(
                vmm_tmp1_, vmm_tmp2_, vmm_tmp3_, vmm_tmp4_, load_tail_size_);
        compute_scalar_op_(Xmm(vmm_acc_.getIdx()), Xmm(vmm_tmp1_.getIdx()));
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::reduce() {
    if (utils::one_of(conf_.src_type, data_type::bf16, data_type::f16)
            && isa == avx2_vnni_2)
        reduce_ne_convert_xf16();
    else
        reduce_base();
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::load_params() {
    mov(reg_src_, ptr[reg_param_ + GET_OFF(src)]);
    mov(reg_dst_, ptr[reg_param_ + GET_OFF(dst)]);
    mov(reg_work_, conf_.reduce_size / simd_w_);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::apply_sum(const int data_idx) {
    if (conf_.with_sum) {
        assert(!conf_.sum_scales.empty()
                && "No scales for sum post operation.");
        const auto sum_injector = [this, data_idx]() {
            const Vmm vmm_prev_dst(vmm_tmp1_.getIdx());
            const Vmm vmm_dst(data_idx);

            io_store_.load(ptr[reg_dst_], vmm_prev_dst, true);
            const float sum_scale = sum_scales_.front();
            if (sum_scale == 1.f)
                uni_vaddps(vmm_dst, vmm_dst, vmm_prev_dst);
            else {
                const Xmm xmm_sum_scale = Xmm(vmm_sum_scale_.getIdx());
                mov(reg_tmp1_.cvt32(), float2int(sum_scale));
                uni_vmovd(xmm_sum_scale, reg_tmp1_.cvt32());
                uni_vbroadcastss(vmm_sum_scale_, xmm_sum_scale);
                uni_vfmadd231ps(vmm_dst, vmm_prev_dst, vmm_sum_scale_);
            }
            sum_scales_.push(sum_scale);
            sum_scales_.pop();
        };
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::apply_postops(const int data_idx) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    if (conf_.with_sum) apply_sum(data_idx);

    if (conf_.with_binary) {
        rhs_arg_params.vmm_idx_to_out_reg.emplace(data_idx, reg_dst_);
        rhs_arg_params.vmm_tail_idx_.emplace(data_idx);
    }

    postops_injector_->compute_vector(data_idx, rhs_arg_params);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::finalize() {
    if (static_cast<std::size_t>(conf_.reduce_size) > load_tail_size_) {
        reduce_vmm_to_scalar(
                vmm_acc_, vmm_tmp1_, vmm_tmp2_, vmm_tmp3_, simd_w_);
    }

    if (conf_.alg == alg_kind::reduction_mean) {
        const Xmm xmm_acc(vmm_acc_.getIdx());
        const Xmm xmm_reduce_size(vmm_tmp1_.getIdx());
        mov(reg_tmp_.cvt32(), float2int(static_cast<float>(conf_.reduce_size)));
        uni_vmovd(xmm_reduce_size, reg_tmp_.cvt32());
        uni_vdivss(xmm_acc, xmm_acc, xmm_reduce_size);
    }

    if (conf_.with_postops) apply_postops(vmm_acc_.getIdx());

    io_store_.store(vmm_acc_, ptr[reg_dst_], true);
}

template <cpu_isa_t isa, typename Vmm>
void jit_uni_reduction_kernel_t<isa, Vmm>::generate() {
    preamble();

    io_store_.init_bf16();
    if (conf_.is_saturation_needed) io_store_.init_saturate_f32();

    if (load_tail_size_ > 0) io_load_.prepare_tail_mask();
    io_store_.prepare_tail_mask();

    load_params();
    init_acc();
    reduce();
    finalize();

    postamble();

    if (conf_.with_eltwise && postops_injector_)
        postops_injector_->prepare_table();
}

template struct jit_uni_reduction_kernel_t<avx512_core_fp16>;
template struct jit_uni_reduction_kernel_t<avx512_core_bf16>;
template struct jit_uni_reduction_kernel_t<avx512_core>;
template struct jit_uni_reduction_kernel_t<avx2_vnni_2>;
template struct jit_uni_reduction_kernel_t<avx2_vnni_2, Xbyak::Xmm>;
template struct jit_uni_reduction_kernel_t<avx2>;
template struct jit_uni_reduction_kernel_t<avx2, Xbyak::Xmm>;
template struct jit_uni_reduction_kernel_t<avx>;
template struct jit_uni_reduction_kernel_t<avx, Xbyak::Xmm>;
template struct jit_uni_reduction_kernel_t<sse41>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
