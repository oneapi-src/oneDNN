/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/brgemm/brgemm_types.hpp"
#include "cpu/x64/brgemm/jit_brdgmm_kernel.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

template <cpu_isa_t isa, typename Wmm>
jit_brdgmm_kernel_base_t<isa, Wmm>::jit_brdgmm_kernel_base_t(
        const brgemm_t &abrd)
    : jit_generator(jit_name(), nullptr, MAX_CODE_SIZE, true, isa)
    , brg(abrd)
    , simd_w_(vreg_traits<Vmm>::vlen / brg.typesize_C)
    , max_vmms_(isa_num_vregs(isa)) {

    if (brg.with_eltwise || brg.with_binary || brg.with_sum) {

        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr bool use_exact_tail_scalar_bcast = false;
        const auto dst_md_wrapper = memory_desc_wrapper(brg.dst_md);
        const size_t tail = tail_length();

        static const bcast_set_t enabled_bcast_strategy
                = {broadcasting_strategy_t::scalar,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::no_broadcast};
        const binary_injector::rhs_arg_static_params_t rhs_sp {
                static_cast<size_t>(vmm_b().getIdx()), r14, r15, r13,
                preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                dst_md_wrapper, tail, k_mask, use_exact_tail_scalar_bcast};
        const binary_injector::static_params_t bsp {
                this->param1, enabled_bcast_strategy, rhs_sp};

        postops_injector_ = utils::make_unique<po_injector_t>(
                this, brg.attr->post_ops_, bsp);

        with_binary_non_scalar_bcast_
                = binary_injector::any_binary_postop_rhs_non_scalar_broadcast(
                        brg.attr->post_ops_, dst_md_wrapper);
    }
    if (brg.is_bf16_emu)
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                bf16_emu_scratch, bf16_emu_reserv_4, bf16_emu_reserv_4);
}

template <cpu_isa_t isa, typename Wmm>
template <typename U>
U jit_brdgmm_kernel_base_t<isa, Wmm>::maybe_mask(
        const U umm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? umm_in | k_mask : umm_in | k_mask | T_z)
                     : umm_in;
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::read_params() {
    Label label_done;

    mov(reg_BS, ptr[param1 + GET_OFF(BS)]);
    mov(reg_aux_C, ptr[param1 + GET_OFF(ptr_C)]);
    mov(reg_aux_D, ptr[param1 + GET_OFF(ptr_D)]);

    if (brg.type == brgemm_offs) {
        mov(reg_A, ptr[param1 + GET_OFF(ptr_A)]);
        mov(reg_B, ptr[param1 + GET_OFF(ptr_B)]);
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux1_A, ptr[param1 + GET_OFF(ptr_A)]);
        mov(reg_aux1_B, ptr[param1 + GET_OFF(ptr_B)]);
        if (brg.brgattr.max_bs > 1) {
            mov(ptr[rsp + reg_A_offs_], reg_aux1_A);
            mov(ptr[rsp + reg_B_offs_], reg_aux1_B);
        }
    }

    if (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad()) {
        mov(reg_aux_batch_addr, ptr[param1 + GET_OFF(batch)]);
        if (brg.brgattr.max_bs > 1)
            mov(ptr[rsp + reg_batch0_addr_offs_], reg_aux_batch_addr);
    }

    if (brg.with_bias) {
        mov(reg_tmp, ptr[param1 + GET_OFF(ptr_bias)]);
        mov(ptr[rsp + reg_bias_offs_], reg_tmp);
    }

    if (brg.with_scales) {
        mov(reg_tmp, ptr[param1 + GET_OFF(ptr_scales)]);
        mov(ptr[rsp + reg_scales_offs_], reg_tmp);
    }

    if (brg.with_dst_scales) {
        mov(reg_tmp, ptr[param1 + GET_OFF(ptr_dst_scales)]);
        mov(ptr[rsp + reg_dst_scales_offs_], reg_tmp);
    }

    if (brg.with_binary) mov(ptr[rsp + abi_param1_offs_], param1);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::load_accumulators(
        int m_blocks, int n_blocks) {
    const int v_substep = vnni_substep();
    for_(int v = 0; v < v_substep; ++v)
    for_(int m = 0; m < m_blocks; ++m)
    for (int n = 0; n < n_blocks; ++n) {
        auto vmm = accm(m_blocks, n_blocks, m, n, v);
        uni_vpxor(vmm, vmm, vmm);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::restore_A_B_matrices() {
    if (brg.brgattr.max_bs > 1
            && (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad()))
        mov(reg_aux_batch_addr, ptr[rsp + reg_batch0_addr_offs_]);

    if (brg.type == brgemm_strd && brg.brgattr.max_bs > 1) {
        mov(reg_aux1_A, ptr[rsp + reg_A_offs_]);
        mov(reg_aux1_B, ptr[rsp + reg_B_offs_]);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::set_A_B_matrices() {

    if (brg.type == brgemm_addr) {
        mov(reg_aux_A, ptr[reg_aux_batch_addr + GET_OFF_BATCH_ELEMENT(ptr.A)]);
        mov(reg_aux_B, ptr[reg_aux_batch_addr + GET_OFF_BATCH_ELEMENT(ptr.B)]);
    } else if (brg.type == brgemm_offs) {
        mov(reg_aux_A, reg_A);
        mov(reg_aux_B, reg_B);
        add(reg_aux_A,
                ptr[reg_aux_batch_addr + GET_OFF_BATCH_ELEMENT(offset.A)]);
        add(reg_aux_B,
                ptr[reg_aux_batch_addr + GET_OFF_BATCH_ELEMENT(offset.B)]);
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux_A, reg_aux1_A);
        mov(reg_aux_B, reg_aux1_B);
        if (brg.brgattr.max_bs > 1) {
            safe_add(reg_aux1_A, brg.stride_a, reg_tmp);
            safe_add(reg_aux1_B, brg.stride_b, reg_tmp);
        }
    }

    add(reg_aux_A, reg_a_offset);
    lea(reg_aux_B, ptr[reg_aux_B + reg_aux_N * brg.typesize_B]);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::advance_A_B_matrices() {
    if (brg.brgattr.max_bs > 1
            && (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad()))
        add(reg_aux_batch_addr, sizeof(brgemm_batch_element_t));
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::cvt2ps(data_type_t type_in,
        const Vmm vmm_in, const Xbyak::Operand &op, bool mask_flag,
        bool store) {
    const int tail_size = tail_length();
    const bool is_load_tail = op.isMEM() && mask_flag && tail_size > 0
            && (tail_size
                    < static_cast<int>(vreg_traits<Vmm>::vlen / sizeof(float)));
    if (IMPLICATION(is_load_tail, isa_has_masks(brg.isa_impl))) {
        const Vmm vmm = maybe_mask(vmm_in, is_load_tail, store);
        switch (type_in) {
            case data_type::f32:
            case data_type::s32: vmovups(vmm, op); break;
            case data_type::bf16:
                vpmovzxwd(vmm, op);
                vpslld(vmm, vmm, 16);
                break;
            case data_type::f16: vcvtph2ps(vmm, op); break;
            case data_type::s8: vpmovsxbd(vmm, op); break;
            case data_type::u8: vpmovzxbd(vmm, op); break;
            default: assert(!"unsupported data type");
        }
    } else {
        uni_vpxor(vmm_in, vmm_in, vmm_in);
        load_data(type_in, vmm_in, op.getAddress(), tail_size);
    }
    if (types::is_integral_dt(type_in)) vcvtdq2ps(vmm_in, vmm_in);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::apply_post_ops(
        int m_blocks, int n_blocks, bool has_n_tail) {

    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    injector_utils::vmm_index_set_t vmm_idxs_param;
    const int v_substep = vnni_substep();

    // collect vmm_idx's to apply post ops.
    // incase of avx2_vnni_2 tails, it is possible we do not need apply post-ops
    // to last vnni_substep
    for_(int v_i = 0; v_i < v_substep; ++v_i)
    for_(int m_i = 0; m_i < m_blocks; ++m_i)
    for (int n_i = 0; n_i < n_blocks; ++n_i) {
        if (get_substep_simd(n_i, v_i, has_n_tail) <= 0) continue;
        const auto vmm_idx = accm(m_blocks, n_blocks, m_i, n_i, v_i).getIdx();
        vmm_idxs_param.insert(vmm_idx);
    }

    if (brg.with_binary) {
        mov(reg_binary_params, ptr[rsp + abi_param1_offs_]);

        if (with_binary_non_scalar_bcast_) {

            for_(int v_i = 0; v_i < v_substep; ++v_i)
            for_(int m_i = 0; m_i < m_blocks; m_i++)
            for (int n_i = 0; n_i < n_blocks; n_i++) {
                const int substep_simd = get_substep_simd(n_i, v_i, has_n_tail);
                if (substep_simd <= 0) continue;
                const auto vmm_idx
                        = accm(m_blocks, n_blocks, m_i, n_i, v_i).getIdx();
                rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, reg_aux_D);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_idx, D_offset(m_i, n_i, v_i));

                if (n_i + 1 == n_blocks && has_n_tail && substep_simd < simd_w_)
                    rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        }
    }

    const auto sum_injector = [&] {
        const float *p_sum_scale = &brg.sum_scale;
        const int32_t *p_sum_zp = &brg.sum_zp;
        const bool p_sum_scale_reg_set = *p_sum_scale != 1.f;
        const bool p_sum_zp_reg_set = *p_sum_zp != 0;

        const injector_utils::conditional_register_preserve_guard_t
                register_guard_sum_scale(
                        (with_binary_non_scalar_bcast_) && p_sum_scale_reg_set,
                        this, {reg_ptr_sum_scale});
        const injector_utils::conditional_register_preserve_guard_t
                register_guard_sum_zp(p_sum_zp_reg_set, this, {reg_ptr_sum_zp});

        if (p_sum_scale_reg_set)
            mov(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));

        auto vmm_sum_zp = vmm_tmp(0);
        if (p_sum_zp_reg_set) {
            mov(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
            vcvtdq2ps(vmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
        }

        for_(int m_i = 0; m_i < m_blocks; m_i++)
        for_(int n_i = 0; n_i < n_blocks; n_i++)
        for (int v_i = 0; v_i < v_substep; v_i++) {
            const int substep_simd = get_substep_simd(n_i, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const auto vmm = accm(m_blocks, n_blocks, m_i, n_i, v_i);
            const auto addr = ptr[reg_aux_D + D_offset(m_i, n_i, v_i)];
            const auto vmm_prev_dst = vmm_tmp(1);
            cvt2ps(brg.sum_dt, vmm_prev_dst, addr, substep_simd != simd_w_,
                    false);
            if (p_sum_zp_reg_set) vsubps(vmm_prev_dst, vmm_sum_zp);
            if (!p_sum_scale_reg_set)
                vaddps(vmm, vmm_prev_dst);
            else {
                if (is_superset(brg.isa_impl, avx512_core)) {
                    vfmadd231ps(vmm, vmm_prev_dst, ptr_b[reg_ptr_sum_scale]);
                } else {
                    auto vmm_scale = vmm_tmp(2);
                    uni_vpbroadcastd(vmm_scale, ptr[reg_ptr_sum_scale]);
                    uni_vfmadd231ps(vmm, vmm_prev_dst, vmm_scale);
                }
            }
        }
    };

    if (brg.with_sum) {
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }

    postops_injector_->compute_vector_range(vmm_idxs_param, rhs_arg_params);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::store_accumulators_apply_post_ops(
        int m_blocks, int n_blocks, bool has_n_tail) {

    const bool dq2ps_required = brg.is_int8;
    const int v_substep = vnni_substep();
    if (brg.with_scales) {
        mov(reg_aux_scales, ptr[rsp + reg_scales_offs_]);
        if (brg.is_oc_scale) {
            lea(reg_aux_scales,
                    ptr[reg_aux_scales + reg_aux_N * sizeof(float)]);
        }
        for_(int m = 0; m < m_blocks; m++)
        for_(int n = 0; n < n_blocks; n++)
        for (int v_i = 0; v_i < v_substep; ++v_i) {
            const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const bool mask_flag = substep_simd < simd_w_;
            const Vmm vmm = maybe_mask(
                    accm(m_blocks, n_blocks, m, n, v_i), mask_flag, false);
            if (dq2ps_required) vcvtdq2ps(vmm, vmm);
            if (IMPLICATION(mask_flag || !brg.is_oc_scale,
                        is_superset(brg.isa_impl, avx512_core))) {
                if (brg.is_oc_scale) {
                    vmulps(vmm, vmm,
                            ptr[reg_aux_scales + scales_offset(n, v_i)]);
                } else {
                    vmulps(vmm, vmm, ptr_b[reg_aux_scales]);
                }
            } else {
                auto vmm_scale = vmm_tmp(0);
                const auto addr = ptr[reg_aux_scales + scales_offset(n, v_i)];
                if (brg.is_oc_scale) {
                    uni_vpxor(vmm_scale, vmm_scale, vmm_scale);
                    load_data(data_type::f32, vmm_scale, addr, substep_simd);
                } else {
                    vbroadcastss(vmm_scale, ptr[reg_aux_scales]);
                }
                vmulps(vmm, vmm, vmm_scale);
            }
        }
    }

    if (brg.with_bias) {
        mov(reg_aux_bias, ptr[rsp + reg_bias_offs_]);
        lea(reg_aux_bias, ptr[reg_aux_bias + reg_aux_N * brg.typesize_bias]);
    }

    for_(int v_i = 0; v_i < v_substep; ++v_i)
    for (int n = 0; n < n_blocks; n++) {
        auto vmm_bias = vmm_tmp(0);
        const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
        if (substep_simd <= 0) continue;
        if (brg.with_bias) {
            auto ptr_bias = ptr[reg_aux_bias + bias_offset(n, v_i)];
            cvt2ps(brg.dt_bias, vmm_bias, ptr_bias, substep_simd != simd_w_,
                    false);
        }
        for (int m = 0; m < m_blocks; m++) {
            auto vmm = accm(m_blocks, n_blocks, m, n, v_i);
            if (dq2ps_required && !brg.with_scales) vcvtdq2ps(vmm, vmm);
            if (brg.with_bias) { vaddps(vmm, vmm, vmm_bias); }
        }
    }

    if (postops_injector_) apply_post_ops(m_blocks, n_blocks, has_n_tail);

    if (brg.with_dst_scales) {
        mov(reg_aux_dst_scales, ptr[rsp + reg_dst_scales_offs_]);
        auto vmm_dst_scales = vmm_tmp(0);
        vbroadcastss(vmm_dst_scales, ptr[reg_aux_dst_scales]);

        for_(int m = 0; m < m_blocks; m++)
        for_(int n = 0; n < n_blocks; n++)
        for (int v_i = 0; v_i < v_substep; ++v_i) {
            const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const bool mask_flag = substep_simd < simd_w_;
            const Vmm vmm = maybe_mask(
                    accm(m_blocks, n_blocks, m, n, v_i), mask_flag, false);
            vmulps(vmm, vmm, ptr_b[reg_aux_dst_scales]);
        }
    }

    const bool dt_requires_saturation
            = one_of(brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    auto vmm_lbound = vmm_tmp(0);
    auto vmm_ubound = vmm_tmp(1);
    if (dt_requires_saturation) {
        init_saturate_f32(
                vmm_lbound, vmm_ubound, reg_tmp, data_type::f32, brg.dt_d);
    }

    if (brg.is_bf16_emu) bf16_emu_->init_vcvtneps2bf16();

    for (int m = 0; m < m_blocks; m++) {
        if (dt_requires_saturation) {
            for_(int n = 0; n < n_blocks; n++)
            for (int v_i = 0; v_i < v_substep; ++v_i) {
                if (get_substep_simd(n, v_i, has_n_tail) <= 0) continue;
                auto vmm = accm(m_blocks, n_blocks, m, n, v_i);
                saturate_f32(vmm, vmm_lbound, vmm_ubound, brg.dt_d);
                vcvtps2dq(vmm, vmm);
            }
        }

        for_(int n = 0; n < n_blocks; n++)
        for (int v_i = 0; v_i < v_substep; ++v_i) {
            const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const auto offset = D_offset(m, n, v_i);
            auto addr = ptr[reg_aux_D + offset];
            auto vmm = accm(m_blocks, n_blocks, m, n, v_i);
            auto vmm_low = Vmm_low_t(vmm.getIdx());
            const bool mask_flag = substep_simd < simd_w_;
            const Vmm r_vmm = maybe_mask(vmm, mask_flag, true);
            const Vmm_low_t r_vmm_low = maybe_mask(vmm_low, mask_flag, true);
            if (IMPLICATION(mask_flag, isa_has_masks(brg.isa_impl))) {
                switch (brg.dt_d) {
                    case data_type::f32:
                    case data_type::s32: vmovups(addr, r_vmm); break;
                    case data_type::bf16:
                        if (brg.is_bf16_emu)
                            bf16_emu_->vcvtneps2bf16(vmm_low, vmm);
                        else
                            vcvtneps2bf16(vmm_low, vmm,
                                    brg.isa_impl == avx2_vnni_2
                                            ? Xbyak::VexEncoding
                                            : Xbyak::EvexEncoding);
                        if (mask_flag)
                            vmovdqu16(addr, r_vmm_low);
                        else
                            vmovups(addr, r_vmm_low);
                        break;
                    case data_type::f16:
                        vcvtps2ph(addr, r_vmm, _op_mxcsr);
                        break;
                    case data_type::s8: vpmovsdb(addr, r_vmm); break;
                    case data_type::u8: vpmovusdb(addr, r_vmm); break;
                    default: assert(!"unknown dst_dt");
                }
            } else {
                store_data(brg.dt_d, vmm, reg_aux_D, offset, substep_simd);
            }
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::store_accumulators_without_post_ops(
        int m_blocks, int n_blocks, bool has_n_tail) {

    const bool dt_requires_saturation
            = brg.is_int8 && brg.dt_c != data_type::s32;
    auto vmm_lbound = vmm_tmp(0);
    auto vmm_ubound = vmm_tmp(1);
    if (dt_requires_saturation) {
        init_saturate_f32(
                vmm_lbound, vmm_ubound, reg_tmp, data_type::f32, brg.dt_d);
    }

    for_(int m = 0; m < m_blocks; m++)
    for_(int n = 0; n < n_blocks; n++)
    for (int v_i = 0; v_i < vnni_substep(); ++v_i) {
        const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
        if (substep_simd <= 0) continue;
        const bool mask_flag = substep_simd < simd_w_;
        auto vmm_acc = accm(m_blocks, n_blocks, m, n, v_i);
        if (dt_requires_saturation) {
            saturate_f32(vmm_acc, vmm_lbound, vmm_ubound, brg.dt_d);
            vcvtps2dq(vmm_acc, vmm_acc);
        }
        const auto offset = C_offset(m, n, v_i);
        if (IMPLICATION(mask_flag, isa_has_masks(brg.isa_impl))) {
            auto vmm_acc_masked = maybe_mask(vmm_acc, mask_flag, true);
            vmovups(ptr[reg_aux_C + offset], vmm_acc_masked);
        } else {
            store_data(brg.dt_c, vmm_acc, reg_aux_C, offset, substep_simd);
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa,
        Wmm>::maybe_transpose_interleaved_vnni_to_plain(int m_blocks,
        int n_blocks, bool has_n_tail) {

    if (vnni_substep() == 1) return;

    // The tail block is always processed as plain.
    // No need to transpose it here.
    const int n_blocks_e = n_blocks - has_n_tail;

    auto ymm_aux0 = vmm_tmp(0);
    for_(int m_i = 0; m_i < m_blocks; m_i++)
    for (int n_i = 0; n_i < n_blocks_e; n_i++) {
        auto ymm_even = accm(m_blocks, n_blocks, m_i, n_i, 0);
        auto ymm_odd = accm(m_blocks, n_blocks, m_i, n_i, 1);
        // reusing ymm_odd as aux
        // TODO: Check for any latency due to register dependency
        auto ymm_aux1 = ymm_odd;
        vpunpckldq(ymm_aux0, ymm_even, ymm_odd);
        vpunpckhdq(ymm_aux1, ymm_even, ymm_odd);
        vperm2i128(ymm_even, ymm_aux0, ymm_aux1, 0x20);
        vperm2i128(ymm_odd, ymm_aux0, ymm_aux1, 0x31);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::store_accumulators(
        int m_blocks, int n_blocks, bool has_n_tail) {

    maybe_transpose_interleaved_vnni_to_plain(m_blocks, n_blocks, has_n_tail);

    if (is_fast_vnni_int8() && brg.is_bf16_emu) {
        // load permute indices from data section
        mov(reg_tmp, permute_index_table);
        vmovdqu32(vmm_permute(), ptr[reg_tmp]);
    }

    if (is_fast_vnni_int8()) {
        for_(int m_i = 0; m_i < m_blocks; ++m_i)
        for (int n_i = 0; n_i < n_blocks; ++n_i) {
            auto vmm_out = accm(m_blocks, n_blocks, m_i, n_i, 0);
            vpermd(vmm_out, vmm_permute(), vmm_out);
        }
    }

    const bool are_post_ops_applicable = one_of(true, brg.with_eltwise,
            brg.with_binary, brg.with_scales, brg.with_bias, brg.with_sum,
            brg.dt_d != brg.dt_c, brg.with_dst_scales);

    Label label_done;
    if (are_post_ops_applicable) {
        store_accumulators_apply_post_ops(m_blocks, n_blocks, has_n_tail);
    } else {
        store_accumulators_without_post_ops(m_blocks, n_blocks, has_n_tail);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::load_a(
        Vmm vmma, int m_i, int n_i, int v_i, bool has_n_tail) {
    const int n_blocks
            = has_n_tail && n_block2_tail() > 0 ? n_block2_tail() : n_block2();
    const int substep_simd = get_substep_simd(n_i, v_i, has_n_tail);
    const bool is_tail_block = has_n_tail && n_i + 1 == n_blocks;
    const bool mask_flag = substep_simd < simd_w_;
    const auto addr = ptr[reg_aux_A + A_offset(m_i, n_i)
            + is_tail_block * v_i * simd_w_ * brg.typesize_A];
    if (IMPLICATION(mask_flag, isa_has_masks(brg.isa_impl))) {
        vmma = maybe_mask(vmma, mask_flag, false);
        if (brg.is_f32) {
            vmovups(vmma, addr);
        } else if (brg.is_bf16) {
            if (brg.isa_impl == avx2_vnni_2) {
                if (is_tail_block) {
                    vpmovzxwd(vmma, addr);
                    vpslld(vmma, vmma, 16);
                } else if (v_i == 0)
                    vcvtneebf162ps(vmma, addr);
                else
                    vcvtneobf162ps(vmma, addr);
            } else {
                vpmovzxwd(vmma, addr);
                if (brg.is_bf16_tmm) vpslld(vmma, vmma, 16);
            }
        } else if (brg.is_f16) {
            if (brg.isa_impl == avx2_vnni_2) {
                if (is_tail_block)
                    vcvtph2ps(vmma, addr);
                else if (v_i == 0)
                    vcvtneeph2ps(vmma, addr);
                else
                    vcvtneoph2ps(vmma, addr);
            } else
                vcvtph2ps(vmma, addr);
        } else if (brg.is_int8) {
            if (is_fast_vnni_int8()) {
                assert(!mask_flag);
                vbroadcasti32x4(vmma, addr);
            } else
                vpmovzxbd(vmma, addr);
        }
    } else {
        uni_vpxor(vmma, vmma, vmma);
        load_data(brg.dt_a, vmma, addr, substep_simd);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::load_b(
        Vmm vmmb, int n_i, int v_i, bool has_n_tail) {
    // for B matrix we assume memory is padded and it is safe to load simd
    // elements. is_tail only used during avx_ne_convert tail optimization.
    const int n_blocks
            = has_n_tail && n_block2_tail() > 0 ? n_block2_tail() : n_block2();
    const bool is_tail_block = has_n_tail && (n_i + 1 == n_blocks);
    const auto addr = ptr[reg_aux_B + B_offset(n_i)
            + is_tail_block * v_i * simd_w_ * brg.typesize_B];
    if (brg.is_f32) {
        vmovups(vmmb, addr);
    } else if (brg.is_int8) {
        // wei is sign extend(s8), where as src is zero extended(u8).
        if (is_fast_vnni_int8()) {
            vbroadcasti32x4(vmmb, addr);
            vmovdqu8(vmmb | kblend_mask | T_z, vmmb);
        } else {
            vpmovsxbd(vmmb, addr);
        }
    } else if (brg.is_f16) {
        if (brg.isa_impl == avx2_vnni_2) {
            if (is_tail_block)
                vcvtph2ps(vmmb, addr);
            else if (v_i == 0)
                vcvtneeph2ps(vmmb, addr);
            else
                vcvtneoph2ps(vmmb, addr);
        } else
            vcvtph2ps(vmmb, addr);
    } else if (brg.is_bf16) {
        if (brg.isa_impl == avx2_vnni_2) {
            if (is_tail_block) {
                vpmovzxwd(vmmb, addr);
                vpslld(vmmb, vmmb, 16);
            } else if (v_i == 0)
                vcvtneebf162ps(vmmb, addr);
            else
                vcvtneobf162ps(vmmb, addr);
        } else {
            vpmovzxwd(vmmb, addr);
            if (brg.is_bf16_tmm) vpslld(vmmb, vmmb, 16);
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::brdgmm_microkernel(int m_blocks,
        int n_blocks, bool has_top_padding, bool has_bottom_padding,
        bool has_tail) {

    const bool has_padding = has_top_padding || has_bottom_padding;
    const int max_bvmms
            = accm(m_blocks, n_blocks, 0, 0, 0).getIdx() - vmm_b(0).getIdx();
    const int v_substep = vnni_substep();

    auto dot_product = [&](Vmm vmma, Vmm vmmb, int m_i, int n_i, int v_i) {
        auto vmm_acc = accm(m_blocks, n_blocks, m_i, n_i, v_i);
        if (brg.is_f32) {
            if (is_fma_embd()) {
                const bool mask_flag = has_tail && (n_i + 1 == n_blocks);
                const auto addr = ptr[reg_aux_A + A_offset(m_i, n_i)];
                vmm_acc = maybe_mask(vmm_acc, mask_flag, false);
                vfmadd231ps(vmm_acc, vmmb, addr);
            } else {
                vfmadd231ps(vmm_acc, vmma, vmmb);
            }
        } else if (brg.is_bf16) {
            if (brg.is_bf16_tmm /* dont use vdpbf16ps on cpus supporting amx due
                                 to poor perf.*/
                    || brg.isa_impl == avx2_vnni_2)
                vfmadd231ps(vmm_acc, vmma, vmmb);
            else
                vdpbf16ps(vmm_acc, vmma, vmmb);
        } else if (brg.is_f16) {
            vfmadd231ps(vmm_acc, vmma, vmmb);
        } else if (brg.is_int8) {
            vpdpbusd(vmm_acc, vmma, vmmb);
        }
    };

    if (!has_padding) {
        // preload vmm_b if possible.
        for_(int v_i = 0; v_i < v_substep; ++v_i)
        for (int nb_i = 0; nb_i < n_blocks; nb_i += max_bvmms) {
            const int n_e = nstl::min(nb_i + max_bvmms, n_blocks) - nb_i;
            for (int i = 0; i < n_e; ++i) {
                const int n_i = nb_i + i;
                if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                load_b(vmm_b(i), n_i, v_i, has_tail);
            }
            for_(int m_i = 0; m_i < m_blocks; ++m_i)
            for (int i = 0; i < n_e; ++i) {
                const int n_i = nb_i + i;
                if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                if (!is_fma_embd()) load_a(vmm_a(), m_i, n_i, v_i, has_tail);
                dot_product(vmm_a(), vmm_b(i), m_i, n_i, v_i);
            }
        }
    } else {
        const int max_req_preload_vmms = n_blocks * vnni_substep();
        const int n_preload_b_vmms = max_bvmms >= max_req_preload_vmms
                ? max_req_preload_vmms
                : max_bvmms - 1 /*for ad-hoc load*/;
        for (int i = 0; i < n_preload_b_vmms; ++i) {
            const int n_i = i % n_blocks;
            const int v_i = i / n_blocks;
            if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
            load_b(vmm_b(i), n_i, v_i, has_tail);
        }

        Label done;
        Label jmp_table_base;
        std::vector<Label> jmp_table_labels(m_blocks);
        if (has_top_padding) {
            // jmp table
            mov(reg_table_base, jmp_table_base);
            lea(reg_table_base,
                    ptr[reg_table_base + reg_aux_A_vpad_top * sizeof(void *)]);
            jmp(ptr[reg_table_base]);
            align(8);
            L(jmp_table_base);
            for (int m_i = 0; m_i < m_blocks; ++m_i) {
                putL(jmp_table_labels[m_i]);
            }
        }

        for (int m_i = 0; m_i < m_blocks; ++m_i) {
            L(jmp_table_labels[m_i]);
            if (has_bottom_padding) {
                cmp(reg_aux_A_vpad_bottom, m_blocks - m_i);
                jge(done, T_NEAR);
            }

            for_(int v_i = 0, p_b_i = 0; v_i < v_substep; ++v_i)
            for (int n_i = 0; n_i < n_blocks; ++n_i, ++p_b_i) {
                if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                if (!is_fma_embd()) load_a(vmm_a(), m_i, n_i, v_i, has_tail);
                if (p_b_i < n_preload_b_vmms) {
                    dot_product(vmm_a(), vmm_b(p_b_i), m_i, n_i, v_i);
                } else {
                    // preloaded vmm_b not available
                    const int b_idx = max_bvmms - 1;
                    load_b(vmm_b(b_idx), n_i, v_i, has_tail);
                    dot_product(vmm_a(), vmm_b(b_idx), m_i, n_i, v_i);
                }
            }
        }
        L(done);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::batch_loop(
        const int m_blocks, const int n_blocks, bool has_n_tail) {

    auto get_padding_info = [&]() {
        const bool do_check_effective_padding = check_effective_padding();
        if (has_vpad()) {
            Label no_top_padding;

            if (brg.brgattr.max_bottom_vpad > 0) {
                if (do_check_effective_padding) {
                    Label done_adjust_bottom_padding;
                    mov(reg_aux_A_vpad_bottom, reg_aux_M);
                    add(reg_aux_A_vpad_bottom, m_blocks - M());
                    add(reg_aux_A_vpad_bottom,
                            ptr[reg_aux_batch_addr
                                    + GET_OFF_BATCH_ELEMENT(vvpad.bottom)]);
                    jge(done_adjust_bottom_padding, T_NEAR);
                    xor_(reg_aux_A_vpad_bottom, reg_aux_A_vpad_bottom);
                    L(done_adjust_bottom_padding);
                } else {
                    mov(reg_aux_A_vpad_bottom,
                            ptr[reg_aux_batch_addr
                                    + GET_OFF_BATCH_ELEMENT(vvpad.bottom)]);
                }
                mov(reg_total_padding, reg_aux_A_vpad_bottom);
            }
            if (brg.brgattr.max_top_vpad > 0) {
                mov(reg_aux_A_vpad_top,
                        ptr[reg_aux_batch_addr
                                + GET_OFF_BATCH_ELEMENT(vvpad.top)]);
                if (do_check_effective_padding) {
                    Label done_adjust_top_padding;
                    sub(reg_aux_A_vpad_top, reg_aux_M);
                    jge(done_adjust_top_padding, T_NEAR);
                    xor_(reg_aux_A_vpad_top, reg_aux_A_vpad_top);
                    L(done_adjust_top_padding);
                }
                if (brg.brgattr.max_bottom_vpad > 0) {
                    add(reg_total_padding, reg_aux_A_vpad_top);
                } else {
                    mov(reg_total_padding, reg_aux_A_vpad_top);
                }
            }
        }
    };

    auto call_brdgmm_microkernel = [&]() {
        const int tpad = brg.brgattr.max_top_vpad;
        const int bpad = brg.brgattr.max_bottom_vpad;
        const bool vpad_exists = has_vpad();
        Label microkernel_with_padding, done_microkernel;

        if (vpad_exists) {
            cmp(reg_total_padding, 0);
            jg(microkernel_with_padding, T_NEAR);
        }
        brdgmm_microkernel(m_blocks, n_blocks, false, false, has_n_tail);
        if (vpad_exists) {
            jmp(done_microkernel, T_NEAR);
            L(microkernel_with_padding);
            if ((tpad + bpad) >= m_blocks) {
                cmp(reg_total_padding, m_blocks);
                jge(done_microkernel, T_NEAR);
            }
            brdgmm_microkernel(m_blocks, n_blocks, tpad, bpad, has_n_tail);
        }
        L(done_microkernel);
    };

    Label bs_loop_label, done_bs_loop;
    load_accumulators(m_blocks, n_blocks);
    cmp(reg_BS, 0);
    jle(done_bs_loop, T_NEAR);
    mov(reg_BS_loop, reg_BS);
    restore_A_B_matrices();

    L(bs_loop_label);
    {
        set_A_B_matrices();
        get_padding_info();
        advance_A_B_matrices();
        call_brdgmm_microkernel();
        dec(reg_BS_loop);
        jg(bs_loop_label, T_NEAR);
    }

    L(done_bs_loop);

    store_accumulators(m_blocks, n_blocks, has_n_tail);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::compute_loop() {

    const bool has_m_block2_tail = m_block2_tail() > 0;
    const int loop_m = (nb_m_block2() - has_m_block2_tail);
    const bool do_loop_m = loop_m > 1;

    const bool has_n_block2_tail = n_block2_tail() > 0;
    const bool need_separate_n_block1_tail_block = n_block1_tail() != 0
            && !has_n_block2_tail && nb_n_block2() > 1
            && !isa_has_masks(brg.isa_impl);
    const int loop_n = nb_n_block2() - has_n_block2_tail
            - need_separate_n_block1_tail_block;
    const bool do_loop_n = loop_n > 1;
    const bool loop_n_update_aux_ptrs = do_loop_n || (loop_n < nb_n_block2());

    auto n_loop = [&](int m_blocks) {
        Label n_loop_label;
        const int n_blocks = n_block2();
        const int n_loop_step = oc_logical_offset(n_blocks);
        const int n_loop_work = loop_n * n_blocks * n_block1();
        const bool vlen_tail_in_loop = n_block1_tail() != 0
                && !need_separate_n_block1_tail_block && !has_n_block2_tail;

        xor_(reg_aux_N, reg_aux_N);

        L(n_loop_label);
        {
            if (do_loop_n) {
                if (vlen_tail_in_loop) {
                    Label done_k_mask;
                    cmp(reg_aux_N, n_loop_work - n_loop_step);
                    jl(done_k_mask, T_NEAR);
                    kmovd(k_mask, k_tail_mask);
                    L(done_k_mask);
                }
            }

            batch_loop(m_blocks, n_blocks, vlen_tail_in_loop);

            if (loop_n_update_aux_ptrs) {
                add(reg_aux_N, n_loop_step);
                add(reg_a_offset, n_loop_step * brg.typesize_A);
                add(reg_aux_C, n_loop_step * brg.typesize_C);
                add(reg_aux_D, n_loop_step * brg.typesize_D);
            }

            if (do_loop_n) {
                cmp(reg_aux_N, n_loop_work);
                jl(n_loop_label, T_NEAR);
            }
        }

        if (need_separate_n_block1_tail_block)
            batch_loop(m_blocks, n_blocks, true);

        if (has_n_block2_tail) {
            batch_loop(m_blocks, n_block2_tail(), n_block1_tail() != 0);
        }
    };

    auto m_loop = [&]() {
        Label m_loop_label;
        const int m_blocks = m_block2();
        const bool reset_mask = isa_has_masks(brg.isa_impl)
                && n_block1_tail() != 0 && do_loop_n && !has_n_block2_tail;

        xor_(reg_aux_M, reg_aux_M);
        xor_(reg_a_offset, reg_a_offset);

        L(m_loop_label);
        {
            if (reset_mask) kxnorq(k_mask, k_mask, k_mask);
            n_loop(m_blocks);

            if (do_loop_m || has_m_block2_tail) {
                add(reg_aux_M, m_blocks);
                const int n_loop_offset
                        = loop_n_update_aux_ptrs * loop_n * n_block2();
                add(reg_a_offset, A_offset(m_blocks, -n_loop_offset));
                add(reg_aux_C, C_offset(m_blocks, -n_loop_offset, 0));
                add(reg_aux_D, D_offset(m_blocks, -n_loop_offset, 0));
            }

            if (do_loop_m) {
                cmp(reg_aux_M, loop_m * m_block2());
                jl(m_loop_label, T_NEAR);
            }
        }

        if (m_block2_tail() > 0) {
            if (reset_mask) { kxnorq(k_mask, k_mask, k_mask); }
            n_loop(m_block2_tail());
        }
    };

    assert(m_block1_tail() == 0);
    m_loop();
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::init_masks() {
    if (!isa_has_masks(brg.isa_impl)) return;

    if (is_fast_vnni_int8()) {
        mov(reg_tmp, 0x8888444422221111);
        kmovq(kblend_mask, reg_tmp);
    }

    if (n_block1_tail() != 0) {
        const auto tail_mask = size_t((1 << n_block1_tail()) - 1);
        const bool has_n_block2_tail = n_block2_tail() > 0;
        mov(reg_tmp, tail_mask);
        if (has_n_block2_tail || nb_n_block2() <= 1) {
            // The mask can be set only once.
            kmovq(k_mask, reg_tmp);
        } else {
            // Need to adjust mask, and set only when needed.
            // So store it temporarily in k_tail_mask.
            kmovq(k_tail_mask, reg_tmp);
        }
    } else if (brg.with_binary) {
        // the post-ops injector seems to use mask unconditionally
        // set a default mask.
        kxnorq(k_mask, k_mask, k_mask);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::generate() {

    preamble();
    sub(rsp, stack_space_needed_);

    init_masks();

    if (is_fast_vnni_int8() && !brg.is_bf16_emu) {
        // load permute indices from data section
        mov(reg_tmp, permute_index_table);
        vmovdqu32(vmm_permute(), ptr[reg_tmp]);
    }

    read_params();
    compute_loop();

    add(rsp, stack_space_needed_);
    postamble();

    if (brg.with_eltwise) postops_injector_->prepare_table();

    if (is_fast_vnni_int8()) {
        align(64);
        L(permute_index_table);
        const uint32_t _idx[]
                = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            dd(_idx[i]);
    }
}

template <cpu_isa_t isa, typename Wmm>
brdgmm_kernel_t<isa, Wmm>::brdgmm_kernel_t(const brgemm_t abrd) {
    brgemm_kernel_ = new jit_brdgmm_kernel_base_t<isa, Wmm>(abrd);
}

template <cpu_isa_t isa, typename Wmm>
status_t brdgmm_kernel_t<isa, Wmm>::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

template <cpu_isa_t isa, typename Wmm>
void brdgmm_kernel_t<isa, Wmm>::operator()(
        brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

template <cpu_isa_t isa, typename Wmm>
brdgmm_kernel_t<isa, Wmm>::~brdgmm_kernel_t() {
    delete brgemm_kernel_;
}

template struct brdgmm_kernel_t<avx512_core_fp16, Xbyak::Zmm>;
template struct brdgmm_kernel_t<avx512_core_bf16, Xbyak::Zmm>;
template struct brdgmm_kernel_t<avx512_core_vnni, Xbyak::Zmm>;
template struct brdgmm_kernel_t<avx512_core, Xbyak::Zmm>;
template struct brdgmm_kernel_t<avx2, Xbyak::Ymm>;
template struct brdgmm_kernel_t<avx2_vnni_2, Xbyak::Ymm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
