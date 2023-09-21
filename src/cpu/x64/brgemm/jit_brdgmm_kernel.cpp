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
    , max_vmms_(isa_num_vregs(isa))
    , compute_dst_zp_(brg.zp_type_c != brgemm_broadcast_t::none)
    , compute_src_zp_(brg.zp_type_a != brgemm_broadcast_t::none)
    , compute_compensation_(compute_src_zp_ || brg.req_s8s8_compensation)
    , has_vpad_(brg.brgattr.max_top_vpad > 0 || brg.brgattr.max_bottom_vpad > 0)
    , has_bpad_(brg.brgattr.max_top_bpad > 0 || brg.brgattr.max_bottom_bpad > 0)
    , vmm_alloc(brg) {

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

    if (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad_) {
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

    if (brg.req_s8s8_compensation) {
        mov(reg_tmp, ptr[param1 + GET_OFF(ptr_buf)]);
        mov(ptr[rsp + reg_s8s8_comp_offs_], reg_tmp);
    }

    if (compute_dst_zp_) {
        mov(reg_tmp, ptr[param1 + GET_OFF(c_zp_values)]);
        mov(ptr[rsp + dst_zp_value_], reg_tmp);
    }

    if (compute_src_zp_) {
        mov(reg_tmp, ptr[param1 + GET_OFF(zp_a_val)]);
        mov(ptr[rsp + src_zp_value_], reg_tmp);

        mov(reg_tmp, ptr[param1 + GET_OFF(a_zp_compensations)]);
        mov(ptr[rsp + zp_compensation_], reg_tmp);
    }

    if (brg.with_binary) mov(ptr[rsp + abi_param1_offs_], param1);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::load_permute_vmm() {
    if (is_fast_vnni_int8()) {
        // load permute indices from data section
        mov(reg_tmp, permute_index_table);
        vmovdqu32(vmm_permute(), ptr[reg_tmp]);
    }
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

    if (req_vmm_reload()) load_permute_vmm();

    if (brg.req_s8s8_compensation) {
        mov(reg_tmp, 128);
        if (is_fast_vnni_int8())
            vpbroadcastb(vmm_shift(), reg_tmp.cvt8());
        else
            uni_vpbroadcastd(vmm_shift(), reg_tmp.cvt32());
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::restore_A_B_matrices() {
    if (brg.brgattr.max_bs > 1
            && (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad_))
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
            && (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad_))
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
            if (is_superset(brg.isa_impl, avx512_core)) {
                vcvtdq2ps(vmm_sum_zp, ptr_b[reg_ptr_sum_zp]);
            } else {
                uni_vpbroadcastd(vmm_sum_zp, ptr[reg_ptr_sum_zp]);
                vcvtdq2ps(vmm_sum_zp, vmm_sum_zp);
            }
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
                    auto vmm_scale = vmm_bcast();
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

            const Vmm vmm = accm(m_blocks, n_blocks, m, n, v_i);
            if (dq2ps_required) vcvtdq2ps(vmm, vmm);

            const bool mask_flag = substep_simd < simd_w_;
            const bool scale_embdbcast = !brg.is_oc_scale;
            if (IMPLICATION(mask_flag || scale_embdbcast,
                        is_superset(brg.isa_impl, avx512_core))) {
                const Vmm vmm_m = maybe_mask(vmm, mask_flag, false);
                if (scale_embdbcast) {
                    vmulps(vmm_m, vmm, ptr_b[reg_aux_scales]);
                } else {
                    vmulps(vmm_m, vmm,
                            ptr[reg_aux_scales + scales_offset(n, v_i)]);
                }
            } else {
                auto vmm_scale = vmm_tmp(0);
                const auto addr = ptr[reg_aux_scales + scales_offset(n, v_i)];
                if (scale_embdbcast) {
                    vbroadcastss(vmm_scale, ptr[reg_aux_scales]);
                } else {
                    load_data(data_type::f32, vmm_scale, addr, substep_simd);
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
        if (!is_superset(brg.isa_impl, avx512_core))
            vbroadcastss(vmm_dst_scales, ptr[reg_aux_dst_scales]);

        for_(int m = 0; m < m_blocks; m++)
        for_(int n = 0; n < n_blocks; n++)
        for (int v_i = 0; v_i < v_substep; ++v_i) {
            const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const Vmm vmm = accm(m_blocks, n_blocks, m, n, v_i);
            if (is_superset(brg.isa_impl, avx512_core)) {
                vmulps(vmm, vmm, ptr_b[reg_aux_dst_scales]);
            } else {
                vmulps(vmm, vmm, vmm_dst_scales);
            }
        }
    }

    if (compute_dst_zp_) {
        auto vmm_dst_zp = vmm_tmp(0);
        mov(reg_dst_zero_point, ptr[rsp + dst_zp_value_]);
        if (is_superset(brg.isa_impl, avx512_core)) {
            vcvtdq2ps(vmm_dst_zp,
                    EVEX_compress_addr(reg_dst_zero_point, 0, true));
        } else {
            uni_vpbroadcastd(vmm_dst_zp, ptr[reg_dst_zero_point]);
            vcvtdq2ps(vmm_dst_zp, vmm_dst_zp);
        }

        for_(int m = 0; m < m_blocks; m++)
        for_(int n = 0; n < n_blocks; n++)
        for (int v_i = 0; v_i < v_substep; ++v_i) {
            const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const Vmm vmm = accm(m_blocks, n_blocks, m, n, v_i);
            vaddps(vmm, vmm, vmm_dst_zp);
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
                    case data_type::s8:
                        if (is_superset(brg.isa_impl, avx512_core))
                            vpmovsdb(addr, r_vmm);
                        else
                            store_data(brg.dt_d, vmm, reg_aux_D, offset,
                                    substep_simd);
                        break;
                    case data_type::u8:
                        if (is_superset(brg.isa_impl, avx512_core))
                            vpmovusdb(addr, r_vmm);
                        else
                            store_data(brg.dt_d, vmm, reg_aux_D, offset,
                                    substep_simd);

                        break;
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
void jit_brdgmm_kernel_base_t<isa, Wmm>::compute_int8_compensation(
        int m_blocks, int n_blocks, bool has_n_tail) {

    const int v_substep = vnni_substep();

    if (brg.req_s8s8_compensation) {
        mov(reg_s8s8_comp, ptr[rsp + reg_s8s8_comp_offs_]);
        lea(reg_s8s8_comp, ptr[reg_s8s8_comp + reg_aux_N * sizeof(int32_t)]);
    }
    if (compute_src_zp_) {
        lea(reg_src_zero_point, ptr[rsp + src_zp_value_]);
        mov(reg_zp_compensation, ptr[rsp + zp_compensation_]);
        lea(reg_zp_compensation,
                ptr[reg_zp_compensation + reg_aux_N * sizeof(int32_t)]);
        if (!is_superset(brg.isa_impl, avx512_core))
            uni_vpbroadcastd(vmm_bcast(), ptr[reg_src_zero_point]);
    }

    for_(int v_i = 0; v_i < v_substep; ++v_i)
    for (int n = 0; n < n_blocks; n++) {
        const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
        if (substep_simd <= 0) continue;
        const size_t offset = comp_offset(n);
        if (brg.req_s8s8_compensation) {
            const Vmm vmm_comp = vmm_s8s8_comp();
            uni_vmovups(
                    vmm_comp, maybe_EVEX_compress_addr(reg_s8s8_comp, offset));
        }
        if (compute_src_zp_) {
            // zero_point: conv(src_x8, wei_s8) - src_shift_s32 * compensation_s32
            const Vmm vmm_zp = vmm_zp_comp();
            vmovups(vmm_zp,
                    maybe_EVEX_compress_addr(reg_zp_compensation, offset));
            if (is_superset(brg.isa_impl, avx512_core)) {
                const bool src_zp_is_common = true;
                vpmulld(vmm_zp, vmm_zp,
                        maybe_EVEX_compress_addr(
                                reg_src_zero_point, 0, src_zp_is_common));
            } else {
                vpmulld(vmm_zp, vmm_zp, vmm_bcast());
            }
        }
        for (int m = 0; m < m_blocks; m++) {
            auto vmm = accm(m_blocks, n_blocks, m, n, v_i);
            if (brg.req_s8s8_compensation) vpaddd(vmm, vmm, vmm_s8s8_comp());
            if (compute_src_zp_) vpaddd(vmm, vmm, vmm_zp_comp());
        }
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::store_accumulators(
        int m_blocks, int n_blocks, bool has_n_tail) {

    maybe_transpose_interleaved_vnni_to_plain(m_blocks, n_blocks, has_n_tail);

    if (is_fast_vnni_int8()) {
        for_(int m_i = 0; m_i < m_blocks; ++m_i)
        for (int n_i = 0; n_i < n_blocks; ++n_i) {
            auto vmm_out = accm(m_blocks, n_blocks, m_i, n_i, 0);
            vpermd(vmm_out, vmm_permute(), vmm_out);
        }
    }

    if (compute_compensation_)
        compute_int8_compensation(m_blocks, n_blocks, has_n_tail);

    const bool are_post_ops_applicable = one_of(true, brg.with_eltwise,
            brg.with_binary, brg.with_scales, brg.with_bias, brg.with_sum,
            brg.dt_d != brg.dt_c, brg.with_dst_scales, compute_dst_zp_);

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
        const bool preserve_8bit_sign
                = brg.is_int8 && one_of(brg.isa_impl, avx2_vnni_2, avx2_vnni);
        const auto dt_a = preserve_8bit_sign ? data_type::u8 : brg.dt_a;
        load_data(dt_a, vmma, addr, substep_simd);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::load_b(
        Vmm vmmb, int n_i, int v_i, bool has_n_tail, bool wei_zp) {
    assert(IMPLICATION(wei_zp, brg.is_int8 && compute_src_zp_));
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
        if (wei_zp) { // load weights for zero-point computation
            vpmovsxbd(vmmb, addr);
            if (is_fast_vnni_int8()) vpermd(vmmb, vmm_permute(), vmmb);
        } else {
            // wei is sign extend(s8), where as src is zero extended(u8).
            if (is_fast_vnni_int8()) {
                vbroadcasti32x4(vmmb, addr);
                vmovdqu8(vmmb | kblend_mask | T_z, vmmb);
            } else {
                vpmovsxbd(vmmb, addr);
            }
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
void jit_brdgmm_kernel_base_t<isa, Wmm>::comp_dot_product(
        compute_pad_kernel_t kernel_type, Vmm vmm_acc, Vmm vmmb) {
    switch (kernel_type) {
        case compute_pad_kernel_t::s8s8_kernel:
            vpdpbusd(vmm_acc, vmm_shift(), vmmb,
                    is_superset(isa, avx512_core) ? Xbyak::EvexEncoding
                                                  : Xbyak::VexEncoding);
            break;
        case compute_pad_kernel_t::zero_point_kernel:
            if (is_superset(brg.isa_impl, avx512_core)) {
                vpmulld(vmm_zp_comp(), vmmb,
                        maybe_EVEX_compress_addr(reg_src_zero_point, 0, true));
            } else {
                uni_vpbroadcastd(vmm_bcast(), ptr[reg_src_zero_point]);
                vpmulld(vmm_zp_comp(), vmmb, vmm_bcast());
            }
            vpaddd(vmm_acc, vmm_acc, vmm_zp_comp());
            break;
        default: assert(!"unsupported comp_kernel type");
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::pad_comp_kernel(
        compute_pad_kernel_t kernel_type, int m_blocks, int n_blocks,
        int padding, const Xbyak::Reg64 reg_pad,
        const std::function<int(int)> &get_mi, bool has_tail) {
    assert(vnni_substep() == 1);
    const int max_m_unroll = padding;
    const bool is_zero_point_kernel
            = kernel_type == compute_pad_kernel_t::zero_point_kernel;
    const int max_bvmms
            = accm(m_blocks, n_blocks, 0, 0, 0).getIdx() - vmm_b(0).getIdx();
    const int n_preload_b_vmms = max_bvmms >= n_blocks
            ? n_blocks
            : max_bvmms - 1 /*for ad-hoc load*/;
    const bool load_broadcast_wei = is_zero_point_kernel;
    for (int i = 0; i < n_preload_b_vmms; ++i) {
        const int n_i = i % n_blocks;
        const int v_i = i / n_blocks;
        if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
        load_b(vmm_b(i), n_i, v_i, has_tail, load_broadcast_wei);
    }

    Label jmp_table_base;
    std::vector<Label> jmp_table_labels(max_m_unroll + 1);
    // jmp table
    mov(reg_table_base, jmp_table_base);
    lea(reg_table_base, ptr[reg_table_base + reg_pad * sizeof(void *)]);
    jmp(ptr[reg_table_base], T_NEAR);
    align(8);
    L(jmp_table_base);
    for (int m_i = 0; m_i <= max_m_unroll; ++m_i) {
        putL(jmp_table_labels[m_i]);
    }

    for (int pad_i = max_m_unroll; pad_i > 0; --pad_i) {
        L(jmp_table_labels[pad_i]);
        if (is_zero_point_kernel)
            lea(reg_src_zero_point, ptr[rsp + src_zp_value_]);
        if (pad_i > m_blocks) continue;
        const int m_i = get_mi(pad_i);
        int p_b_i = 0;
        for (int n_i = 0; n_i < n_blocks; ++n_i, ++p_b_i) {
            if (get_substep_simd(n_i, 0, has_tail) <= 0) continue;
            const Vmm vmm_acc = accm(m_blocks, n_blocks, m_i, n_i, 0);
            if (p_b_i < n_preload_b_vmms) {
                comp_dot_product(kernel_type, vmm_acc, vmm_b(p_b_i));
            } else {
                // preloaded vmm_b not available
                const Vmm vmm_wei = vmm_b(max_bvmms - 1);
                load_b(vmm_wei, n_i, 0, has_tail, load_broadcast_wei);
                comp_dot_product(kernel_type, vmm_acc, vmm_wei);
            }
        }
    }
    L(jmp_table_labels[0]);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::batch_pad_kernel(
        int m_blocks, int n_blocks, bool has_tail) {

    assert(vnni_substep() == 1);
    const int max_bvmms
            = accm(m_blocks, n_blocks, 0, 0, 0).getIdx() - vmm_b(0).getIdx();

    auto kernel_body = [&](compute_pad_kernel_t kernel_type) {
        const bool is_zero_point_kernel
                = kernel_type == compute_pad_kernel_t::zero_point_kernel;
        if (is_zero_point_kernel)
            lea(reg_src_zero_point, ptr[rsp + src_zp_value_]);
        for (int nb_i = 0; nb_i < n_blocks; nb_i += max_bvmms) {
            const int n_e = nstl::min(nb_i + max_bvmms, n_blocks) - nb_i;
            for (int i = 0; i < n_e; ++i) {
                const int n_i = nb_i + i;
                if (get_substep_simd(n_i, 0, has_tail) <= 0) continue;
                const bool load_broadcast_wei = is_zero_point_kernel;
                load_b(vmm_b(i), n_i, 0, has_tail, load_broadcast_wei);
            }
            for_(int m_i = 0; m_i < m_blocks; ++m_i)
            for (int i = 0; i < n_e; ++i) {
                const int n_i = nb_i + i;
                if (get_substep_simd(n_i, 0, has_tail) <= 0) continue;
                const Vmm vmm_acc = accm(m_blocks, n_blocks, m_i, n_i, 0);
                comp_dot_product(kernel_type, vmm_acc, vmm_b(i));
            }
        }
    };

    if (brg.req_s8s8_compensation)
        kernel_body(compute_pad_kernel_t::s8s8_kernel);
    if (compute_src_zp_) kernel_body(compute_pad_kernel_t::zero_point_kernel);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::brdgmm_microkernel(int m_blocks,
        int n_blocks, bool has_top_padding, bool has_bottom_padding,
        bool has_tail, int shift_a) {

    const bool has_padding = has_top_padding || has_bottom_padding;
    const int max_bvmms
            = accm(m_blocks, n_blocks, 0, 0, 0).getIdx() - vmm_b(0).getIdx();
    const int v_substep = vnni_substep();
    assert(max_bvmms > 0);

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
            if (brg.isa_impl == avx2_vnni_2 && brg.dt_a == data_type::s8)
                vpdpbssd(vmm_acc, vmma, vmmb);
            else
                vpdpbusd(vmm_acc, vmma, vmmb,
                        is_superset(isa, avx512_core) ? Xbyak::EvexEncoding
                                                      : Xbyak::VexEncoding);
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
            if (grouped_bs()) {
                for_(int m_i = 0; m_i < m_blocks; ++m_i)
                for (int i = 0; i < n_e; ++i) {
                    const int n_i = nb_i + i;
                    if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                    const auto vmm_A = vmm_a(m_i + shift_a, i);
                    if (shift_a == 0 || m_i == m_blocks - 1) {
                        if (!is_fma_embd())
                            load_a(vmm_A, m_i, n_i, v_i, has_tail);
                        if (brg.req_s8s8_compensation)
                            vpaddb(vmm_A, vmm_A, vmm_shift());
                    }
                }
            }

            for_(int m_i = 0; m_i < m_blocks; ++m_i)
            for (int i = 0; i < n_e; ++i) {
                const int n_i = nb_i + i;
                if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                const auto vmm_A = vmm_a(m_i + shift_a, i);
                if (!grouped_bs() && (shift_a == 0 || m_i == m_blocks - 1)) {
                    if (!is_fma_embd()) load_a(vmm_A, m_i, n_i, v_i, has_tail);
                    if (brg.req_s8s8_compensation)
                        vpaddb(vmm_A, vmm_A, vmm_shift());
                }
                dot_product(vmm_A, vmm_b(i), m_i, n_i, v_i);
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
            jmp(ptr[reg_table_base], T_NEAR);

            align(64);
            L(jmp_table_base);
            for (int m_i = 0; m_i < m_blocks; ++m_i) {
                putL(jmp_table_labels[m_i]);
            }
        }

        for (int m_i = 0; m_i < m_blocks; ++m_i) {
            L(jmp_table_labels[m_i]);
            if (has_bottom_padding
                    && (m_blocks - m_i) <= brg.brgattr.max_bottom_vpad) {
                cmp(reg_aux_A_vpad_bottom, m_blocks - m_i);
                jge(done, T_NEAR);
            }

            if (grouped_bs()) {
                for_(int v_i = 0, p_b_i = 0; v_i < v_substep; ++v_i)
                for (int n_i = 0; n_i < n_blocks; ++n_i, ++p_b_i) {
                    if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                    if (shift_a == 0 || m_i == m_blocks - 1) {
                        const auto vmm_A = vmm_a(m_i + shift_a, n_i);
                        if (!is_fma_embd())
                            load_a(vmm_A, m_i, n_i, v_i, has_tail);
                        if (brg.req_s8s8_compensation)
                            vpaddb(vmm_A, vmm_A, vmm_shift());
                    }
                }
            }

            for_(int v_i = 0, p_b_i = 0; v_i < v_substep; ++v_i)
            for (int n_i = 0; n_i < n_blocks; ++n_i, ++p_b_i) {
                if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                const auto vmm_A = vmm_a(m_i + shift_a, n_i);
                if (!grouped_bs() && (shift_a == 0 || m_i == m_blocks - 1)) {
                    if (!is_fma_embd()) load_a(vmm_A, m_i, n_i, v_i, has_tail);
                    if (brg.req_s8s8_compensation)
                        vpaddb(vmm_A, vmm_A, vmm_shift());
                }
                if (p_b_i < n_preload_b_vmms) {
                    dot_product(vmm_A, vmm_b(p_b_i), m_i, n_i, v_i);
                } else {
                    // preloaded vmm_b not available
                    const int b_idx = max_bvmms - 1;
                    load_b(vmm_b(b_idx), n_i, v_i, has_tail);
                    dot_product(vmm_A, vmm_b(b_idx), m_i, n_i, v_i);
                }
            }
        }
        L(done);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::get_vertical_padding_info(
        const int m_blocks) {
    const bool do_check_effective_padding = check_effective_padding();
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
                ptr[reg_aux_batch_addr + GET_OFF_BATCH_ELEMENT(vvpad.top)]);
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

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::get_batch_padding_info() {
    mov(reg_total_padding,
            ptr[reg_aux_batch_addr
                    + GET_OFF_BATCH_ELEMENT(has_s8s8_comp_batch_pad)]);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::vertical_pad_kernel(
        const int m_blocks, const int n_blocks, bool has_n_tail) {
    const int tpad = brg.brgattr.max_top_vpad;
    const int bpad = brg.brgattr.max_bottom_vpad;
    if (tpad > 0) {
        auto get_mi = [=](int pad_i) { return pad_i - 1; };
        if (brg.req_s8s8_compensation)
            pad_comp_kernel(compute_pad_kernel_t::s8s8_kernel, m_blocks,
                    n_blocks, brg.brgattr.max_top_vpad, reg_aux_A_vpad_top,
                    get_mi, has_n_tail);
        if (compute_src_zp_)
            pad_comp_kernel(compute_pad_kernel_t::zero_point_kernel, m_blocks,
                    n_blocks, brg.brgattr.max_top_vpad, reg_aux_A_vpad_top,
                    get_mi, has_n_tail);
    }
    if (bpad > 0) {
        auto get_mi = [=](int pad_i) { return m_blocks - pad_i; };
        if (brg.req_s8s8_compensation)
            pad_comp_kernel(compute_pad_kernel_t::s8s8_kernel, m_blocks,
                    n_blocks, brg.brgattr.max_bottom_vpad,
                    reg_aux_A_vpad_bottom, get_mi, has_n_tail);
        if (compute_src_zp_)
            pad_comp_kernel(compute_pad_kernel_t::zero_point_kernel, m_blocks,
                    n_blocks, brg.brgattr.max_bottom_vpad,
                    reg_aux_A_vpad_bottom, get_mi, has_n_tail);
    }
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::call_brdgmm_microkernel(
        const int m_blocks, const int n_blocks, bool has_n_tail, int shift_a) {

    // padding for vertical dimensions
    const int tpad = brg.brgattr.max_top_vpad;
    const int bpad = brg.brgattr.max_bottom_vpad;
    Label microkernel_with_padding, done_microkernel, skip_microkernel_l;

    if (has_vpad_) {
        cmp(reg_total_padding, 0);
        jg(microkernel_with_padding, T_NEAR);
    }
    brdgmm_microkernel(m_blocks, n_blocks, false, false, has_n_tail, shift_a);
    if (has_vpad_) {
        jmp(done_microkernel, T_NEAR);
        L(microkernel_with_padding);

        if ((tpad + bpad) >= m_blocks) {
            cmp(reg_total_padding, m_blocks);
            jge(skip_microkernel_l, T_NEAR);
        }
        brdgmm_microkernel(m_blocks, n_blocks, tpad, bpad, has_n_tail, shift_a);
        L(skip_microkernel_l);

        vertical_pad_kernel(m_blocks, n_blocks, has_n_tail);
    }
    L(done_microkernel);
}

template <cpu_isa_t isa, typename Wmm>
void jit_brdgmm_kernel_base_t<isa, Wmm>::batch_loop(
        const int m_blocks, const int n_blocks, bool has_n_tail) {

    Label bs_loop_label, done_bs_loop;
    load_accumulators(m_blocks, n_blocks);
    cmp(reg_BS, 0);
    jle(done_bs_loop, T_NEAR);

    mov(reg_BS_loop, reg_BS);
    restore_A_B_matrices();

    auto bs_iteration = [&](int shift_a) {
        Label compute_brdgemm_l, end_batch_loop_l;
        set_A_B_matrices();
        if (compute_compensation_ && has_bpad_) {
            get_batch_padding_info();
            test(reg_total_padding, reg_total_padding);
            jle(compute_brdgemm_l, T_NEAR);

            batch_pad_kernel(m_blocks, n_blocks, has_n_tail);
            jmp(end_batch_loop_l, T_NEAR);
        }
        L(compute_brdgemm_l);
        if (has_vpad_) get_vertical_padding_info(m_blocks);
        call_brdgmm_microkernel(m_blocks, n_blocks, has_n_tail, shift_a);
        L(end_batch_loop_l);
    };

    L(bs_loop_label);
    {
        for (int sh = 0; sh < bs_group(); sh++) {
            bs_iteration(sh);
            advance_A_B_matrices();
        }
        sub(reg_BS_loop, bs_group());
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
                    assert(isa_has_masks(brg.isa_impl));
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

    assert(brg.bdb_tail == 0);
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

    if (assign_data_vmm_once()) load_permute_vmm();

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
const jit_generator *brdgmm_kernel_t<isa, Wmm>::get_jit_generator() const {
    return brgemm_kernel_;
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
template struct brdgmm_kernel_t<avx2_vnni, Xbyak::Ymm>;
template struct brdgmm_kernel_t<avx2_vnni_2, Xbyak::Ymm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
