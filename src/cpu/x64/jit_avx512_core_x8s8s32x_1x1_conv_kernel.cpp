/*******************************************************************************
* Copyright 2018-2025 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_avx512_core_x8s8s32x_1x1_conv_kernel.hpp"
#include "cpu/x64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::prop_kind;
using namespace Xbyak;

template <typename Vmm>
_jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::
        _jit_avx512_core_x8s8s32x_1x1_conv_kernel(
                const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr,
                const memory_desc_t &dst_md)
    : jit_generator(jit_name())
    , jcp(ajcp)
    , attr_(attr)
    , postops_injector_(nullptr) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr unsigned helper_vmm_idx = 31;
        const size_t oc_block_tail = jcp.oc_block % isa_simd_width_;
        const size_t tail_size = oc_block_tail
                ? oc_block_tail
                : jcp.oc_without_padding % isa_simd_width_;
        static constexpr bool use_exact_tail_scalar_bcast = true;

        const rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx,
                r14, r15, r13, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(dst_orig),
                memory_desc_wrapper(dst_md), tail_size, postops_mask,
                use_exact_tail_scalar_bcast};
        const static_params_t static_params {
                this->param1, rhs_arg_static_params};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<avx512_core, Vmm>>(
                this, jcp.post_ops, static_params);
    }
    if (jcp.dst_dt == data_type::bf16 && !isa_has_bf16(jcp.isa))
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this,
                bf16_emu_reserv_1, bf16_emu_reserv_2, bf16_emu_reserv_3,
                bf16_emu_reserv_4, bf16_emu_reserv_5, bf16_emu_reserv_5);
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::bcast_loop(
        int load_loop_blk) {
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);

    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, EVEX_compress_addr(rsp, bcast_loop_work_off));

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop);
    {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            reduce_loop(load_loop_blk, jcp.ur, false);
            if (i < num_substeps - 1) {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_substep);
            } else {
                add(aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_bcast_substep);
                int output_offset = jcp.bcast_loop_output_step
                        - (num_substeps - 1) * jcp.bcast_loop_output_substep;

                add(aux_reg_output_data, output_offset);
            }
        }
        sub(bcast_loop_iter, jcp.bcast_block);
        cmp(bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        cmp(bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        reduce_loop(load_loop_blk, jcp.ur_tail, true);
        L(bcast_loop_tail_out);
    }
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::cvt2ps(data_type_t type_in,
        const Vmm vmm_in, const Xbyak::Operand &op, bool mask_flag) {
    using namespace data_type;
    const Vmm vmm = mask_flag ? vmm_in | k_load_dim_mask | T_z : vmm_in;
    switch (type_in) {
        case f32:
        case s32: vmovups(vmm, op); break;
        case bf16: vpmovzxwd(vmm, op); break;
        case s8: vpmovsxbd(vmm, op); break;
        case u8: vpmovzxbd(vmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (one_of(type_in, s32, s8, u8))
        vcvtdq2ps(vmm_in, vmm_in);
    else if (type_in == bf16)
        vpslld(vmm_in, vmm_in, 16);
}

template <typename F>
static void iterate(const int load_loop_blk, const int ur,
        const bool last_oc_block_flag, const bool force_masking, const F &f) {
    for (int i_load = 0; i_load < load_loop_blk; i_load++) {
        const bool mask_flag = force_masking
                || (last_oc_block_flag && i_load + 1 == load_loop_blk);
        for (int i_ur = 0; i_ur < ur; i_ur++)
            f(mask_flag, i_load, i_ur);
    }
}
template <typename F>
static void iterate(const int load_loop_blk, const int ur,
        const bool last_oc_block_flag, const F &f) {
    iterate(load_loop_blk, ur, last_oc_block_flag, false, f);
}
template <typename F>
static void iterate(const int load_loop_blk, const int ur, const F &f) {
    iterate(load_loop_blk, ur, false, false, f);
}

template <typename Vmm>
Address _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::output_ptr(
        const int i_load, const int i_ur) {
    const size_t ur_stride = jcp.with_dw_conv
            ? jcp.nb_load_blocking * jcp.oc_block * i_ur
            : jcp.oc_without_padding * jcp.ngroups * i_ur;

    return EVEX_compress_addr(aux_reg_output_data,
            jcp.typesize_out * (ur_stride + i_load * jcp.load_block));
};

template <typename Vmm>
int _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::vreg_accum_idx(
        const int load_loop_blk, int i_load, int i_ur) const {
    return (i_ur * load_loop_blk + i_load);
};

template <typename Vmm>
Vmm _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::vreg_accum(
        const int load_loop_blk, int i_load, int i_ur) const {
    return Vmm(vreg_accum_idx(load_loop_blk, i_load, i_ur));
};

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::apply_sum(
        const int load_loop_blk, const int ur, const bool mask_flag_in,
        const float *p_sum_scale, const int32_t *p_sum_zp) {
    if (jcp.with_sum) {
        const float sum_scale = *p_sum_scale;
        const int32_t sum_zp = *p_sum_zp;
        const auto sum_injector_lam
                = [this, sum_scale, sum_zp, load_loop_blk](const bool mask_flag,
                          const int i_load, const int i_ur) {
                      const auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                      cvt2ps(jcp.sum_dt, vmm_prev_dst, output_ptr(i_load, i_ur),
                              mask_flag);
                      if (sum_zp != 0) vsubps(vmm_prev_dst, vmm_tmp);
                      if (sum_scale == 1.f)
                          vaddps(r, vmm_prev_dst);
                      else
                          vfmadd231ps(
                                  r, vmm_prev_dst, zword_b[reg_ptr_sum_scale]);
                  };
        // Capture by value has to be applied since this lambda is called from
        // a different context when stack values are unavailable.
        const auto sum_injector = [load_loop_blk, ur, mask_flag_in,
                                          sum_injector_lam]() {
            iterate(load_loop_blk, ur, mask_flag_in, sum_injector_lam);
        };
        if (sum_zp != 0) vcvtdq2ps(vmm_tmp, ptr_b[rsp + reg_ptr_sum_zp_off]);
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::apply_postops(
        const int load_loop_blk, const int ur, const bool mask_flag_in,
        const float *p_sum_scale, const int32_t *p_sum_zp) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum) {

        apply_sum(load_loop_blk, ur, mask_flag_in, p_sum_scale, p_sum_zp);

        injector_utils::vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params,
                    rhs_arg_params_tail;
            const auto mask_tail = jcp.oc_without_padding % jcp.load_block;
            const bool oc_blk_is_smaller_than_vmm
                    = jcp.oc_block < isa_simd_width_;
            iterate(load_loop_blk, ur, mask_tail, oc_blk_is_smaller_than_vmm,
                    [&](const bool mask_flag, const int i_load,
                            const int i_ur) {
                        const int ur_stride
                                = jcp.oc_without_padding * jcp.ngroups * i_ur;
                        const size_t aux_output_l_off = jcp.typesize_out
                                * (ur_stride + i_load * jcp.load_block);
                        const auto vmm_idx
                                = vreg_accum_idx(load_loop_blk, i_load, i_ur);
                        vmm_idxs.emplace(vmm_idx);

                        rhs_arg_params_tail.vmm_idx_to_out_reg.emplace(
                                vmm_idx, aux_reg_output_data);
                        rhs_arg_params_tail.vmm_idx_to_out_elem_off_val.emplace(
                                vmm_idx, aux_output_l_off);
                        if (mask_flag)
                            rhs_arg_params_tail.vmm_tail_idx_.emplace(vmm_idx);
                    });
            rhs_arg_params = rhs_arg_params_tail;
            rhs_arg_params.vmm_tail_idx_.clear();

            mov(abi_param1, EVEX_compress_addr(rsp, reg_abi_param1_backup));

            Label postops_done;
            if (mask_tail || oc_blk_is_smaller_than_vmm) {
                Label postops_no_tail;
                if (mask_tail) {
                    test(reg_reduce_pos_flag, FLAG_OC_LAST);
                    jz(postops_no_tail, T_NEAR);
                    cmp(reg_load_loop_work, 0);
                    jg(postops_no_tail, T_NEAR);
                }
                postops_injector_->compute_vector_range(
                        vmm_idxs, rhs_arg_params_tail);
                jmp(postops_done, T_NEAR);
                L(postops_no_tail);
            }
            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params);
            L(postops_done);

        } else {
            iterate(load_loop_blk, ur,
                    [&](const bool, const int i_load, const int i_ur) {
                        vmm_idxs.emplace(
                                vreg_accum_idx(load_loop_blk, i_load, i_ur));
                    });
            postops_injector_->compute_vector_range(vmm_idxs);
        }
    }
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::reduce_loop(
        int load_loop_blk, int ur, bool wraparound) {
    auto vreg_load = [ur, load_loop_blk](int i_load) {
        return Vmm(ur * load_loop_blk + i_load);
    };

    auto bias_ptr = [this](int i_load) {
        return EVEX_compress_addr(
                reg_bias_data, jcp.typesize_bia * jcp.oc_block * i_load);
    };

    auto comp_ptr = [this](int i_load) {
        return EVEX_compress_addr(
                reg_comp_data, sizeof(int32_t) * jcp.oc_block * i_load);
    };

    auto scale_ptr = [this](int i_load) {
        return EVEX_compress_addr(reg_ptr_scales,
                jcp.is_oc_scale * (sizeof(float) * jcp.oc_block * i_load));
    };

    auto bcast_ptr = [this](int i_reduce, int i_ur, bool bcast) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        assert(jcp.reduce_loop_unroll == jcp.reduce_block);

        int offt = (jcp.ic_without_padding * i_ur * jcp.ngroups + i_reduce);

        return EVEX_compress_addr(
                aux_reg_bcast_data, jcp.typesize_in * offt, bcast);
    };

    auto load_ptr = [this](int i_reduce, int i_load) {
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;

        int offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;

        return EVEX_compress_addr(aux_reg_load_data,
                u1 * jcp.reduce_loop_load_step + jcp.typesize_in * offt);
    };

    auto init = [this, load_loop_blk, ur]() {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                vpxord(r, r, r);
            }
        if (jcp.signed_input) {
            mov(reg_scratch, -128);
            vpbroadcastb(vmm_shift, reg_scratch.cvt8());
        }
    };

    auto store = [&](const bool mask_flag_in) {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale = nullptr;
        const int32_t *p_sum_zp = nullptr;
        if (sum_idx != -1) {
            p_sum_scale = &p.entry_[sum_idx].sum.scale;
            p_sum_zp = &p.entry_[sum_idx].sum.zero_point;
        }
        const auto p_sum_scale_val = p_sum_scale ? *p_sum_scale : 1.f;
        const auto p_sum_zp_val = p_sum_zp ? *p_sum_zp : 0;
        const bool is_scale_or_zp_sum
                = p_sum_zp_val != 0 || p_sum_scale_val != 1.f;
        mov(EVEX_compress_addr(rsp, reg_bcast_data_off), reg_bcast_data);
        mov(reg_ptr_scales, EVEX_compress_addr(rsp, reg_ptr_sum_scale_off));
        if (is_scale_or_zp_sum) {
            mov(EVEX_compress_addr(rsp, reg_load_data_off), reg_load_data);
            if (p_sum_zp_val != 0) {
                mov(reg_load_data, p_sum_zp_val);
                mov(ptr[rsp + reg_ptr_sum_zp_off], reg_load_data);
            }
            if (p_sum_scale_val != 1.f)
                mov(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));
        }
        if (jcp.signed_input && (!jcp.has_vnni)) {
            mov(reg_scratch, float2int(jcp.wei_adj_scale));
        }
        if (jcp.src_zero_point) {
            mov(reg_zp_compensation,
                    EVEX_compress_addr(rsp, reg_zp_compensation_off));
            mov(reg_src_zero_point,
                    EVEX_compress_addr(rsp, reg_src_zero_point_off));
        }
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            auto vmm_bias = vmm_tmp;
            auto vmm_comp = vmm_bcast;
            if (jcp.with_bias) {
                if (jcp.signed_input || jcp.dst_scale)
                    mov(reg_bias_data,
                            EVEX_compress_addr(rsp, reg_bias_data_off));
                cvt2ps(jcp.bia_dt, vmm_bias, bias_ptr(i_load), mask_flag);
            }
            if (jcp.signed_input) {
                mov(reg_comp_data, EVEX_compress_addr(rsp, reg_comp_data_off));
                cvt2ps(data_type::s32, vmm_comp, comp_ptr(i_load), mask_flag);
            }
            if (jcp.src_zero_point) {
                // zero_point: conv(src_x8, wei_s8) - src_shift_s32 * compensation_s32
                const int zp_offset = sizeof(int32_t) * i_load * jcp.load_block;
                vmovups(vmm_zp,
                        EVEX_compress_addr(reg_zp_compensation, zp_offset));
                vpmulld(vmm_zp, vmm_zp,
                        EVEX_compress_addr(
                                reg_src_zero_point, 0, jcp.zp_src_is_common));
                // upscale to f32
                const Vmm vmm_
                        = mask_flag ? vmm_zp | k_load_dim_mask | T_z : vmm_zp;
                vcvtdq2ps(vmm_, vmm_);
            }
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                vcvtdq2ps(r, r);
                if (jcp.signed_input) vaddps(r, r, vmm_comp);
                if (jcp.src_zero_point) vaddps(r, r, vmm_zp);

                const Vmm mask_vmm = mask_flag ? r | k_load_dim_mask | T_z : r;
                vmulps(mask_vmm, r, scale_ptr(i_load));

                if (jcp.with_bias) vaddps(r, r, vmm_bias);
            }
        }

        apply_postops(load_loop_blk, ur, mask_flag_in, p_sum_scale, p_sum_zp);

        if (jcp.dst_scale) {
            mov(reg_ptr_dst_scale, EVEX_compress_addr(rsp, reg_dst_scale_off));
            vmovups(vmm_dst_scale, EVEX_compress_addr(reg_ptr_dst_scale, 0));

            /* Apply dst scale to accumulator */
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                const bool mask_flag
                        = mask_flag_in && i_load == load_loop_blk - 1;
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    const auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                    const Vmm mask_vmm
                            = mask_flag ? r | k_load_dim_mask | T_z : r;
                    vmulps(mask_vmm, r, vmm_dst_scale);
                }
            }
        }

        if (jcp.dst_zero_point) {
            mov(reg_dst_zero_point,
                    EVEX_compress_addr(rsp, reg_dst_zero_point_off));
            vcvtdq2ps(vmm_zp, EVEX_compress_addr(reg_dst_zero_point, 0, true));

            /* Add dst zero_point to accumulator */
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    const auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                    vaddps(r, r, vmm_zp);
                }
            }
        }

        // Properly saturate the accumulators for integer datatypes
        if (one_of(jcp.dst_dt, u8, s8, s32)) {
            init_saturate_f32(vmm_zero, vmm_saturation,
                    reg_ptr_saturation_ubound, f32, jcp.dst_dt);
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                    saturate_cvt_f32(r, vmm_zero, vmm_saturation, jcp.dst_dt);
                }
            }
        }

        if (jcp.dst_dt == data_type::bf16 && !isa_has_bf16(jcp.isa))
            bf16_emu_->init_vcvtneps2bf16();

        // store to the destination
        if (jcp.dst_dt == data_type::bf16 && isa_has_bf16(jcp.isa)) {
            // Optimization: use single store instruction for pair
            // of the nearest vectors along LOAD dimension
            for (int i_ur = 0; i_ur < ur; i_ur++) {
                int i_load = 0;
                for (; i_load < rnd_dn(load_loop_blk, 2); i_load += 2) {
                    auto vmm_dst = vreg_accum(load_loop_blk, i_load, i_ur);
                    auto vmm_dst_next
                            = vreg_accum(load_loop_blk, i_load + 1, i_ur);
                    vcvtne2ps2bf16(vmm_dst, vmm_dst_next, vmm_dst);
                    bool mask_flag
                            = mask_flag_in && i_load + 2 == load_loop_blk;
                    vmovdqu16(output_ptr(i_load, i_ur),
                            maybe_mask_vmm(vmm_dst, mask_flag));
                }
                if (load_loop_blk % 2 != 0) {
                    auto vmm_accum = vreg_accum(load_loop_blk, i_load, i_ur);
                    auto vmm_down = Vmm_down_t(vmm_accum.getIdx());
                    vcvtneps2bf16(vmm_down, vmm_accum);
                    vmovdqu16(output_ptr(i_load, i_ur),
                            maybe_mask_vmm_down(vmm_down,
                                    jcp.ic_block == 4 || mask_flag_in));
                }
            }
        } else {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                const bool mask_flag
                        = mask_flag_in && i_load == load_loop_blk - 1;
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    auto r = vreg_accum(load_loop_blk, i_load, i_ur);
                    const Vmm r_vmm = mask_flag ? r | k_load_dim_mask : r;

                    switch (jcp.dst_dt) {
                        case data_type::f32:
                        case data_type::s32:
                            vmovups(output_ptr(i_load, i_ur), r_vmm);
                            break;
                        case data_type::s8:
                            vpmovsdb(output_ptr(i_load, i_ur), r_vmm);
                            break;
                        case data_type::u8:
                            vpmovusdb(output_ptr(i_load, i_ur), r_vmm);
                            break;
                        case data_type::bf16: {
                            bf16_emu_->vcvtneps2bf16(
                                    ymm_store, Zmm(r.getIdx()));
                            vmovdqu16(output_ptr(i_load, i_ur),
                                    maybe_mask_vmm_down(vmm_store(),
                                            jcp.ic_block == 4 || mask_flag));
                        } break;
                        default: assert(!"unknown dst_dt");
                    }
                }
            }
        }
        mov(reg_bcast_data, EVEX_compress_addr(rsp, reg_bcast_data_off));
        if (is_scale_or_zp_sum)
            mov(reg_load_data, EVEX_compress_addr(rsp, reg_load_data_off));
    };

    auto compute = [this](Vmm vreg_acc, Vmm vreg_wei, Vmm vreg_src) {
        if (jcp.has_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else {
            vpmaddubsw(vmm_tmp, vreg_src, vreg_wei);
            vpmaddwd(vmm_tmp, vmm_tmp, vmm_one);
            vpaddd(vreg_acc, vreg_acc, vmm_tmp);
        }
    };

    auto fma_block = [&](bool last_block) {
        int reduce_step = 4;
        int ic_tail_size = jcp.ic_without_padding % reduce_step;
        int loop_unroll = last_block && jcp.ic != jcp.ic_without_padding
                ? rnd_up(jcp.ic_without_padding % jcp.ic_block, reduce_step)
                : jcp.reduce_loop_unroll;
        for (int i_reduce = 0; i_reduce < loop_unroll;
                i_reduce += reduce_step) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                vmovups(vreg_load(i_load), load_ptr(i_reduce, i_load));
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                if (last_block && ic_tail_size != 0
                        && i_reduce == loop_unroll - reduce_step) {
                    Xmm xmm_bcast = Xmm(vmm_bcast.getIdx());
                    load_bytes(xmm_bcast, aux_reg_bcast_data,
                            jcp.ic_without_padding * i_ur + i_reduce,
                            ic_tail_size);
                    vpbroadcastd(vmm_bcast, xmm_bcast);
                } else {
                    vpbroadcastd(vmm_bcast, bcast_ptr(i_reduce, i_ur, false));
                }
                if (jcp.signed_input) vpsubb(vmm_bcast, vmm_bcast, vmm_shift);
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    compute(vreg_accum(load_loop_blk, i_load, i_ur),
                            vreg_load(i_load), vmm_bcast);
                }
            }
        }
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop);
    {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    if (jcp.ic != jcp.ic_without_padding) {
        fma_block(true);
    } else {
        fma_block(false);
    }

    if (jcp.oc_without_padding != jcp.oc) {
        Label end_store, common_store;
        mov(EVEX_compress_addr(rsp, reg_bcast_data_off), reg_bcast_data);

        /*Check if it is the last load_loop_blk*/
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
        cmp(reg_load_loop_work, 0);
        jg(common_store, T_NEAR);

        /*Check if it is the last ocb*/
        test(reg_reduce_pos_flag, FLAG_OC_LAST);
        jz(common_store, T_NEAR);

        store(true);
        jmp(end_store, T_NEAR);

        L(common_store);
        store(false);

        L(end_store);

        add(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    } else {
        store(false);
    }
}

template <typename Vmm>
void _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Vmm>::generate() {

    preamble();

    const int simd_w = jcp.ic_block;
    xor_(reg_scratch, reg_scratch);
    Reg16 _t = reg_scratch.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(vmm_one, _t);

    sub(rsp, stack_space_needed);
    if (jcp.with_binary)
        mov(EVEX_compress_addr(rsp, reg_abi_param1_backup), abi_param1);

    if (jcp.with_bias) mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    if (jcp.signed_input) {
        mov(EVEX_compress_addr(rsp, reg_bias_data_off), reg_bias_data);
        mov(reg_comp_data, ptr[param1 + GET_OFF(compensation)]);
        mov(EVEX_compress_addr(rsp, reg_comp_data_off), reg_comp_data);
    }
    if (jcp.src_zero_point) {
        mov(reg_zp_compensation, ptr[param1 + GET_OFF(zp_compensation)]);
        mov(EVEX_compress_addr(rsp, reg_zp_compensation_off),
                reg_zp_compensation);
        mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
        mov(EVEX_compress_addr(rsp, reg_src_zero_point_off),
                reg_src_zero_point);
    }
    if (jcp.dst_scale) {
        if (!jcp.signed_input)
            mov(EVEX_compress_addr(rsp, reg_bias_data_off), reg_bias_data);
        mov(reg_ptr_dst_scale, ptr[param1 + GET_OFF(dst_scale)]);
        mov(EVEX_compress_addr(rsp, reg_dst_scale_off), reg_ptr_dst_scale);
    }
    if (jcp.dst_zero_point) {
        mov(reg_dst_zero_point, ptr[param1 + GET_OFF(dst_zero_point)]);
        mov(EVEX_compress_addr(rsp, reg_dst_zero_point_off),
                reg_dst_zero_point);
    }
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
    mov(EVEX_compress_addr(rsp, reg_ptr_sum_scale_off), reg_ptr_scales);
    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(EVEX_compress_addr(rsp, bcast_loop_work_off), reg_bcast_loop_work);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(first_last_flag)]);

    if (jcp.ic_block == 4 && jcp.dst_dt == data_type::bf16) {
        Reg32 reg_tail_32 = reg_load_dim_tail_mask.cvt32();
        mov(reg_tail_32, (1 << jcp.ic_block) - 1);
        kmovb(k_load_dim_tail_mask, reg_tail_32);
    }

    const int load_dim_tail
            = (one_of(jcp.prop_kind, forward_training, forward_inference)
                              ? jcp.oc_without_padding
                              : jcp.load_dim)
            % jcp.load_block;
    const bool use_extended_mask
            = jcp.dst_dt == data_type::bf16 && isa_has_bf16(jcp.isa);
    if (load_dim_tail) {
        Reg32 reg_tail_32 = reg_load_dim_tail_mask.cvt32();
        mov(reg_tail_32, (1 << load_dim_tail) - 1);
        kmovw(k_load_dim_tail_mask, reg_tail_32);
        kmovw(postops_mask, reg_tail_32);

        if (use_extended_mask) {
            mov(reg_tail_32.cvt32(),
                    (1 << (load_dim_tail + jcp.load_block)) - 1);
            kmovd(k_load_dim_tail_mask_extended, reg_tail_32.cvt32());
        }
    } else if (jcp.with_binary)
        if (jcp.oc_block != isa_simd_width_) {
            const int mask = (1 << jcp.oc_block) - 1;
            const Reg32 reg_tail_32 = reg_load_dim_tail_mask.cvt32();
            mov(reg_tail_32, mask);
            kmovw(postops_mask, reg_tail_32);
        }

    auto load_loop_body = [&](int load_loop_blk) {
        if (load_dim_tail) {
            kxnorw(k_load_dim_mask, k_load_dim_mask, k_load_dim_mask);
            if (use_extended_mask)
                kxnord(k_load_dim_mask_extended, k_load_dim_mask_extended,
                        k_load_dim_mask_extended);
            Label no_update_mask;
            test(reg_reduce_pos_flag, FLAG_OC_LAST);
            jz(no_update_mask, T_NEAR);
            cmp(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
            jg(no_update_mask, T_NEAR);
            kmovw(k_load_dim_mask, k_load_dim_tail_mask);
            if (use_extended_mask)
                kmovd(k_load_dim_mask_extended, k_load_dim_tail_mask_extended);
            L(no_update_mask);
        } else if (jcp.ic_block == 4 && jcp.dst_dt == data_type::bf16) {
            kmovw(k_load_dim_mask, k_load_dim_tail_mask);
        }

        bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        if (jcp.with_bias) {
            if (jcp.signed_input || jcp.dst_scale)
                mov(reg_bias_data, EVEX_compress_addr(rsp, reg_bias_data_off));
            add(reg_bias_data,
                    load_loop_blk * jcp.load_block * jcp.typesize_bia);
            if (jcp.signed_input || jcp.dst_scale)
                mov(EVEX_compress_addr(rsp, reg_bias_data_off), reg_bias_data);
        }
        if (jcp.signed_input) {
            mov(reg_comp_data, EVEX_compress_addr(rsp, reg_comp_data_off));
            add(reg_comp_data,
                    load_loop_blk * jcp.load_block * sizeof(int32_t));
            mov(EVEX_compress_addr(rsp, reg_comp_data_off), reg_comp_data);
        }
        if (jcp.src_zero_point) {
            mov(reg_zp_compensation,
                    EVEX_compress_addr(rsp, reg_zp_compensation_off));
            add(reg_zp_compensation,
                    load_loop_blk * jcp.load_block * sizeof(int32_t));
            mov(EVEX_compress_addr(rsp, reg_zp_compensation_off),
                    reg_zp_compensation);
        }
        mov(EVEX_compress_addr(rsp, reg_bcast_data_off), reg_bcast_data);
        mov(reg_ptr_scales, EVEX_compress_addr(rsp, reg_ptr_sum_scale_off));
        add(reg_ptr_scales,
                jcp.is_oc_scale * load_loop_blk * jcp.load_block
                        * sizeof(float));
        mov(EVEX_compress_addr(rsp, reg_ptr_sum_scale_off), reg_ptr_scales);
        mov(reg_bcast_data, EVEX_compress_addr(rsp, reg_bcast_data_off));
        add(reg_output_data, load_loop_blk * jcp.load_block * jcp.typesize_out);
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    };

    Label load_loop_blk[7];

    static const int ur_cases_fma_expl_bcast[] = {2, 5, 6, 9, 14, 32};
    const int size_ur_cases_fma = sizeof(ur_cases_fma_expl_bcast);
    const int *ur_cases_fma = ur_cases_fma_expl_bcast;
    const int *ur_cases = ur_cases_fma;
    const int num_ur_cases = (size_ur_cases_fma) / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            jle(load_loop_blk[label_idx], T_NEAR);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = num_ur_cases - ur_idx - 1;
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    cmp(reg_load_loop_work, 0);
                    je(load_loop_blk[num_ur_cases], T_NEAR);
                }

                for (int _i = 1; _i <= label_idx + 1; _i++) {
                    prefetcht0(ptr[reg_load_data + _i * jcp.ic * jcp.oc_block]);
                    prefetcht1(ptr[reg_output_data + _i * jcp.oc_block]);
                }

                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    cmp(reg_load_loop_work, 2 * label_idx * simd_w);
                    je(load_loop_blk[label_idx - 1], T_NEAR);
                }
                cmp(reg_load_loop_work, (label_idx + 1) * simd_w);
                jge(load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                cmp(reg_load_loop_work, simd_w * (idx + 1));
                je(load_loop_blk[idx], T_NEAR);
            }
            if (ur_idx < num_ur_cases - 2) {
                cmp(reg_load_loop_work, simd_w);
                jle(load_loop_blk[0], T_NEAR);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    add(rsp, stack_space_needed);

    postamble();

    if (jcp.with_eltwise)
        postops_injector_->prepare_table(/* generate = */ true);
}

status_t jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_t *&src_md, memory_desc_t &weights_md,
        memory_desc_t &dst_md, memory_desc_t &bias_md,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {

    if (!mayiuse(avx512_core)) return status::unimplemented;

    // used for bf16 output
    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                        : bf16_emulation_t::get_isa();

    const memory_desc_wrapper src_d(src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!one_of(src_d.data_type(), data_type::u8, data_type::s8)
            || weights_d.data_type() != data_type::s8
            || !one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8, data_type::bf16))
        return status::unimplemented;

    jcp.nthr = nthreads;

    jcp.has_vnni = mayiuse(avx512_core_vnni);

    int ndims = src_d.ndims();
    jcp.ndims = ndims;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    const bool is_1d = ndims == 3;
    const bool is_3d = ndims == 5;

    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    jcp.signed_input = (src_d.data_type() == data_type::s8);

    jcp.os = static_cast<dim_t>(jcp.od) * jcp.oh * jcp.ow;
    jcp.is = static_cast<dim_t>(jcp.id) * jcp.ih * jcp.iw;

    if (jcp.os > INT_MAX || jcp.is > INT_MAX) return status::unimplemented;

    const auto &post_ops = attr.post_ops_;
    const int dw_conv_ind = post_ops.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;
    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.
    const int eltwise_ind
            = post_ops.find(primitive_kind::eltwise, 0, dw_conv_ind);
    jcp.with_eltwise = eltwise_ind != -1;

    const int binary_ind
            = post_ops.find(primitive_kind::binary, 0, dw_conv_ind);
    const int prelu_ind = post_ops.find(primitive_kind::prelu, 0, dw_conv_ind);
    jcp.with_binary = !everyone_is(-1, binary_ind, prelu_ind);

    const int sum_ind = post_ops.find(primitive_kind::sum, 0, dw_conv_ind);
    jcp.with_sum = sum_ind != -1;

    if (dw_conv_ind >= 0) {
        // dw_conv and post_ops after it are handled externally, so skip them
        jcp.post_ops.entry_.assign(post_ops.entry_.cbegin(),
                post_ops.entry_.cbegin() + dw_conv_ind);
    } else {
        jcp.post_ops = post_ops;
    }

    const auto zp = attr.zero_points_;
    jcp.dst_zero_point = !zp.has_default_values(DNNL_ARG_DST);
    jcp.src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
    jcp.zp_src_is_common
            = zp.get_mask(DNNL_ARG_SRC) == 0; // otherwise, it's per-channel
    assert(IMPLICATION(jcp.src_zero_point, jcp.zp_src_is_common));

    if ((jcp.dst_zero_point || jcp.src_zero_point) && jcp.with_dw_conv)
        return status::unimplemented;

    format_tag_t dat_tag = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);

    bool args_ok = jcp.src_tag == dat_tag && jcp.dst_tag == dat_tag;
    if (!args_ok) return status::unimplemented;

    if (jcp.ngroups == 1) {
        jcp.oc = rnd_up(jcp.oc, 16);
        jcp.ic = rnd_up(jcp.ic, 16);
    }

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = false;
    static constexpr bool sum_requires_scale_one = false;
    static constexpr bool sum_requires_zp_zero = false;
    const bool post_ops_ok_ = post_ops_ok(post_ops_ok_args_t(avx512_core,
            {eltwise, binary, sum}, jcp.post_ops, &dst_d, sum_at_pos_0_only,
            sum_requires_scale_one, sum_requires_zp_zero));
    if (!post_ops_ok_) return status::unimplemented;

    const int simd_w = (jcp.ic % 16 == 0 && jcp.oc % 16 == 0) ? 16
            : (jcp.ic % 8 == 0 && jcp.oc % 8 == 0)            ? 8
                                                              : 4;

    auto set_or_check_wei_format = [&]() -> bool {
        using namespace format_tag;
        using namespace memory_extra_flags;
        const format_tag_t wei_tags[3][2][3]
                = {{{OIw4i16o4i, OIhw4i16o4i, OIdhw4i16o4i},
                           {gOIw4i16o4i, gOIhw4i16o4i, gOIdhw4i16o4i}},
                        {{OIw2i8o4i, OIhw2i8o4i, OIdhw2i8o4i},
                                {gOIw2i8o4i, gOIhw2i8o4i, gOIdhw2i8o4i}},
                        {{OIw4o4i, OIhw4o4i, OIdhw4o4i},
                                {gOIw4o4i, gOIhw4o4i, gOIdhw4o4i}}};

        const int simd_idx = simd_w == 16 ? 0 : simd_w == 8 ? 1 : 2;
        const auto wei_tag = wei_tags[simd_idx][with_groups][ndims - 3];
        memory_desc_t want_wei_md = weights_md;
        CHECK_BOOL(memory_desc_init_by_tag(want_wei_md, wei_tag));

        if (jcp.signed_input) {
            want_wei_md.extra.flags = 0 | compensation_conv_s8s8 | scale_adjust;
            want_wei_md.extra.compensation_mask
                    = (1 << 0) + (with_groups ? (1 << 1) : 0);
            want_wei_md.extra.scale_adjust
                    = mayiuse(avx512_core_vnni) ? 1.f : 0.5f;
        }
        if (jcp.src_zero_point) {
            want_wei_md.extra.flags |= compensation_conv_asymmetric_src;
            want_wei_md.extra.asymm_compensation_mask
                    = (1 << 0) + (with_groups ? (1 << 1) : 0);
        }

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) return status::unimplemented;

    args_ok = true && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
            && jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0
            && jcp.stride_d == 1 && jcp.stride_h == 1
            && jcp.stride_w == 1 // TODO: support some strides
            && jcp.od == jcp.id && jcp.oh == jcp.ih
            && jcp.ow == jcp.iw // enforce rpad = 0
            && jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.sum_dt = post_ops.get_sum_dt(jcp.dst_dt);

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    const int SMALL_SPATIAL = 7 * 7;
    const int BIG_REDUCE_DIM = 1024;

    int load_blocking = 0;
    int load_blocking_max = 0;
    int bcast_blocking = 0;
    int bcast_blocking_max = 0;
    int reduce_blocking = 0;
    int reduce_blocking_max = 0;
    jcp.load_grp_count = 1;
    jcp.use_vmovntps = false;

    const int L2_size
            = platform::get_per_core_cache_size(2) / sizeof(jcp.typesize_in);
    const int L2_capacity = (L2_size * 3) / 4;

    const bool req_extra_bf16_regs
            = jcp.dst_dt == data_type::bf16 && !isa_has_bf16(jcp.isa);
    int size_treshold = req_extra_bf16_regs ? 25 : 28;
    int max_regs = 0;
    int min_regs = 6;
    if (jcp.has_vnni && !req_extra_bf16_regs)
        max_regs = ((jcp.oh > size_treshold && jcp.ow > size_treshold)
                           && (jcp.oc < 128 || jcp.ic < 128))
                ? min_regs
                : 9;
    else
        max_regs = 8;
    jcp.expl_bcast = true;

    if (jcp.mb == 1 && jcp.ic > 128
            && (jcp.oh <= size_treshold && jcp.ow <= size_treshold)) {
        if (jcp.os <= SMALL_SPATIAL && jcp.oc * jcp.ic < L2_size)
            max_regs = min_regs; // mobilenet_v2 performance improvement
        jcp.ur = nstl::min<dim_t>(max_regs, jcp.os);
    } else {
        const int spatial = jcp.od * jcp.oh;
        jcp.ur = 1;
        for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
            if ((spatial >= size_treshold && spatial % ur_w == 0)
                    || (spatial < size_treshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = nstl::min<dim_t>(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i--) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
    }
    if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);

    jcp.reduce_dim = jcp.ic;
    jcp.reduce_block = jcp.ic_block;

    jcp.load_dim = jcp.oc;
    jcp.load_block = jcp.oc_block;

    jcp.bcast_dim = jcp.is;

    jcp.bcast_block = jcp.ur;

    jcp.reduce_loop_unroll = jcp.reduce_block;
    jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll * jcp.typesize_in;

    jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;

    jcp.bcast_loop_output_step
            = jcp.ur * jcp.ngroups * jcp.oc_without_padding * jcp.typesize_out;
    jcp.bcast_loop_output_substep = -1; // unused
    jcp.bcast_loop_bcast_step
            = jcp.ur * jcp.ngroups * jcp.ic_without_padding * jcp.typesize_in;
    jcp.bcast_loop_bcast_substep = -1; // unused

    jcp.load_loop_load_step = jcp.reduce_dim * jcp.load_block * jcp.typesize_in;

    jcp.load_loop_iter_step = jcp.load_block;

    jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

    int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    reduce_blocking = nb_reduce;
    if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 64;
    else if (jcp.bcast_dim > SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 16;
    reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
    reduce_blocking *= jcp.reduce_block;

    bool cmp_reduce = reduce_blocking <= jcp.reduce_dim;
    if (cmp_reduce) jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
    load_blocking = jcp.load_dim;

    jcp.load_grp_count = div_up(jcp.nthr, jcp.mb * jcp.ngroups * nb_bcast);
    jcp.load_grp_count = best_divider(
            jcp.nthr, jcp.load_grp_count, 2 * jcp.load_grp_count, false);

    if (jcp.bcast_dim <= SMALL_SPATIAL
            && jcp.load_dim * jcp.reduce_dim >= L2_size) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 4);
    } else if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.mb <= jcp.nthr
            && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2); //
        load_blocking = jcp.load_block;
    }

    bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                             div_up(jcp.nthr, jcp.load_grp_count))
            * jcp.bcast_block;
    bcast_blocking = nstl::min<dim_t>(jcp.bcast_dim, bcast_blocking);
    bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

    int space_for_bcast = (L2_capacity - /* kernel_size - */
            2 * jcp.load_block * reduce_blocking - jcp.ur * reduce_blocking
            - 3 * 1024);
    if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity) space_for_bcast /= 2;

    int bcast_in_cache
            = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
    bcast_blocking = nstl::min(
            bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

    load_blocking_max = load_blocking;
    bcast_blocking_max = bcast_blocking * 3 / 2;
    reduce_blocking_max = reduce_blocking;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);
    assert(load_blocking % jcp.load_block == 0);
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);

    assert(jcp.reduce_loop_unroll % 4 == 0);
    assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);

    assert(jcp.bcast_block % jcp.ur == 0);
    assert(jcp.reduce_dim % jcp.reduce_block == 0);

    jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;
    jcp.nb_reduce_blocking_max = reduce_blocking_max / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    // miniumum size of load dim chunk for work distribution within threads
    jcp.nb_load_chunk = 1;
    // peformance improvements for googlenet_v3, mb=1;
    // TODO: generalize this condition and rewrite it in appropriate manner
    int ncores_per_socket = (int)cpu().getNumCores(
            Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    if (jcp.mb == 1 && jcp.nb_load % 4 == 0 && jcp.ic / jcp.oc >= 4
            && jcp.ic * jcp.oc <= L2_size && jcp.nthr <= ncores_per_socket) {
        jcp.nb_load_chunk = 4;
        jcp.load_grp_count = nstl::max(jcp.nb_load / 4, jcp.load_grp_count);
    }

    /* adjust the thread decomposition
     * to improve the perf for small size problem
     * the threshold 8192 is empirical
     * simply set the thread to max of nb_load and nb_bcast now
     * TODO: add get_thr_eff func to compute optimal thread
     * TODO: Threshold can be increase when init stride > 1 */
    auto bcast_size
            = (dim_t)jcp.mb * jcp.ngroups * jcp.bcast_dim * jcp.reduce_dim;
    if (jcp.typesize_in * bcast_size < 8192 && jcp.ngroups < jcp.nthr
            && jcp.nb_bcast * jcp.nb_load < jcp.nthr) {
        int nthr = nstl::max(jcp.nb_load, jcp.nb_bcast);
        jcp.nthr = nstl::min(jcp.nthr, nthr);
    }

    const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
    const auto &dst_scales = attr.scales_.get(DNNL_ARG_DST);
    jcp.is_oc_scale = wei_scales.get_mask() > 0;
    jcp.dst_scale = !dst_scales.has_default_values();

    jcp.wei_adj_scale
            = (weights_d.extra().flags & memory_extra_flags::scale_adjust)
            ? weights_d.extra().scale_adjust
            : 1.f;

    return status::success;
}

void jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace dnnl::impl::memory_tracking::names;

    const int wei_mask = attr.scales_.get_mask(DNNL_ARG_WEIGHTS);
    const dim_t scales_count
            = wei_mask == 0 ? 1 : static_cast<dim_t>(jcp.oc) * jcp.ngroups;
    const dim_t count = nstl::max<dim_t>(scales_count, (dim_t)jcp.ic_block);
    scratchpad.book<float>(key_conv_adjusted_scales, count);
}

template struct _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Xbyak::Zmm>;
template struct _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Xbyak::Ymm>;
template struct _jit_avx512_core_x8s8s32x_1x1_conv_kernel<Xbyak::Xmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
