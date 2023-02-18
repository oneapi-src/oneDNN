/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2021-2023 FUJITSU LIMITED
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
#include "common/memory.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/injectors/injector_utils.hpp"
#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/jit_sve_512_x8s8s32x_conv_kernel.hpp"

#define GET_OFF(field) static_cast<int32_t>(offsetof(jit_conv_call_s, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::data_type;

namespace {
void pick_loop_order(jit_conv_conf_t &jcp, int nthr) {
    jcp.loop_order = loop_cwgn;
    if (jcp.ngroups > 1) {
        jcp.loop_order = loop_ngcw;
        if (jcp.mb < nthr)
            jcp.loop_order = jcp.ndims == 3 ? loop_nwcg : loop_nhwcg;
    } else if (jcp.mb >= nthr && jcp.ic_without_padding <= 16) {
        jcp.loop_order = loop_ngcw;
    }
}
} // namespace

jit_sve_512_x8s8s32x_fwd_kernel::jit_sve_512_x8s8s32x_fwd_kernel(
        const jit_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jcp(ajcp), attr_(attr) {
    int ch_block = jcp.is_depthwise ? jcp.ch_block : jcp.ic_block;
    switch (ch_block) {
        case 16: sve_len_ = 64; break;
        case 8: sve_len_ = 32; break;
        case 4: sve_len_ = 16; break;
        default: assert(!"unreachable"); break;
    }
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr size_t helper_vmm_idx = 31;
        const size_t oc_block_tail = jcp.oc_block % isa_simd_width_;
        const size_t tail_size = oc_block_tail
                ? oc_block_tail
                : jcp.oc_without_padding % isa_simd_width_;
        static constexpr bool use_exact_tail_scalar_bcast = false;

        const rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx,
                x14, x15, x13, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(dst_orig),
                memory_desc_wrapper(dst_md), tail_size, postops_mask,
                use_exact_tail_scalar_bcast};
        const static_params_t static_params {
                this->param1, rhs_arg_static_params};

        const eltwise_injector::static_params_t eltwise_params {
                true, x14, mask_tmp, mask_tmp2};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<sve_512>>(
                this, jcp.post_ops, static_params, eltwise_params);
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::prepare_output(int ur_w) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    for (int k = 0; k < nb_oc_block; k++)
        for (int j = 0; j < ur_w; j++) {
            auto vmm = vmm_out(j, k);
            eor(vmm.d, vmm.d, vmm.d);
        }
    if (!jcp.signed_input) {
        eor(reg_scratch, reg_scratch, reg_scratch);
        if (jcp.is_depthwise && !jcp.is_fast_depthwise) {
            mov_imm(WReg(reg_tmp0_imm.getIdx()), 128);
            dup(vmm_shift.s, WReg(reg_tmp0_imm.getIdx()));
        } else {
            dup(vmm_shift.b, -128);
        }
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::cvt2ps(data_type_t type_in,
        const ZReg vmm_in, const XReg reg_base, const int offset,
        bool mask_flag) {

    auto vmm = vmm_in;
    auto reg_addr = get_comp_addr_reg(reg_base, offset);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            if (mask_flag)
                ld1w(vmm.s, ktail_mask / T_z, ptr(reg_addr));
            else {
                // Because reg_addr maybe disaligned,
                // `ldr(vmm, ptr(reg_addr))` can't be used.
                ld1w(vmm.s, mask_all_one / T_z, ptr(reg_addr));
            }
            break;
        case data_type::s8:
            sub(reg_stack, reg_stack, 64);
            str(vmm_tmp, ptr(reg_stack));
            vmm_load_src(vmm_tmp, reg_addr, mask_flag);
            zip1(vmm_tmp.b, vmm_tmp.b, vmm_tmp.b);
            zip1(vmm_tmp.h, vmm_tmp.h, vmm_tmp.h);
            sxtb(vmm.s, mask_all_one / T_m, vmm_tmp.s);
            if (mask_flag) {
                not_(mask_tmp.b, mask_all_one.b, ktail_mask.b);
                mov(vmm.s, mask_tmp / T_m, 0);
            }
            ldr(vmm_tmp, ptr(reg_stack));
            add(reg_stack, reg_stack, 64);
            break;
        case data_type::u8:
            sub(reg_stack, reg_stack, 64);
            str(vmm_tmp, ptr(reg_stack));
            vmm_load_src(vmm_tmp, reg_addr, mask_flag);
            zip1(vmm_tmp.b, vmm_tmp.b, vmm_tmp.b);
            zip1(vmm_tmp.h, vmm_tmp.h, vmm_tmp.h);
            uxtb(vmm.s, mask_all_one / T_m, vmm_tmp.s);
            if (mask_flag) {
                not_(mask_tmp.b, mask_all_one.b, ktail_mask.b);
                mov(vmm.s, mask_tmp / T_m, 0);
            }
            ldr(vmm_tmp, ptr(reg_stack));
            add(reg_stack, reg_stack, 64);
            break;
        default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32) scvtf(vmm_in.s, mask_all_one, vmm_in.s);
}

template <typename F>
static void iterate(const int nb_oc_block, const int ur_w,
        const bool last_oc_block_flag, const bool force_masking, const F &f) {
    for (int k = 0; k < nb_oc_block; k++) {
        const bool mask_flag
                = force_masking || (last_oc_block_flag && k + 1 == nb_oc_block);
        for (int j = 0; j < ur_w; j++)
            f(mask_flag, k, j);
    }
}
template <typename F>
static void iterate(const int nb_oc_block, const int ur_w,
        const bool last_oc_block_flag, const F &f) {
    iterate(nb_oc_block, ur_w, last_oc_block_flag, false, f);
}
template <typename F>
static void iterate(const int nb_oc_block, const int ur_w, const F &f) {
    iterate(nb_oc_block, ur_w, false, false, f);
}

void jit_sve_512_x8s8s32x_fwd_kernel::apply_sum(int ur_w,
        bool last_oc_block_flag, const int nb_oc_block, const int oc_block,
        const float *p_sum_scale, const int32_t *p_sum_zp) {
    if (jcp.with_sum) {
        const float sum_scale = *p_sum_scale;
        const int32_t sum_zp = *p_sum_zp;
        const auto sum_injector_lam = [this, oc_block, sum_scale, sum_zp](
                                              const bool mask_flag, const int k,
                                              const int j) {
            int aux_output_offset = jcp.typesize_out
                    * (k * oc_block + j * jcp.oc_without_padding * jcp.ngroups);
            auto vmm = vmm_out(j, k);
            cvt2ps(jcp.sum_dt, vmm_prev_dst, reg_out, aux_output_offset,
                    mask_flag);
            if (sum_zp != 0) {
                fsub(vmm_prev_dst.s, vmm_prev_dst.s, vmm_sum_zp.s);
            }
            if (sum_scale == 1.f) {
                fadd(vmm.s, vmm.s, vmm_prev_dst.s);
            } else {
                ld1rw(vmm_tmp.s, mask_all_one / T_z, ptr(reg_ptr_sum_scale));
                fmla(vmm.s, mask_all_one / T_m, vmm_prev_dst.s, vmm_tmp.s);
            }
        };
        const auto sum_injector = [=]() {
            iterate(nb_oc_block, ur_w, last_oc_block_flag, sum_injector_lam);
        };
        if (sum_scale != 1.f) {
            mov_imm(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));
        }
        if (sum_zp != 0) {
            mov_imm(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
            ld1rb(vmm_tmp.s, mask_all_one / T_m, ptr(reg_ptr_sum_zp));
            scvtf(vmm_sum_zp.s, mask_all_one / T_m, vmm_tmp.s);
        }
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::apply_postops(int ur_w,
        bool last_oc_block_flag, const int nb_oc_block, const int oc_block,
        const float *p_sum_scale, const int32_t *p_sum_zp) {
    if (jcp.with_eltwise || jcp.with_binary || jcp.with_sum) {
        apply_sum(ur_w, last_oc_block_flag, nb_oc_block, oc_block, p_sum_scale,
                p_sum_zp);

        injector_utils::vmm_index_set_t vmm_idxs;
        if (jcp.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
            const bool oc_blk_is_smaller_than_vmm = oc_block < isa_simd_width_;
            iterate(nb_oc_block, ur_w, last_oc_block_flag,
                    oc_blk_is_smaller_than_vmm,
                    [&](const bool mask_flag, const int k, const int j) {
                        const size_t aux_output_l_off = jcp.typesize_out
                                * (k * oc_block
                                        + j * jcp.oc_without_padding
                                                * jcp.ngroups);
                        const auto vmm_idx = vmm_out_idx(j, k);
                        vmm_idxs.emplace(vmm_idx);

                        rhs_arg_params.vmm_idx_to_out_reg.emplace(
                                vmm_idx, reg_out);
                        rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                                vmm_idx, aux_output_l_off);
                        if (mask_flag)
                            rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
                    });

            postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params);
        } else {
            iterate(nb_oc_block, ur_w,
                    [&](const bool, const int k, const int j) {
                        vmm_idxs.emplace(vmm_out_idx(j, k));
                    });
            postops_injector_->compute_vector_range(vmm_idxs);
        }
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::store_output(
        int ur_w, bool last_oc_block_flag) {
    int nb_oc_block
            = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
    int oc_block = jcp.is_depthwise ? jcp.ch_block : jcp.oc_block;

    ldr(reg_bias, ptr(reg_param1, GET_OFF(bias)));
    ldr(reg_ptr_scales, ptr(reg_param1, GET_OFF(scales)));
    if (!jcp.signed_input)
        ldr(reg_compensation, ptr(reg_param1, GET_OFF(compensation)));

    if (jcp.src_zero_point) {
        ldr(reg_zp_compensation, ptr(reg_param1, GET_OFF(zp_compensation)));
        ldr(reg_src_zero_point, ptr(reg_param1, GET_OFF(src_zero_point)));
    }

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = nullptr;
    const int32_t *p_sum_zp = nullptr;
    if (sum_idx != -1) {
        const auto &p_entry = p.entry_[sum_idx];
        p_sum_scale = &p_entry.sum.scale;
        p_sum_zp = &p_entry.sum.zero_point;
    }

    for (int k = 0; k < nb_oc_block; k++) {
        const bool mask_flag
                = last_oc_block_flag && k == nb_oc_block - 1 && mask_gflag;
        PReg mask = mask_flag ? ktail_mask : mask_all_one;
        int scale_offset = jcp.is_oc_scale * (sizeof(float) * k * oc_block);
        if (jcp.with_bias) {
            int bias_offset = jcp.typesize_bia * k * oc_block;

            cvt2ps(jcp.bia_dt, vmm_bias, reg_bias, bias_offset, mask_flag);
        }
        if (!jcp.signed_input) {
            int comp_offset = sizeof(int32_t) * k * oc_block;

            auto reg_addr = get_comp_addr_reg(reg_compensation, comp_offset);
            ld1w(vmm_comp.s, mask / T_z, ptr(reg_addr));
        }
        if (jcp.src_zero_point) {
            // zero_point: conv(src_x8, wei_s8) - src_shift_s32 * compensation_s32
            int zp_offset = sizeof(int32_t) * k * oc_block;
            auto reg_addr = get_comp_addr_reg(reg_zp_compensation, zp_offset);
            ld1w(vmm_zp.s, mask / T_z, ptr(reg_addr));
            auto reg_addr2 = get_comp_addr_reg(reg_src_zero_point, 0);
            st1w(vmm_tmp.s, mask_all_one / T_z, ptr(reg_stack, -1, MUL_VL));
            vmm_load_zero_point(vmm_tmp, reg_addr2, mask, jcp.zp_src_is_common);
            mul(vmm_zp.s, mask / T_m, vmm_tmp.s);
            ld1w(vmm_tmp.s, mask_all_one / T_z, ptr(reg_stack, -1, MUL_VL));
        }
        /* optimization under specific conditions: preload scale_offset data */
        if ((!jcp.is_fast_depthwise && jcp.signed_input
                    && !jcp.src_zero_point)) {
            auto reg_addr = get_comp_addr_reg(reg_ptr_scales, scale_offset);
            ld1w(vmm_pre_load.s, mask / T_z, ptr(reg_addr));
        }
        /* add to accum: compensation, bias and permute */
        for (int j = 0; j < ur_w; j++) {
            auto vmm = vmm_out(j, k);
            if (jcp.is_fast_depthwise) {
                auto zmm = zmm_out(j, k);
                auto zmm_tmp1 = ZReg(31);
                auto zmm_tmp2 = ZReg(30);
                auto zmm_tmp3 = ZReg(29);
                sub(reg_stack, reg_stack, 64);
                str(zmm_tmp1, ptr(reg_stack));
                sub(reg_stack, reg_stack, 64);
                str(zmm_tmp2, ptr(reg_stack));
                sub(reg_stack, reg_stack, 64);
                str(zmm_tmp3, ptr(reg_stack));
                mov(zmm_tmp1.s, 15);
                and_(zmm_tmp1.b, mask_all_one, zmm_permute.b);
                for (int i = 0; i < 16; i++) {
                    cmpeq(mask_tmp.s, mask_all_one, zmm_tmp1.s, i);
                    dup(zmm_tmp2.s, zmm.s[i]);
                    mov(zmm_tmp3.s, mask_tmp / T_m, zmm_tmp2.s);
                }
                mov(zmm.d, zmm_tmp3.d);
                ldr(zmm_tmp3, ptr(reg_stack));
                add(reg_stack, reg_stack, 64);
                ldr(zmm_tmp2, ptr(reg_stack));
                add(reg_stack, reg_stack, 64);
                ldr(zmm_tmp1, ptr(reg_stack));
                add(reg_stack, reg_stack, 64);
            }
            /* add comp in s32 to avoid loss of precision
               when convert s32 to f32 in integer(2^24)
               TODO: do the same to bias */
            if (!jcp.signed_input) sub(vmm.s, vmm.s, vmm_comp.s);
            if (jcp.src_zero_point) add(vmm.s, vmm.s, vmm_zp.s);
            scvtf(vmm.s, mask_all_one, vmm.s);
            if ((!jcp.is_fast_depthwise && jcp.signed_input
                        && !jcp.src_zero_point)) {
                /* optimization under specific conditions: optimize using preloaded scale_offset data */
                fmul(vmm.s, vmm.s, vmm_pre_load.s);
            } else {
                auto reg_addr = get_comp_addr_reg(reg_ptr_scales, scale_offset);
                st1w(vmm_tmp.s, mask_all_one / T_z, ptr(reg_stack, -1, MUL_VL));
                ld1w(vmm_tmp.s, mask / T_z, ptr(reg_addr));
                fmul(vmm.s, vmm.s, vmm_tmp.s);
                ld1w(vmm_tmp.s, mask_all_one / T_z, ptr(reg_stack, -1, MUL_VL));
            }
            if (jcp.with_bias) fadd(vmm.s, vmm.s, vmm_bias.s);
        }
    }

    apply_postops(ur_w, last_oc_block_flag, nb_oc_block, oc_block, p_sum_scale,
            p_sum_zp);

    if (jcp.dst_scale) {
        ldr(reg_dst_scale, ptr(reg_param1, GET_OFF(dst_scale)));
        auto reg_addr = get_comp_addr_reg(reg_dst_scale, 0);
        ld1w(vmm_dst_scale.s, mask_all_one, ptr(reg_addr));

        /* Apply dst scale to accumulator */
        for (int k = 0; k < nb_oc_block; k++) {
            const bool mask_flag = last_oc_block_flag && k == nb_oc_block - 1;
            for (int j = 0; j < ur_w; j++) {
                ZReg vmm = vmm_out(j, k);
                fmul(vmm.s, vmm.s, vmm_dst_scale.s);
                if (mask_flag) mov(vmm.s, ktail_mask / T_m, vmm.s);
            }
        }
    }

    if (jcp.dst_zero_point) {
        ldr(reg_dst_zero_point, ptr(reg_param1, GET_OFF(dst_zero_point)));
        auto reg_addr = get_comp_addr_reg(reg_dst_zero_point, 0);
        vmm_load_zero_point(vmm_zp, reg_addr, mask_all_one, true);
        scvtf(vmm_zp.s, mask_all_one / T_m, vmm_zp.s);

        /* Add dst zero_point to accumulator */
        for (int k = 0; k < nb_oc_block; k++) {
            for (int j = 0; j < ur_w; j++) {
                ZReg vmm = vmm_out(j, k);
                fadd(vmm.s, vmm.s, vmm_zp.s);
            }
        }
    }

    // Properly saturate the accumulators for integer datatypes
    if (one_of(jcp.dst_dt, data_type::u8, data_type::s8, data_type::s32)) {
        if (jcp.dst_dt == data_type::u8) {
            eor(vmm_zero.d, vmm_zero.d, vmm_zero.d);
        }
        float saturation_ubound = types::max_value<float>(jcp.dst_dt);
        mov_imm(aux_reg_saturation, float2int(saturation_ubound));
        dup(vmm_saturation.s, WReg(aux_reg_saturation.getIdx()));

        for (int k = 0; k < nb_oc_block; k++) {
            for (int j = 0; j < ur_w; j++) {
                auto vmm = vmm_out(j, k);
                if (jcp.dst_dt == data_type::u8) {
                    fmaxnm(vmm.s, mask_all_one, vmm_zero.s);
                    fmax(vmm.s, mask_all_one, vmm_zero.s);
                }
                fminnm(vmm.s, mask_all_one, vmm_saturation.s);
                fmin(vmm.s, mask_all_one, vmm_saturation.s);

                frintn(vmm.s, mask_all_one, vmm.s);
                fcvtzs(vmm.s, mask_all_one, vmm.s);
            }
        }
    }

    /* write out register to output_addr */
    for (int k = 0; k < nb_oc_block; k++) {
        const bool mask_flag
                = last_oc_block_flag && k == nb_oc_block - 1 && mask_gflag;
        for (int j = 0; j < ur_w; j++) {
            int aux_output_offset = jcp.typesize_out
                    * (k * oc_block + j * jcp.oc_without_padding * jcp.ngroups);

            auto base = reg_out;
            auto re = get_offset(aux_output_offset);

            auto reg_tmp_adr = ((j % 4) == 0) ? reg_tmp0_adr
                    : ((j % 4) == 1)          ? reg_tmp1_adr
                    : ((j % 4) == 2)          ? reg_tmp2_adr
                                              : reg_tmp3_adr;
            add_imm(reg_tmp_adr, base, re, reg_tmp0_imm);

            auto vmm = vmm_out(j, k);

            auto _mask = mask_flag ? ktail_mask : mask_all_one;
            switch (jcp.dst_dt) {
                case data_type::f32:
                case data_type::s32:
                    st1w(vmm.s, _mask, ptr(reg_tmp_adr));
                    break;
                case data_type::s8:
                    smin(vmm.s, 127);
                    smax(vmm.s, -128);
                    st1b(vmm.s, _mask, ptr(reg_tmp_adr));
                    break;
                case data_type::u8:
                    umin(vmm.s, 255);
                    st1b(vmm.s, _mask, ptr(reg_tmp_adr));
                    break;
                default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::compute_ker_dw(int ur_w, int pad_l,
        int pad_r, ic_block_t last_ic_block_flag, bool h_padded) {

    if (sve_len_ != 64)
        assert(!"invalid group blocking for depthwise convolution");

    const bool compute_kernel = IMPLICATION(h_padded, !jcp.signed_input);

    if (jcp.src_zero_point) {
        sub(reg_stack, reg_stack, 8);
        str(aux_reg_ker_d, ptr(reg_stack));
        auto reg_addr = get_comp_addr_reg(reg_param1, GET_OFF(src_zero_point));
        ldr(reg_src_zero_point, ptr(reg_addr));
    }

    auto input_spatial_index = [=](int oi, int ki) {
        return (ki * (jcp.dilate_w + 1) + oi * jcp.stride_w - pad_l);
    };

    auto input_offset2 = [=](int ii, int ci) {
        if (jcp.is_fused_conv)
            return jcp.typesize_in
                    * (ii * jcp.dw_conv_buffer_oc + ci * jcp.ch_block);
        else
            return jcp.typesize_in * (ii * jcp.ngroups + ci * jcp.ch_block);
    };

    auto input_offset3 = [=](int oi, int ci, int ki) {
        return jcp.typesize_in * input_offset2(input_spatial_index(oi, ki), ci);
    };

    auto kernel_offset = [=](int ci, int ki) {
        return jcp.typesize_in * ((ci * jcp.kh * jcp.kw + ki) * jcp.ch_block);
    };

    auto compute = [=](ZReg vreg_acc, ZReg vreg_wei, ZReg vreg_src) {
        // okay for depthwise since src is zero-extended
        sdot(vreg_acc.s, vreg_src.b, vreg_wei.b);
    };

    int ii_start = 0;
    int ii_end = -1;
    if (jcp.is_resrc_depthwise && !h_padded) {
        // find bounds of input spatial indices
        bool first = true;
        for (int ki = 0; ki < jcp.kw; ki++) {
            int oi_start = get_ow_start(ki, pad_l);
            int oi_end = get_ow_end(ur_w, ki, pad_r);
            for (int oi = oi_start; oi < oi_end; oi++) {
                int ii = input_spatial_index(oi, ki);
                if (first || ii < ii_start) ii_start = ii;
                if (first || ii > ii_end) ii_end = ii;
                first = false;
            }
        }
    }

    if (!jcp.signed_input) {
        eor(zmm_shifted_zero.d, zmm_shifted_zero.d, zmm_shifted_zero.d);
        sub(zmm_shifted_zero.b, zmm_shifted_zero.b, vmm_shift.b);
    }

    for (int ci = 0; ci < jcp.nb_ch_blocking; ci++) {
        const bool mask_flag = last_ic_block_flag != no_last_block
                && ci == jcp.nb_ch_blocking - 1;
        if (jcp.is_resrc_depthwise && !h_padded) {
            // now we can load input once and reuse up to jcp.kw times
            for (int ii = ii_start; ii <= ii_end; ii++) {
                int aux_input_offset = input_offset2(ii, ci);
                auto zmm_inp_tmp = zmm_inp(ii, jcp.nb_ch_blocking);
                auto zmm_inp_msk = zmm_inp_tmp;
                if (jcp.is_fast_depthwise) {
                    assert(!mask_flag);
                    if (ii == ii_start)
                        not_(mask_tmp.b, mask_all_one, ktail_mask.b);
                    if (aux_input_offset % 16 == 0
                            && aux_input_offset <= 65520) {
                        ldr(QReg(zmm_inp_tmp.getIdx()),
                                ptr(aux_reg_inp, aux_input_offset));
                    } else {
                        auto reg_addr = get_comp_addr_reg(
                                aux_reg_inp, aux_input_offset);
                        ldr(QReg(zmm_inp_tmp.getIdx()), ptr(reg_addr));
                    }
                    dup(zmm_inp_tmp.q, zmm_inp_tmp.q[0]);
                    sel(zmm_inp_tmp.s, mask_tmp, zmm_inp_tmp.s, vmm_zero.s);
                } else {
                    auto reg_addr
                            = get_comp_addr_reg(aux_reg_inp, aux_input_offset);
                    auto zmm_tmp = ZReg(31);
                    sub(reg_stack, reg_stack, 64);
                    str(zmm_tmp, ptr(reg_stack));
                    if (mask_flag) {
                        eor(mask_tmp.b, mask_all_one, mask_tmp.b, mask_tmp.b);
                        eor(mask_tmp2.b, mask_all_one, mask_tmp2.b,
                                mask_tmp2.b);
                        uzp1(mask_tmp.h, ktail_mask.h, mask_tmp.h);
                        uzp1(mask_tmp.b, mask_tmp.b, mask_tmp2.b);
                    } else {
                        ptrue(mask_tmp.b, VL16);
                    }
                    ld1b(zmm_tmp.b, mask_tmp, ptr(reg_addr));
                    zip1(zmm_tmp.b, zmm_tmp.b, zmm_tmp.b);
                    zip1(zmm_tmp.h, zmm_tmp.h, zmm_tmp.h);
                    uxtb(zmm_inp_msk.s, mask_all_one / T_m, zmm_tmp.s);
                    if (mask_flag) {
                        not_(mask_tmp.b, mask_all_one.b, ktail_mask.b);
                        mov(zmm_inp_msk.s, mask_tmp / T_m, 0);
                    }
                    ldr(zmm_tmp, ptr(reg_stack));
                    add(reg_stack, reg_stack, 64);
                }
                if (!jcp.signed_input)
                    sub(zmm_inp_tmp.b, zmm_inp_tmp.b, vmm_shift.b);
            }
        }
        for (int ki = 0; ki < jcp.kw; ki++) {
            int aux_kernel_offset = kernel_offset(ci, ki);
            const int oi_start = get_ow_start(ki, pad_l);
            const int oi_end = get_ow_end(ur_w, ki, pad_r);
            if (compute_kernel) {
                if (jcp.is_fast_depthwise) {
                    auto reg_addr
                            = get_comp_addr_reg(aux_reg_ker, aux_kernel_offset);
                    ldr(QReg(zmm_wei.getIdx()), ptr(reg_addr));
                    ptrue(mask_tmp.d, VL2);
                    splice(zmm_wei.d, mask_tmp.d, zmm_wei.d);
                    ptrue(mask_tmp.d, VL4);
                    splice(zmm_wei.d, mask_tmp.d, zmm_wei.d);
                    not_(mask_tmp.b, mask_all_one, kblend_mask.b);
                    mov(zmm_wei.b, kblend_mask / T_m, zmm_wei.b);
                    mov(zmm_wei.b, mask_tmp / T_m, 0);
                } else {
                    auto reg_addr
                            = get_comp_addr_reg(aux_reg_ker, aux_kernel_offset);
                    auto zmm_tmp = ZReg(30);
                    sub(reg_stack, reg_stack, 64);
                    str(zmm_tmp, ptr(reg_stack));
                    ldr(QReg(zmm_tmp.getIdx()), ptr(reg_addr));
                    zip1(zmm_tmp.b, zmm_tmp.b, zmm_tmp.b);
                    zip1(zmm_tmp.h, zmm_tmp.h, zmm_tmp.h);
                    sxtb(zmm_wei.s, mask_all_one / T_m, zmm_tmp.s);
                    ldr(zmm_tmp, ptr(reg_stack));
                    add(reg_stack, reg_stack, 64);
                }
                if (h_padded) {
                    assert(!jcp.signed_input);
                    for (int oi = 0; oi < ur_w; oi++)
                        compute(zmm_out(oi, ci), zmm_wei, zmm_shifted_zero);
                } else {
                    auto r_zmm_src = zmm_src;
                    int oi_start = get_ow_start(ki, pad_l);
                    int oi_end = get_ow_end(ur_w, ki, pad_r);
                    int start_ = !jcp.signed_input ? 0 : oi_start;
                    int end_ = !jcp.signed_input ? ur_w : oi_end;
                    for (int oi = start_; oi < end_; oi++) {
                        if (oi >= oi_start && oi < oi_end) {
                            if (jcp.is_resrc_depthwise) {
                                int ii = input_spatial_index(oi, ki);
                                zmm_src = zmm_inp(ii, jcp.nb_ch_blocking);
                            } else {
                                int aux_input_offset
                                        = input_offset3(oi, ci, ki);
                                if (jcp.is_fast_depthwise) {
                                    assert(!mask_flag);
                                    auto reg_addr = get_comp_addr_reg(
                                            aux_reg_inp, aux_input_offset);
                                    ldr(QReg(r_zmm_src.getIdx()),
                                            ptr(reg_addr));
                                    ptrue(mask_tmp.d, VL2);
                                    splice(r_zmm_src.d, mask_tmp.d,
                                            r_zmm_src.d);
                                    ptrue(mask_tmp.d, VL4);
                                    splice(r_zmm_src.d, mask_tmp.d,
                                            r_zmm_src.d);
                                } else {
                                    auto reg_addr = get_comp_addr_reg(
                                            aux_reg_inp, aux_input_offset);
                                    auto zmm_tmp = ZReg(31);
                                    sub(reg_stack, reg_stack, 64);
                                    str(zmm_tmp, ptr(reg_stack));
                                    if (mask_flag) {
                                        eor(mask_tmp.b, mask_all_one,
                                                mask_tmp.b, mask_tmp.b);
                                        eor(mask_tmp2.b, mask_all_one,
                                                mask_tmp2.b, mask_tmp2.b);
                                        uzp1(mask_tmp.h, ktail_mask.h,
                                                mask_tmp.h);
                                        uzp1(mask_tmp.b, mask_tmp.b,
                                                mask_tmp2.b);
                                    } else {
                                        ptrue(mask_tmp.b, VL16);
                                    }
                                    ld1b(zmm_tmp.b, mask_tmp, ptr(reg_addr));
                                    zip1(zmm_tmp.b, zmm_tmp.b, zmm_tmp.b);
                                    zip1(zmm_tmp.h, zmm_tmp.h, zmm_tmp.h);
                                    uxtb(r_zmm_src.s, mask_all_one / T_m,
                                            zmm_tmp.s);
                                    if (mask_flag) {
                                        not_(mask_tmp.b, mask_all_one.b,
                                                ktail_mask.b);
                                        mov(r_zmm_src.s, mask_tmp / T_m, 0);
                                    }
                                    ldr(zmm_tmp, ptr(reg_stack));
                                    add(reg_stack, reg_stack, 64);
                                }
                                if (!jcp.signed_input)
                                    sub(zmm_src.b, zmm_src.b, vmm_shift.b);
                            }
                            compute(zmm_out(oi, ci), zmm_wei, zmm_src);
                        } else {
                            assert(!jcp.signed_input);
                            compute(zmm_out(oi, ci), zmm_wei, zmm_shifted_zero);
                        }
                    }
                }
            }
            if (jcp.src_zero_point) {
                /* calculate src_zero_point padding as:
                *      (is_padding ?
                *           src_zero_point_s32 * conv(1, wei_s32) : 0) */
                if (jcp.is_fast_depthwise || !compute_kernel) {
                    auto reg_addr
                            = get_comp_addr_reg(aux_reg_ker, aux_kernel_offset);
                    sub(reg_stack, reg_stack, 64);
                    str(vmm_tmp, ptr(reg_stack));
                    vmm_load_src(vmm_tmp, reg_addr, false);
                    zip1(vmm_tmp.b, vmm_tmp.b, vmm_tmp.b);
                    zip1(vmm_tmp.h, vmm_tmp.h, vmm_tmp.h);
                    sxtb(zmm_wei.s, mask_all_one / T_m, vmm_tmp.s);
                    ldr(vmm_tmp, ptr(reg_stack));
                    add(reg_stack, reg_stack, 64);
                    if (jcp.is_fast_depthwise) {
                        auto zmm_tmp1 = ZReg(28);
                        auto zmm_tmp2 = ZReg(30);
                        auto zmm_tmp3 = ZReg(29);
                        sub(reg_stack, reg_stack, 64);
                        str(zmm_tmp1, ptr(reg_stack));
                        sub(reg_stack, reg_stack, 64);
                        str(zmm_tmp2, ptr(reg_stack));
                        sub(reg_stack, reg_stack, 64);
                        str(zmm_tmp3, ptr(reg_stack));
                        mov(zmm_tmp1.s, 15);
                        and_(zmm_tmp1.b, mask_all_one, zmm_permute.b);
                        for (int i = 0; i < 16; i++) {
                            cmpeq(mask_tmp.s, mask_all_one, zmm_tmp1.s, i);
                            dup(zmm_tmp2.s, zmm_wei.s[i]);
                            mov(zmm_tmp3.s, mask_tmp / T_m, zmm_tmp2.s);
                        }
                        mov(zmm_wei.d, zmm_tmp3.d);
                        ldr(zmm_tmp3, ptr(reg_stack));
                        add(reg_stack, reg_stack, 64);
                        ldr(zmm_tmp2, ptr(reg_stack));
                        add(reg_stack, reg_stack, 64);
                        ldr(zmm_tmp1, ptr(reg_stack));
                        add(reg_stack, reg_stack, 64);
                    }
                } // else: already loaded weights from previous block
                int zp_offset = 0;
                for (int oi = 0; oi < ur_w; oi++) {
                    if (oi < oi_start || oi >= oi_end || h_padded) {
                        auto reg_addr = get_comp_addr_reg(
                                reg_src_zero_point, zp_offset);
                        vmm_load_zero_point(vmm_zp_tmp, reg_addr, mask_all_one,
                                jcp.zp_src_is_common);
                        mla(zmm_out(oi, ci).s, mask_all_one / T_m, zmm_wei.s,
                                vmm_zp_tmp.s);
                    }
                }
            }
        }
    }
    if (jcp.src_zero_point) {
        ldr(aux_reg_ker_d, ptr(reg_stack));
        add(reg_stack, reg_stack, 8);
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::compute_ker(int ur_w, int pad_l,
        int pad_r, ic_block_t last_ic_block_flag, bool h_padded) {
    if (jcp.is_depthwise)
        return compute_ker_dw(ur_w, pad_l, pad_r, last_ic_block_flag, h_padded);

    const bool compute_kernel = IMPLICATION(h_padded, !jcp.signed_input);

    assert(IMPLICATION(h_padded, jcp.src_zero_point || !jcp.signed_input));

    if (jcp.src_zero_point) {
        sub(reg_stack, reg_stack, 8);
        str(aux_reg_ker_d, ptr(reg_stack));
        ldr(reg_src_zero_point, ptr(param1, GET_OFF(src_zero_point)));
    }

    int kw = jcp.kw;
    int stride_w = jcp.stride_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int ch_block_all = jcp.ch_block * ic_block * oc_block;

    int nb_oc_block = jcp.nb_oc_blocking;

    auto input_offset = [=](int oi, int ic, int ki) {
        return jcp.typesize_in
                * ((ki * (jcp.dilate_w + 1) + oi * stride_w - pad_l)
                                * jcp.ic_without_padding * jcp.ngroups
                        + 4 * ic);
    };
    auto kernel_offset = [=](int ii, int ic, int ki) {
        return jcp.typesize_in
                * ((ii * jcp.nb_ic * jcp.kd * jcp.kh * jcp.kw + ki)
                                * ch_block_all
                        + 4 * ic * oc_block);
    };
    auto compute = [=](ZReg vreg_acc, ZReg vreg_wei, ZReg vreg_src) {
        sdot(ZRegS(vreg_acc.getIdx()), ZRegB(vreg_src.getIdx()),
                ZRegB(vreg_wei.getIdx()));
    };

    for (int ki = 0; ki < kw; ki++) {
        int jj_start = get_ow_start(ki, pad_l);
        int jj_end = get_ow_end(ur_w, ki, pad_r);
        int ic_tail_size = jcp.ic_without_padding % 4;
        int _start = (!jcp.signed_input) ? 0 : jj_start;
        int _end = (!jcp.signed_input) ? ur_w : jj_end;
        /* Skip the last loads of input if (ic%16)/4 < ic_block/4 */
        int icb = (last_ic_block_flag != no_last_block)
                ? div_up((jcp.ic_without_padding % ic_block), 4)
                : ic_block / 4;
        if (compute_kernel) {
            for (int ic = 0; ic < icb; ic++) {
                if (h_padded) {
                    /* fill padded area with shifted values */
                    if (ic == 0) {
                        auto inp = vmm_inp(0, nb_oc_block);
                        eor(inp.d, inp.d, inp.d);
                        sub(inp.b, inp.b, vmm_shift.b);
                    }
                } else {
                    for (int jj = _start; jj < _end; jj++) {
                        int aux_input_offset = input_offset(jj, ic, ki);
                        if (jj >= jj_start && jj < jj_end) {
                            if (last_ic_block_flag == last_sp_block
                                    && ic_tail_size != 0 && ic == icb - 1) {
                                auto xmm_tmp = VReg16B(
                                        vmm_inp(jj, nb_oc_block).getIdx());
                                for (int r = 0; r < ic_tail_size; ++r) {
                                    add_imm(reg_tmp0_adr, aux_reg_inp,
                                            (aux_input_offset + r),
                                            reg_tmp0_imm);
                                    ldrb(WReg(reg_tmp1_imm.getIdx()),
                                            ptr(reg_tmp0_adr));
                                    ins(VReg16B(xmm_tmp.getIdx())[r],
                                            WReg(reg_tmp1_imm.getIdx()));
                                }
                                dup(vmm_inp(jj, nb_oc_block).s,
                                        ZRegS(xmm_tmp.getIdx())[0]);
                            } else {
                                auto base = aux_reg_inp;
                                auto re = get_offset(aux_input_offset);

                                if ((-0x40 <= re) && (re < 0x40)
                                        && ((re % 4) == 0))
                                    ld1rw(vmm_inp(jj, nb_oc_block).s,
                                            mask_all_one,
                                            ptr(base,
                                                    static_cast<int32_t>(re)));
                                else {
                                    auto reg_tmp_adr = ((jj % 4) == 0)
                                            ? reg_tmp0_adr
                                            : ((jj % 4) == 1) ? reg_tmp1_adr
                                            : ((jj % 4) == 2) ? reg_tmp2_adr
                                                              : reg_tmp3_adr;
                                    auto reg_tmp_imm = ((jj % 4) == 0)
                                            ? reg_tmp0_imm
                                            : ((jj % 4) == 1) ? reg_tmp1_imm
                                            : ((jj % 4) == 2) ? reg_tmp2_imm
                                                              : reg_tmp3_imm;
                                    add_imm(reg_tmp_adr, base, re, reg_tmp_imm);
                                    ld1rw(vmm_inp(jj, nb_oc_block).s,
                                            mask_all_one, ptr(reg_tmp_adr));
                                }
                            }
                            if (!jcp.signed_input)
                                sub(vmm_inp(jj, nb_oc_block).b,
                                        vmm_inp(jj, nb_oc_block).b,
                                        vmm_shift.b);
                        } else {
                            // fill padded area with shifted value in
                            // first iteration
                            if (!jcp.signed_input && ic == 0) {
                                auto inp = vmm_inp(jj, nb_oc_block);
                                eor(inp.d, inp.d, inp.d);
                                sub(inp.b, inp.b, vmm_shift.b);
                            }
                        }
                    }
                }
                for (int ii = 0; ii < nb_oc_block; ii++) {
                    int aux_kernel_offset = kernel_offset(ii, ic, ki);
                    auto reg_addr
                            = get_comp_addr_reg(aux_reg_ker, aux_kernel_offset);
                    ld1w(vmm_wei.s, mask_all_one, ptr(reg_addr));
                    for (int jj = _start; jj < _end; jj++) {
                        auto inp = h_padded ? vmm_inp(0, nb_oc_block)
                                            : vmm_inp(jj, nb_oc_block);
                        compute(vmm_out(jj, ii), vmm_wei, inp);
                    }
                }
            }
        }
        if (jcp.src_zero_point) {
            /* calculate src_zero_point padding as:
             *      (is_padding ? src_zero_point_s32 * conv(1, wei_s8) : 0) */
            ZReg vmm_tmp = vmm_inp(0, nb_oc_block);
            for (int jj = 0; jj < ur_w; jj++) {
                if (jj < jj_start || jj >= jj_end || h_padded) {
                    for (int ii = 0; ii < nb_oc_block; ii++) {
                        eor(vmm_zp_tmp.d, vmm_zp_tmp.d, vmm_zp_tmp.d);
                        for (int ic = 0; ic < icb; ic++) {
                            int aux_kernel_offset = kernel_offset(ii, ic, ki);
                            auto reg_addr = get_comp_addr_reg(
                                    aux_reg_ker, aux_kernel_offset);
                            ld1w(vmm_tmp.s, mask_all_one, ptr(reg_addr));
                            sdot(vmm_zp_tmp.s, vmm_zp_one.b, vmm_tmp.b);
                        }
                        int zp_offset = 0;
                        auto reg_addr = get_comp_addr_reg(
                                reg_src_zero_point, zp_offset);
                        vmm_load_zero_point(vmm_tmp, reg_addr, mask_all_one,
                                jcp.zp_src_is_common);
                        mla(vmm_out(jj, ii).s, mask_all_one / T_m, vmm_tmp.s,
                                vmm_zp_tmp.s);
                    }
                }
            }
        }
    }
    if (jcp.src_zero_point) {
        ldr(aux_reg_ker_d, ptr(reg_stack));
        add(reg_stack, reg_stack, 8);
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::kh_loop(
        int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag) {
    Label kd_label, kh_label, skip_kd_loop, skip_kh_loop;
    Label f_overflow_label, no_f_overflow_label, d_h_f_overflow_label,
            t_overflow_label, no_t_overflow_label, b_overflow_label,
            no_b_overflow_label, back_overflow_label, no_back_overflow_label,
            d_h_back_overflow_label;

    int ch_block_all = jcp.ch_block * jcp.ic_block * jcp.oc_block;
    int shift_kernel_ptr = jcp.typesize_in * jcp.kw * ch_block_all;
    int shift_input_ptr
            = jcp.typesize_in * jcp.iw * jcp.ic_without_padding * jcp.ngroups;

    if (jcp.ndims == 5) {
        mov(aux_reg_ker_d, reg_ker);
        mov(aux_reg_inp_d, reg_inp);
        if (!jcp.signed_input || jcp.src_zero_point) {
            //TODO: May be avoided when f_pad=0 and dd0
            //TODO: Potential optimization by precomputing, when kd <<< od?
            ldr(reg_ki, ptr(reg_param1, GET_OFF(f_overflow)));
            cmp(reg_ki, 0);
            b(EQ, no_f_overflow_label);
            L(f_overflow_label);
            {
                mov(aux_reg_ker, aux_reg_ker_d);
                mov_imm(reg_kj, jcp.kh);
                L(d_h_f_overflow_label);
                {
                    compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);
                    adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr,
                            reg_tmp0_imm);
                    subs(reg_kj, reg_kj, 1);
                    b(NE, d_h_f_overflow_label);
                }
                add_imm(aux_reg_ker_d, aux_reg_ker_d, shift_kernel_ptr * jcp.kh,
                        reg_tmp0_imm);
                subs(reg_ki, reg_ki, 1);
                b(NE, f_overflow_label);
            }
            L(no_f_overflow_label);
        }

        ldr(reg_ki, ptr(reg_param1, GET_OFF(kd_padding)));
        if ((!jcp.signed_input || jcp.src_zero_point)
                || (jcp.dilate_d >= jcp.id)
                || (!(!jcp.signed_input || jcp.src_zero_point)
                        && (jcp.kd - 1) * (jcp.dilate_d + 1)
                                < nstl::max(jcp.f_pad, jcp.back_pad))) {
            cmp(reg_ki, 0);
            b(EQ, skip_kd_loop);
        }
        L(kd_label);
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    } else {
        if (jcp.is_fused_conv) {
            mov(aux_reg_inp_buffer_ptr, reg_inp_buffer_ptr);
        } else {
            mov(aux_reg_inp, reg_inp);
        }
        mov(aux_reg_ker, reg_ker);
    }

    if ((!jcp.signed_input || jcp.src_zero_point) && jcp.ndims > 3) {
        ldr(reg_overflow, ptr(reg_param1, GET_OFF(t_overflow)));
        cmp(reg_overflow, 0);
        b(EQ, no_t_overflow_label);
        L(t_overflow_label);
        {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp0_imm);
            subs(reg_overflow, reg_overflow, 1);
            cmp(reg_overflow, 0);
            b(GT, t_overflow_label);
        }
        L(no_t_overflow_label);
    }
    ldr(reg_kj, ptr(reg_param1, GET_OFF(kh_padding)));
    if ((!jcp.signed_input || jcp.src_zero_point) || (jcp.dilate_h >= jcp.ih)
            || (!(!jcp.signed_input || jcp.src_zero_point)
                    && (jcp.kh - 1) * (jcp.dilate_h + 1)
                            < nstl::max(jcp.t_pad, jcp.b_pad))) {
        cmp(reg_kj, 0);
        b(EQ, skip_kh_loop);
    }
    L(kh_label);
    {
        if (jcp.is_fused_conv) {
            ldr(aux_reg_inp, ptr(aux_reg_inp_buffer_ptr));
            add(aux_reg_inp, aux_reg_inp, reg_inp);
        }
        compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, false);

        adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp0_imm);
        if (jcp.is_fused_conv) {
            adds_imm(aux_reg_inp_buffer_ptr, aux_reg_inp_buffer_ptr,
                    sizeof(void *), reg_tmp0_imm);
        } else {
            adds_imm(aux_reg_inp, aux_reg_inp,
                    shift_input_ptr * (jcp.dilate_h + 1), reg_tmp0_imm);
        }
        subs(reg_kj, reg_kj, 1);
        cmp(reg_kj, 0);
        b(GT, kh_label);
    }
    L(skip_kh_loop);
    if ((!jcp.signed_input || jcp.src_zero_point) && jcp.ndims > 3) {
        ldr(reg_overflow, ptr(reg_param1, GET_OFF(b_overflow)));
        cmp(reg_overflow, 0);
        b(EQ, no_b_overflow_label);
        L(b_overflow_label);
        {
            compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);

            adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr, reg_tmp0_imm);
            subs(reg_overflow, reg_overflow, 1);
            cmp(reg_overflow, 0);
            b(GT, b_overflow_label);
        }
        L(no_b_overflow_label);
    }

    if (jcp.ndims == 5) {
        adds_imm(aux_reg_inp_d, aux_reg_inp_d,
                shift_input_ptr * jcp.ih * (jcp.dilate_d + 1), reg_tmp0_imm);
        adds_imm(aux_reg_ker_d, aux_reg_ker_d, shift_kernel_ptr * jcp.kh,
                reg_tmp0_imm);
        subs(reg_ki, reg_ki, 1);
        b(NE, kd_label);

        L(skip_kd_loop);
        if (!jcp.signed_input || jcp.src_zero_point) {
            ldr(reg_ki, ptr(reg_param1, GET_OFF(back_overflow)));
            cmp(reg_ki, 0);
            b(EQ, no_back_overflow_label);
            L(back_overflow_label);
            {
                mov(aux_reg_ker, aux_reg_ker_d);
                mov(reg_kj, jcp.kh);
                L(d_h_back_overflow_label);
                {
                    compute_ker(ur_w, pad_l, pad_r, last_ic_block_flag, true);
                    adds_imm(aux_reg_ker, aux_reg_ker, shift_kernel_ptr,
                            reg_tmp0_imm);
                    subs(reg_kj, reg_kj, 1);
                    b(NE, d_h_back_overflow_label);
                }
                adds_imm(aux_reg_ker_d, aux_reg_ker_d,
                        shift_kernel_ptr * jcp.kh, reg_tmp0_imm);
                subs(reg_ki, reg_ki, 1);
                b(NE, back_overflow_label);
            }
            L(no_back_overflow_label);
        }
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::icb_loop(
        int ur_w, int pad_l, int pad_r, bool is_last_sp_block) {

    if (jcp.src_zero_point && !jcp.is_depthwise) {
        eor(reg_scratch, reg_scratch, reg_scratch);
        dup(vmm_zp_one.b, 0x1);
    }

    prepare_output(ur_w);

    // IC loop
    Label icb_label;
    mov_imm(reg_icb, jcp.nb_ic);
    L(icb_label);
    const bool do_icb_loop
            = jcp.is_depthwise ? jcp.nb_ch > jcp.nb_ch_blocking : jcp.nb_ic > 1;
    if (jcp.ngroups % jcp.ch_block != 0 || jcp.ic_without_padding != jcp.ic) {
        Label common_ker, end_ker;
        if (do_icb_loop) {
            if (jcp.is_depthwise)
                cmp(reg_oc_blocks, jcp.nb_ch - jcp.nb_ch_blocking);
            else
                cmp(reg_icb, 1); // The last IC block
            b(NE, common_ker);
        }
        kh_loop(ur_w, pad_l, pad_r,
                is_last_sp_block ? last_sp_block : last_ic_block);
        if (do_icb_loop) {
            b(end_ker);

            L(common_ker);
            kh_loop(ur_w, pad_l, pad_r, no_last_block);

            L(end_ker);
        }
    } else {
        kh_loop(ur_w, pad_l, pad_r, no_last_block);
    }
    // End of IC Loop
    if (do_icb_loop) {
        int inp_step = jcp.ic_block;
        int ker_step = jcp.kd * jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;
        adds_imm(reg_inp, reg_inp, jcp.typesize_in * inp_step, reg_tmp0_imm);
        adds_imm(reg_ker, reg_ker, jcp.typesize_in * ker_step, reg_tmp0_imm);

        subs(reg_icb, reg_icb, 1);
        cmp(reg_icb, 0);
        b(GT, icb_label);

        subs_imm(reg_inp, reg_inp, jcp.typesize_in * inp_step * jcp.nb_ic,
                reg_tmp0_imm);
        subs_imm(reg_ker, reg_ker, jcp.typesize_in * ker_step * jcp.nb_ic,
                reg_tmp0_imm);
    }

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        Label common_store, end_store;

        if (jcp.is_depthwise)
            cmp(reg_oc_blocks, jcp.nb_ch - jcp.nb_ch_blocking);
        else
            cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);

        b(NE, common_store);

        store_output(ur_w, true); // last oc block
        b(end_store);

        L(common_store);
        store_output(ur_w, false);

        L(end_store);
    } else {
        store_output(ur_w, false);
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::vmm_mask_all_one() {
    mask_gflag = false;
    if (sve_len_ == 64) {
        mask_gflag = true;
        ptrue(mask_all_one.b);
    } else if (sve_len_ == 32) {
        ptrue(mask_all_one.b, VL32);
    } else if (sve_len_ == 16) {
        ptrue(mask_all_one.b, VL16);
    } else {
        assert(!"unreachable");
    }
}

void jit_sve_512_x8s8s32x_fwd_kernel::vmm_load_src(
        ZReg src, XReg reg_addr, bool mask_flag) {
    if (mask_flag) {
        eor(mask_tmp.b, mask_all_one, mask_tmp.b, mask_tmp.b);
        eor(mask_tmp2.b, mask_all_one, mask_tmp2.b, mask_tmp2.b);
        uzp1(mask_tmp.h, ktail_mask.h, mask_tmp.h);
        uzp1(mask_tmp.b, mask_tmp.b, mask_tmp2.b);
    } else {
        if (sve_len_ == 64)
            ptrue(mask_tmp.b, VL16);
        else if (sve_len_ == 32)
            ptrue(mask_tmp.b, VL8);
        else if (sve_len_ == 16)
            ptrue(mask_tmp.b, VL4);
        else
            assert(!"unreabhable");
    }

    ld1b(src.b, mask_tmp, ptr(reg_addr));
}

void jit_sve_512_x8s8s32x_fwd_kernel::vmm_load_zero_point(
        ZReg src, XReg reg_addr, PReg mask, bool bcast) {
    if (bcast)
        ld1rw(src.s, mask, ptr(reg_addr));
    else
        ld1w(src.s, mask, ptr(reg_addr));
}

void jit_sve_512_x8s8s32x_fwd_kernel::generate() {
    Label permute_index_table;
    int in_ic_shift = jcp.is_fused_conv ? jcp.dw_conv_buffer_oc
                                        : jcp.ic_without_padding * jcp.ngroups;
    const int urw_inp_stride = jcp.ur_w * jcp.stride_w;
    const int n_urw_l_pad
            = nstl::min(div_up(jcp.l_pad, urw_inp_stride), jcp.ow / jcp.ur_w);
    const int inp_shift_pad = nstl::max(0,
            jcp.typesize_in * (n_urw_l_pad * urw_inp_stride - jcp.l_pad)
                    * in_ic_shift);
    int inp_shift = jcp.typesize_in * (jcp.ur_w * jcp.stride_w * in_ic_shift);
    int out_shift = jcp.typesize_out
            * (jcp.ur_w * jcp.oc_without_padding * jcp.ngroups);
    preamble();

    vmm_mask_all_one();

    if (jcp.is_depthwise) {
        bool is_zero_point = jcp.src_zero_point || jcp.dst_zero_point;
        // dst zero point and dst scale reuse the same register
        int idx = jcp.max_regs_ur - 1
                + nstl::max(2 * is_zero_point, static_cast<int>(jcp.dst_scale));
        if (!jcp.is_resrc_depthwise) zmm_src = ZReg(++idx);
        if (jcp.is_fast_depthwise) zmm_permute = ZReg(++idx);
        if (!jcp.signed_input) zmm_shifted_zero = ZReg(++idx);
        // due to extra register used for shifts and compensations
        // and/or saturation, we increment by one more
        if (!jcp.signed_input || jcp.need_saturation) ++idx;
        assert(IMPLICATION(!jcp.dst_scale && !is_zero_point
                /*&& jcp.dst_dt != data_type::bf16*/,
                idx == ker_dw_reg_base_idx));
    }

    if (jcp.is_fused_conv) {
        ldr(reg_inp_buffer_ptr, ptr(reg_param1, GET_OFF(src)));
        /* In case of fused depthwise convolution, `param.src` is not a pointer
        to input, instead it points to a buffer containing pointers to
        consecutive rows of input in format wc with c=jcp.dw_conv_buffer_oc.
        Example: [ptr_to_inp_row0, ptr_to_inp_row1, ptr_to_inp_row2].
        Traverse the data as
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row0 ...
            add(reg_input_buffer_ptr, sizeof(void*))
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row1 ...
            add(reg_input_buffer_ptr, sizeof(void*))
            mov(reg_data, ptr[reg_input_buffer_ptr])
            ... process row2 ...
        */
        mov_imm(reg_inp, 0);
    } else {
        ldr(reg_inp, ptr(reg_param1, GET_OFF(src)));
    }
    ldr(reg_out, ptr(reg_param1, GET_OFF(dst)));
    ldr(reg_ker, ptr(reg_param1, GET_OFF(filt)));

    if (jcp.ngroups % jcp.ch_block != 0 || jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.is_depthwise
                ? jcp.ngroups % jcp.ch_block
                : jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        ldr(reg_oc_blocks, ptr(reg_param1, GET_OFF(oc_blocks)));
        auto regw_tmp = reg_oi;
        mov(regw_tmp, mask);
        auto vmm_tmp1 = ZReg(31);
        auto vmm_tmp2 = ZReg(30);
        index(vmm_tmp1.s, 0, 1);
        mov(vmm_tmp2.s, 1);
        lsl(vmm_tmp2.s, mask_all_one / T_m, vmm_tmp1.s);
        dup(vmm_tmp1.s, WReg(regw_tmp.getIdx()));
        and_(vmm_tmp1.d, vmm_tmp1.d, vmm_tmp2.d);
        cmpne(ktail_mask.s, mask_all_one, vmm_tmp1.s, 0);
        cmpne(postops_mask.s, mask_all_one, vmm_tmp1.s, 0);
    } else if (jcp.with_binary)
        if (jcp.oc_block != isa_simd_width_) {
            const int mask = (1 << jcp.oc_block) - 1;
            auto regw_tmp = reg_oi;
            mov(regw_tmp, mask);
            auto vmm_tmp1 = ZReg(31);
            auto vmm_tmp2 = ZReg(30);
            index(vmm_tmp1.s, 0, 1);
            mov(vmm_tmp2.s, 1);
            lsl(vmm_tmp2.s, mask_all_one / T_m, vmm_tmp1.s);
            dup(vmm_tmp1.s, WReg(regw_tmp.getIdx()));
            and_(vmm_tmp1.d, vmm_tmp1.d, vmm_tmp2.d);
            cmpne(postops_mask.s, mask_all_one, vmm_tmp1.s, 0);
        }
    if (jcp.is_fast_depthwise) {
        // prepare mask register for blending weights
        movk(reg_scratch, uint16_t(0x1111), 0);
        movk(reg_scratch, uint16_t(0x2222), 16);
        movk(reg_scratch, uint16_t(0x4444), 32);
        movk(reg_scratch, uint16_t(0x8888), 48);
        sub(reg_stack, reg_stack, 8);
        str(reg_scratch, ptr(reg_stack));
        ldr(kblend_mask, ptr(reg_stack));
        add(reg_stack, reg_stack, 8);
        // load permute indices from data section
        adr(reg_scratch, permute_index_table);
        ld1w(zmm_permute.s, mask_all_one, ptr(reg_scratch));
    }

    const int extended_filter_size
            = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const int r_pad = nstl::max(0, jcp.r_pad);
    const int ow_with_no_rpad = 1
            + (jcp.iw + jcp.l_pad + nstl::min(0, jcp.r_pad)
                      - extended_filter_size)
                    / jcp.stride_w;
    const int n_urw_per_ow_block = jcp.ow_block / jcp.ur_w;
    const int max_safe_iw = nstl::max(
            0, jcp.iw - div_up(ic_sub_step, jcp.ic_without_padding));
    const int max_safe_ow = jcp.ic_without_padding % ic_sub_step == 0
            ? jcp.ow
            : (max_safe_iw + jcp.l_pad - extended_filter_size) / jcp.stride_w;
    Label middle_block_label, done_compute;
    std::vector<Label> ow_block_jmp_table;

    // r_pad_fall_through is a special ow_block, where the block overlaps
    // both middle_block and r_pad/ur_w_tail region when it exists.
    // The number of ur_w's to compute in middle_block before executing
    // r_pad region is stored in r_pad_fall_through_n_urw and the ow_block
    // number is stored in r_pad_fall_through_ow_block.
    int r_pad_fall_through_ow_block = 0;
    int r_pad_fall_through_n_urw = 0;

    if (jcp.nb_ow > 1) {
        // Only one ow block is processed, per jit call.
        // Number of this ow block is passed as parameter owb,
        // and padding processing depends on this number.
        //
        // The compute block to run is determined by using a jmp-table.
        // jmp-table Layout:
        //  idx -> addr
        //  0   -> [...l_pad_region label[0]...]
        //         : : : : : : : : : : : : : : :
        //  L ->   [...l_pad_region label[L]...]
        //  L+1 -> [...r_pad_region label[0]...]
        //         : : : : : : : : : : : : : : :
        //  L+R -> [...r_pad_region label[R]...]
        //
        // Note: Label for middle_block is not stored in the jmp-table.
        //
        // During jit call, the jump address is calculated as below:
        // if (owb < L) {
        //   jmp([jmp_table + owb*sizeof(void*)]);
        // } else if (owb < X) {
        //   // X is the number of ow_blocks before r_pad region (see below).
        //   jmp(middle_block);
        // } else {
        //   sub(owb, X);
        //   jmp([jmp_table + owb*sizeof(void*) + L*sizeof(void)]);
        // }
        //
        // To configure the jmp-table, we need to determine some constants
        // (namely, r_pad_fall_through_n_urw, r_pad_fall_through_ow_block,
        // n_l_pad_labels, n_labels) ahead of writing the compute assembly. So,
        // we simulate the filter path without writing the assembly initially.
        // This makes the math for calculating the constants become simple and
        // self explanatory.

        // Begin simulation without writing assembly
        int n_l_pad_labels = 0;
        int n_labels = 0;
        int cur_ow = 0;

        // l_pad region:
        n_l_pad_labels = div_up(n_urw_l_pad, n_urw_per_ow_block);
        n_labels = n_l_pad_labels;
        cur_ow += n_urw_l_pad * jcp.ur_w;

        // middle_region:
        int n_urw_middle_block_loop = 0;
        int cur_r_pad = nstl::max(0,
                calculate_end_padding(jcp.l_pad, cur_ow + jcp.ur_w, jcp.iw,
                        jcp.stride_w, extended_filter_size));
        if (cur_ow + jcp.ur_w <= jcp.ow && cur_r_pad == 0) {
            n_urw_middle_block_loop
                    = nstl::max(0,
                              nstl::min(ow_with_no_rpad, max_safe_ow) - cur_ow)
                    / jcp.ur_w;
            cur_ow += n_urw_middle_block_loop * jcp.ur_w;
        }
        r_pad_fall_through_n_urw = (cur_ow / jcp.ur_w) % n_urw_per_ow_block;
        r_pad_fall_through_ow_block = cur_ow / (n_urw_per_ow_block * jcp.ur_w);

        // r_pad or last_sp_block
        if (cur_ow + jcp.ur_w <= jcp.ow) {
            if (r_pad_fall_through_n_urw == 0) ++n_labels;
            const int n_urw_r_pad_region = (jcp.ow - cur_ow) / jcp.ur_w;
            n_labels += nstl::max(0,
                    div_up(r_pad_fall_through_n_urw + n_urw_r_pad_region,
                            n_urw_per_ow_block)
                            - 1);
        }

        if (jcp.ur_w_tail != 0) {
            if (jcp.ow % jcp.ow_block == jcp.ur_w_tail) ++n_labels;
        }
        // End of simulation

        ow_block_jmp_table.resize(n_labels);

        // Begin jump-table logic
        Label ow_block_jmp_table_label;
        if (!ow_block_jmp_table.empty()) {
            adr(reg_jmp_tbl_base, ow_block_jmp_table_label);
        }
        mov_imm(reg_oi, n_urw_per_ow_block);
        ldr(reg_owb, ptr(reg_param1, GET_OFF(owb)));
        if (jcp.l_pad > 0) {
            Label middle_or_rpad_check;
            cmp_imm(reg_owb, n_l_pad_labels, reg_tmp0_imm);
            b(GE, middle_or_rpad_check);
            mov_imm(reg_tmp0_imm, sizeof(void *));
            madd(reg_tmp0_adr, reg_owb, reg_tmp0_imm, reg_jmp_tbl_base);
            br(reg_tmp0_adr);
            L(middle_or_rpad_check);
            // harness passes shifted src pointer that does not take
            // left-padding into account. So, we must re-shift here.
            const int inp_shift_pad_middle_block = -1 * jcp.typesize_in
                    * nstl::min(jcp.l_pad, n_urw_l_pad * urw_inp_stride)
                    * in_ic_shift;
            add_imm(reg_inp, reg_inp, inp_shift_pad_middle_block, reg_tmp0_imm);
        }
        if (r_pad_fall_through_n_urw != 0) {
            Label reg_scratch_end_label;
            mov_imm(reg_scratch, r_pad_fall_through_n_urw);
            cmp_imm(reg_owb, r_pad_fall_through_ow_block, reg_tmp1_imm);
            b(NE, reg_scratch_end_label);
            add(reg_oi, reg_scratch, 0);
            L(reg_scratch_end_label);
            if (n_urw_middle_block_loop > 0) {
                subs_imm(reg_owb, reg_owb, r_pad_fall_through_ow_block,
                        reg_tmp0_imm);
                // simple middle_block
                b(LE, middle_block_label);
                sub(reg_owb, reg_owb, 1);
            } else {
                sub_imm(reg_owb, reg_owb, r_pad_fall_through_ow_block + 1,
                        reg_tmp0_imm);
            }
        } else {
            subs_imm(reg_owb, reg_owb, r_pad_fall_through_ow_block,
                    reg_tmp0_imm);
            // simple middle_block
            if (n_urw_middle_block_loop) b(LT, middle_block_label);
        }
        // r_pad-only region
        if (!ow_block_jmp_table.empty()) {
            mov_imm(reg_tmp0_imm, sizeof(void *));
            mul(reg_tmp1_imm, reg_owb, reg_tmp0_imm);
            add_imm(reg_tmp0_adr, reg_tmp1_imm, n_l_pad_labels * sizeof(void *),
                    reg_tmp2_imm);
            add(reg_tmp1_adr, reg_jmp_tbl_base, reg_tmp0_adr);
            br(reg_tmp1_adr);
        }

        if (!ow_block_jmp_table.empty()) {
            //align(8);
            L(ow_block_jmp_table_label);
            {
                for (size_t i = 0; i < ow_block_jmp_table.size(); ++i) {
                    adr(reg_tmp0_adr, ow_block_jmp_table[i]);
                    br(reg_tmp0_adr);
                }
            }
        }
        // End of jump-table logic
    }

    // Begin kernel
    int cur_ow = 0;
    int cur_n_oi = 0; // used only for jcp.nb_ow > 1 scenario
    int label_cntr = 0;
    int cur_l_pad = 0;
    if (jcp.l_pad > 0) {
        for (cur_l_pad = jcp.l_pad;
                cur_l_pad > 0 && cur_ow + jcp.ur_w <= jcp.ow;
                cur_l_pad -= urw_inp_stride) {
            if (jcp.nb_ow > 1 && cur_n_oi == 0) {
                // cur_n_oi == 0 signifies beginning of new ow_block
                // (or end of previous block)
                const dim_t inp_lpad_region_shift = -label_cntr * jcp.ow_block
                        * jcp.stride_w * in_ic_shift;
                L(ow_block_jmp_table[label_cntr++]);
                // harness passes shifted src pointer that does not take
                // left-padding into account. So, we must re-shift here.
                add_imm(reg_inp, reg_inp, inp_lpad_region_shift, reg_tmp0_imm);
            }

            cur_ow += jcp.ur_w;
            int cur_r_pad = nstl::max(0,
                    calculate_end_padding(jcp.l_pad, cur_ow, jcp.iw,
                            jcp.stride_w, extended_filter_size));
            icb_loop(jcp.ur_w, cur_l_pad, cur_r_pad, cur_ow > max_safe_ow);
            add_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);
            sub(reg_oi, reg_oi, 1);

            if (jcp.nb_ow > 1 && ++cur_n_oi == n_urw_per_ow_block) {
                // We compute one owb per jit call. So, insert an
                // unconditional jmp, after computing one owb.
                b(done_compute);
                cur_n_oi = 0;
            }
        }
        if (jcp.nb_ow == 1 || cur_n_oi != 0) {
            // Let it "fall-through" middle_block_label
            add_imm(reg_inp, reg_inp, inp_shift_pad, reg_tmp0_imm);
        }
    }

    // middle_block
    {
        int cur_r_pad = nstl::max(0,
                calculate_end_padding(jcp.l_pad, cur_ow + jcp.ur_w, jcp.iw,
                        jcp.stride_w, extended_filter_size));
        if (cur_r_pad == 0 && cur_ow + jcp.ur_w <= jcp.ow) {
            int n_oi_middle_block_loop
                    = nstl::max(0,
                              nstl::min(ow_with_no_rpad, max_safe_ow) - cur_ow)
                    / jcp.ur_w;
            if (jcp.nb_ow == 1 && n_oi_middle_block_loop > 1)
                mov_imm(reg_oi, n_oi_middle_block_loop);
            L(middle_block_label);
            if (n_oi_middle_block_loop > 0) {
                icb_loop(jcp.ur_w, 0, 0, false);
                add_imm(reg_inp, reg_inp, inp_shift, reg_tmp0_imm);
                add_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);
                if (n_oi_middle_block_loop > 1) {
                    subs(reg_oi, reg_oi, 1);
                    b(GT, middle_block_label);
                }
            }
            cur_ow += n_oi_middle_block_loop * jcp.ur_w;
            cur_n_oi = (cur_n_oi + n_oi_middle_block_loop) % n_urw_per_ow_block;
        }
    }

    // r_pad region or last_sp_block
    if (cur_ow + jcp.ur_w <= jcp.ow) {
        if (jcp.nb_ow > 1) {
            if (cur_n_oi == 0) {
                b(done_compute);
            } else {
                // r_pad fall-through
                ldr(reg_owb, ptr(reg_param1, GET_OFF(owb)));
                cmp(reg_owb, r_pad_fall_through_ow_block);
                b(NE, done_compute);
            }
        }

        while (cur_ow + jcp.ur_w <= jcp.ow) {
            if (jcp.nb_ow > 1 && cur_n_oi == 0) {
                L(ow_block_jmp_table[label_cntr++]);
            }
            cur_ow += jcp.ur_w;
            int cur_r_pad = calculate_end_padding(jcp.l_pad, cur_ow, jcp.iw,
                    jcp.stride_w, extended_filter_size);
            assert(cur_r_pad > 0 || cur_ow > max_safe_ow); // else, why be here?
            icb_loop(jcp.ur_w, 0, cur_r_pad, cur_ow > max_safe_ow);
            add_imm(reg_inp, reg_inp, inp_shift, reg_tmp0_imm);
            add_imm(reg_out, reg_out, out_shift, reg_tmp0_imm);

            if (jcp.nb_ow > 1 && ++cur_n_oi == n_urw_per_ow_block) {
                // We compute one owb per jit call. So, insert an
                // unconditional jmp, after computing one owb.
                b(done_compute);
                cur_n_oi = 0;
            }
        }
        // Let it fall-through ur_w_tail
    }

    // ur_w_tail
    if (jcp.ur_w_tail != 0) {
        if (jcp.nb_ow > 1) {
            if (cur_n_oi == 0) {
                b(done_compute);
                L(ow_block_jmp_table[label_cntr++]);
            } else {
                // In case, when there is no r_pad region, then there exists an
                // ambiguity btw middle_blocks and r_pad_fall_through_ow_block.
                // If not properly distinguished, there can be a race condition
                // as middle_blocks and r_pad_fall_through_ow_block both try to
                // compute ur_w_tail work at the end.
                ldr(reg_owb, ptr(reg_param1, GET_OFF(owb)));
                cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
                b(NE, done_compute);
            }
        }
        icb_loop(jcp.ur_w_tail, nstl::max(0, cur_l_pad), r_pad, true);
    }
    L(done_compute);
    assert(ow_block_jmp_table.size() == static_cast<size_t>(label_cntr));
    postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();

    if (jcp.is_fast_depthwise) {
        align(64);
        L(permute_index_table);
        const uint32_t _idx[]
                = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
        for (size_t i = 0; i < sizeof(_idx) / sizeof(_idx[0]); ++i)
            dd(_idx[i]);
    }
}

status_t jit_sve_512_x8s8s32x_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();
    const bool is_1d = ndims == 3;
    const bool is_2d = ndims == 4;
    const bool is_3d = ndims == 5;
    assert(is_1d || is_2d || is_3d);

    if (!(mayiuse(sve_512)
                && one_of(src_d.data_type(), data_type::u8, data_type::s8)
                && weights_d.data_type() == data_type::s8
                && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                        data_type::s8, data_type::u8)))
        return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.nthr = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
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

    jcp.ur_h = 1; /* no code-unrolling by h so far */
    jcp.dilate_d = is_3d ? cd.dilates[0] : 0;
    jcp.dilate_h = is_1d ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    if (kernel_outside_src) return status::unimplemented;

    jcp.signed_input = (src_d.data_type() == data_type::s8) ? true : false;
    jcp.need_saturation = utils::one_of(
            dst_d.data_type(), data_type::u8, data_type::s8, data_type::s32);
    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);

    if (jcp.is_depthwise && is_3d)
        // NOTE: 3D depthwise is not currently supported here.
        return status::unimplemented;

    if (jcp.is_depthwise) {
        jcp.ch_block = 16;
        jcp.ic_block = 1;
        jcp.oc_block = 1;
    } else {
        jcp.ch_block = 1;
        jcp.ic_block = 16;
        jcp.oc_block = 16;

        if (jcp.ngroups == 1) {
            /* For non grouped convolutions, pad channels by 16 if needed */
            jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
            jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        } else if (jcp.ngroups != 1
                && ((jcp.ic % jcp.ic_block != 0)
                        || (jcp.oc % jcp.oc_block != 0))) {
            /* For grouped convolutions, oneDNN doesn't support padding.
               When channels per group is not multiple of 4, 8, 16, return unimplemented. */
            jcp.ic_block = (jcp.ic % 8 == 0) && (jcp.oc % 8 == 0) ? 8 : 4;
            jcp.oc_block = jcp.ic_block;
        }
        if (jcp.ic % jcp.ic_block != 0 || jcp.oc % jcp.oc_block != 0)
            return status::unimplemented;
    }

    jcp.simd_w = jcp.is_depthwise ? jcp.ch_block : jcp.ic_block;

    const auto zp = attr.zero_points_;
    jcp.dst_zero_point = !zp.has_default_values(DNNL_ARG_DST);
    jcp.src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
    jcp.zp_src_is_common
            = zp.common(DNNL_ARG_SRC); // otherwise, it's per-channel
    assert(IMPLICATION(jcp.src_zero_point, jcp.zp_src_is_common));

    if ((jcp.dst_zero_point || jcp.src_zero_point) && jcp.is_fused_conv)
        return status::unimplemented;

    const auto &src_scales = attr.scales_.get(DNNL_ARG_SRC);
    const auto &wei_scales = attr.scales_.get(DNNL_ARG_WEIGHTS);
    const auto &dst_scales = attr.scales_.get(DNNL_ARG_DST);
    const int wei_mask_per_oc = 1 << (int)with_groups;
    jcp.is_oc_scale = wei_scales.mask_ == wei_mask_per_oc;
    jcp.dst_scale = !dst_scales.has_default_values();

    // only common src & dst scales are supported
    // only common and per-oc-channel weight scales are supported
    const bool scales_ok = one_of(wei_scales.mask_, 0, wei_mask_per_oc)
            && everyone_is(src_scales.mask_, dst_scales.mask_, 0);
    if (!scales_ok) return status::unimplemented;

    jcp.is_fast_depthwise = true && jcp.is_depthwise
            && jcp.ngroups % jcp.ch_block == 0; /* groups not multiple of
    ch_block (= 16) would require byte masking for load from src */

    jcp.is_resrc_depthwise = jcp.is_depthwise && jcp.stride_w < jcp.kw
            && jcp.kw < 4 && jcp.dilate_w == 0;
    if (jcp.is_depthwise) {
        jcp.max_regs_ur = 31 - jcp.is_fast_depthwise - !jcp.is_resrc_depthwise
                - (!jcp.signed_input)
                - (!jcp.signed_input || jcp.need_saturation); // both alias
    } else {
        jcp.max_regs_ur = 31;
    }

    // TODO: re-implement so that the JIT Kernel uses the least amount of
    // registers. Currently, there are issues because of compile and run time
    // definitions.
    if (jcp.dst_scale) jcp.max_regs_ur = 26;
    if (jcp.src_zero_point || jcp.dst_zero_point) jcp.max_regs_ur = 25;

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        using namespace memory_extra_flags;
        format_tag_t wei_tag;
        if (jcp.ic_block == 16 || jcp.ch_block == 16) {
            if (is_3d) {
                wei_tag = with_groups ? gOIdhw4i16o4i : OIdhw4i16o4i;
            } else if (is_1d) {
                wei_tag = with_groups ? jcp.is_depthwise ? Goiw16g : gOIw4i16o4i
                                      : OIw4i16o4i;
            } else {
                assert(is_2d);
                wei_tag = with_groups
                        ? jcp.is_depthwise ? Goihw16g : gOIhw4i16o4i
                        : OIhw4i16o4i;
            }
        } else if (jcp.ic_block == 8) {
            assert(with_groups);
            wei_tag = is_3d ? gOIdhw2i8o4i : is_2d ? gOIhw2i8o4i : gOIw2i8o4i;
        } else {
            assert(with_groups && jcp.ic_block == 4);
            wei_tag = is_3d ? gOIdhw4o4i : is_2d ? gOIhw4o4i : gOIw4o4i;
        }

        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);
        if (!jcp.signed_input) {
            want_wei_md.extra.flags = 0 | compensation_conv_s8s8 | scale_adjust;
            want_wei_md.extra.compensation_mask = (1 << 0)
                    + (with_groups && !jcp.is_depthwise ? (1 << 1) : 0);
            want_wei_md.extra.scale_adjust = 1.f;
        }
        if (jcp.src_zero_point) {
            want_wei_md.extra.flags |= compensation_conv_asymmetric_src;
            want_wei_md.extra.asymm_compensation_mask = (1 << 0)
                    + (with_groups && !jcp.is_depthwise ? (1 << 1) : 0);
        }

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }

        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) return status::unimplemented;

    format_tag_t dat_tag = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag));
        jcp.src_tag = dat_tag;
    } else {
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.src_tag != dat_tag) return status::unimplemented;

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, dat_tag));
        jcp.dst_tag = dat_tag;
    } else {
        jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);
    }
    if (jcp.dst_tag != dat_tag) return status::unimplemented;

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));
    }

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    CHECK(attr.set_default_formats(&dst_md));

    const auto &post_ops = attr.post_ops_;
    const int eltwise_ind = post_ops.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;

    const int binary_ind = post_ops.find(primitive_kind::binary);
    jcp.with_binary = binary_ind != -1;

    const int sum_ind = post_ops.find(primitive_kind::sum);
    jcp.with_sum = sum_ind != -1;
    jcp.sum_dt = post_ops.get_sum_dt(jcp.dst_dt);

    jcp.post_ops = post_ops;

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = false;
    static constexpr bool sum_requires_scale_one = false;
    static constexpr bool sum_requires_zp_zero = false;
    const bool post_ops_ok_ = post_ops_ok({sve_512, {eltwise, binary, sum},
            jcp.post_ops, &dst_d, sum_at_pos_0_only, sum_requires_scale_one,
            sum_requires_zp_zero});
    if (!post_ops_ok_) return status::unimplemented;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    // Try to use 4 channel-groups at a time to avoid false sharing (depthwise)
    int nb_ch_blocking = 4;
    for (/* init above */; nb_ch_blocking > 1; nb_ch_blocking--)
        if (jcp.nb_ch % nb_ch_blocking == 0) break;
    jcp.nb_ch_blocking = jcp.is_depthwise ? nb_ch_blocking : 1;

    // If OC blocking is incommensurate with the number of OC blocks (general
    // requirement for all convolutions), or if it results in an unrolling
    // factor smaller than the left padding (special requirement for SSD:fc6),
    // then search for a smaller OC blocking that satisfies both constraints.
    auto is_oc_blocking_ok = [&](int block) {
        int ur_w = nstl::min(jcp.ow, jcp.max_regs_ur / (block + 1));
        return jcp.nb_oc % block == 0 && jcp.l_pad <= ur_w
                && jcp.ow % ur_w != 1;
    };

    // choose nb_oc work chunk size for distribution within threads
    int max_threading_nb_oc_chunk = 4;
    jcp.nb_oc_blocking_thr_chunk
            = nstl::min(max_threading_nb_oc_chunk, jcp.nb_oc);
    for (; jcp.nb_oc_blocking_thr_chunk > 1; jcp.nb_oc_blocking_thr_chunk--) {
        if (is_oc_blocking_ok(jcp.nb_oc_blocking_thr_chunk)) break;
    }

    // choose oc blocking for computational kernel
    jcp.nb_oc_blocking = jcp.nb_oc_blocking_thr_chunk;

    if (jcp.is_resrc_depthwise)
        jcp.ur_w = (jcp.max_regs_ur - jcp.kw + jcp.stride_w)
                / (jcp.nb_ch_blocking + jcp.stride_w);
    else
        jcp.ur_w = jcp.max_regs_ur
                / (jcp.is_depthwise ? jcp.nb_ch_blocking
                                    : jcp.nb_oc_blocking + 1);
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    auto get_thr_eff = [=](int nb_ow, int nthr) {
        int base_work_amount = jcp.mb * jcp.nb_ch * jcp.od * jcp.oh
                * (jcp.nb_oc / jcp.nb_oc_blocking_thr_chunk);
        auto work_amount = base_work_amount * nb_ow;
        return float(work_amount) / rnd_up(work_amount, nthr);
    };

    auto get_ow_block = [=](int ur_w, int nthr) {
        int res_ow_block = jcp.ow;
        float best_thr_eff = get_thr_eff(1, nthr);
        float thr_eff;
        int max_nb_ow = div_up(jcp.ow, ur_w);
        for (int nb_ow = 1; nb_ow <= max_nb_ow; nb_ow++) {
            int ow_block
                    = nstl::min(rnd_up(div_up(jcp.ow, nb_ow), ur_w), jcp.ow);
            if (ow_block < jcp.nb_oc_blocking_thr_chunk * jcp.oc_block
                    && best_thr_eff > 0.8f)
                break;
            if (div_up(jcp.ow, ow_block) != nb_ow) continue;
            thr_eff = get_thr_eff(nb_ow, nthr);
            if (ow_block >= ur_w && thr_eff > 1.1f * best_thr_eff) {
                res_ow_block = ow_block;
                best_thr_eff = thr_eff;
            }
            if (best_thr_eff > 0.9f) break;
        }
        return res_ow_block;
    };

    jcp.ow_block = get_ow_block(jcp.ur_w, jcp.nthr);
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    if (jcp.oc % jcp.oc_block != 0) return status::unimplemented;

    pick_loop_order(jcp, jcp.nthr);

    jcp.wei_adj_scale
            = (weights_d.extra().flags & memory_extra_flags::scale_adjust)
            ? weights_d.extra().scale_adjust
            : 1.f;

    return status::success;
}

void jit_sve_512_x8s8s32x_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {
    const int wei_mask = attr.scales_.get(DNNL_ARG_WEIGHTS).mask_;
    const dim_t scales_count = wei_mask == 0 ? 1 : jcp.oc * jcp.ngroups;
    dim_t count = wei_mask == 0 ? (dim_t)16 : scales_count;
    scratchpad.book<float>(key_conv_adjusted_scales, count);
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
