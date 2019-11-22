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

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "memory.hpp"

#include "jit_avx2_x8s8s32x_1x1_conv_kernel.hpp"
#include "jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {

using namespace dnnl::impl::utils;

using namespace Xbyak;

void jit_avx2_x8s8s32x_1x1_conv_kernel::cvt2ps(data_type_t type_in,
        const Ymm &ymm_in, const Reg64 &reg, int offset, int load_size) {

    load_data(type_in, ymm_in, reg, offset, load_size);
    if (type_in != data_type::f32) vcvtdq2ps(ymm_in, ymm_in);
}

bool jit_avx2_x8s8s32x_1x1_conv_kernel::maybe_eltwise(int position) {
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* eltwise before sum */
        return p.contain(eltwise, 0);
    } else if (position == 1) {
        /* eltwise after sum */
        return p.contain(sum, 0) && p.contain(eltwise, 1);
    }

    return false;
}

void jit_avx2_x8s8s32x_1x1_conv_kernel::bcast_loop(int load_loop_blk) {
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);

    mov(aux_reg_output_data, reg_output_data);
    mov(reg_bcast_loop_iter, ptr[rsp + bcast_loop_work_off]);

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(reg_bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop);
    {
        assert(jcp.bcast_block == jcp.ur);
        reduce_loop(load_loop_blk, jcp.ur, 0, false);
        add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step);
        add(aux_reg_output_data, jcp.bcast_loop_output_step);

        sub(reg_bcast_loop_iter, jcp.bcast_block);
        cmp(reg_bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        cmp(reg_bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        L(bcast_loop_tail_out);
    }
}

void jit_avx2_x8s8s32x_1x1_conv_kernel::reduce_loop(
        int load_loop_blk, int ur, int substep, bool wraparound) {
    auto vreg_load = [&](int i_load) {
        const int ymm_idx = ur * load_loop_blk + i_load;
        assert(ymm_idx < 13);
        return Ymm(ymm_idx);
    };

    auto vreg_accum = [&](int i_load, int i_ur) {
        const int ymm_idx = i_ur * load_loop_blk + i_load;
        assert(ymm_idx < 13);
        return Ymm(ymm_idx);
    };

    auto ymm_bias_alpha = [&]() {
        const int ymm_idx = ur * load_loop_blk;
        assert(ymm_idx < 13);
        return Ymm(ymm_idx);
    };

    auto xmm_bias_alpha = [&]() { return Xmm(ymm_bias_alpha().getIdx()); };

    auto bcast_ptr = [&](int i_reduce, int i_ur) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        assert(jcp.reduce_loop_unroll == jcp.reduce_block);

        int offt = (jcp.ic_without_padding * i_ur + i_reduce);

        return ptr[aux_reg_bcast_data + jcp.typesize_in * offt];
    };

    auto load_ptr = [&](int i_reduce, int i_load) {
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;

        int offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;

        return ptr[aux_reg_load_data + u1 * jcp.reduce_loop_load_step
                + jcp.typesize_in * offt];
    };

    auto init = [&]() {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                vpxor(r, r, r);
            }
        if (jcp.signed_input) {
            auto xmm_shift = Xbyak::Xmm(ymm_shift.getIdx());
            auto _t32 = reg_init_bcast.cvt32();
            mov(_t32, (int8_t)-128);
            vpinsrb(xmm_shift, xmm_shift, _t32, 0);
            vpbroadcastb(ymm_shift, xmm_shift);
        }
    };

    auto store = [&](const bool mask_flag_in) {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale
                = (sum_idx != -1) ? &p.entry_[sum_idx].sum.scale : nullptr;
        mov(ptr[rsp + reg_bcast_data_off], reg_bcast_data);
        mov(reg_ptr_scales, ptr[rsp + reg_ptr_sum_scale_off]);
        if (p_sum_scale && *p_sum_scale != 1.f) {
            mov(ptr[rsp + reg_load_data_off], reg_load_data);
            mov(reg_ptr_sum_scale, (size_t)p_sum_scale);
        }
        if (jcp.signed_input) {
            mov(reg_store_bcast, float2int(jcp.wei_adj_scale));
            vmovq(xmm_bias_alpha(), reg_store_bcast);
            vbroadcastss(ymm_bias_alpha(), xmm_bias_alpha());
        }
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            auto ymm_bias = ymm_tmp;
            auto ymm_comp = ymm_bcast;
            if (jcp.with_bias) {
                if (jcp.signed_input)
                    mov(reg_bias_data, ptr[rsp + reg_bias_data_off]);
                cvt2ps(jcp.bia_dt, ymm_bias, reg_bias_data,
                        jcp.typesize_bia * jcp.oc_block * i_load,
                        mask_flag ? get_tail_size() : simd_w);
                if (jcp.signed_input)
                    vmulps(ymm_bias, ymm_bias, ymm_bias_alpha());
            }
            if (jcp.signed_input) {
                mov(reg_comp_data, ptr[rsp + reg_comp_data_off]);
                cvt2ps(data_type::s32, ymm_comp, reg_comp_data,
                        sizeof(int32_t) * jcp.oc_block * i_load,
                        mask_flag ? get_tail_size() : simd_w);
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                vcvtdq2ps(r, r);
                if (jcp.signed_input) vaddps(r, r, ymm_comp);
                if (jcp.with_bias) vaddps(r, r, ymm_bias);

                const auto ptr_scales_offset = jcp.is_oc_scale
                        * (sizeof(float) * jcp.oc_block * i_load);
                if (mask_flag) {
                    cvt2ps(data_type::f32, ymm_zero, reg_ptr_scales,
                            ptr_scales_offset, get_tail_size());
                    vmulps(r, r, ymm_zero);
                    vpxor(ymm_zero, ymm_zero, ymm_zero);
                    vblendps(r, ymm_zero, r, ((1 << get_tail_size()) - 1));
                } else
                    vmulps(r, r, ptr[reg_ptr_scales + ptr_scales_offset]);
            }
        }

        if (maybe_eltwise(0))
            eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);

        if (p_sum_scale) { // post_op: sum
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    const bool mask_flag
                            = mask_flag_in && i_load == load_loop_blk - 1;
                    auto ymm_prev_dst = ymm_zero;

                    auto r = vreg_accum(i_load, i_ur);
                    cvt2ps(jcp.dst_dt, ymm_prev_dst, aux_reg_output_data,
                            jcp.typesize_out
                                    * (jcp.oc_without_padding * i_ur
                                            + i_load * jcp.load_block),
                            mask_flag ? get_tail_size() : simd_w);

                    if (*p_sum_scale == 1.f)
                        vaddps(r, ymm_prev_dst);
                    else {
                        vbroadcastss(ymm_tmp, ptr[reg_ptr_sum_scale]);
                        vfmadd231ps(r, ymm_prev_dst, ymm_tmp);
                    }
                }
            }
        }

        if (maybe_eltwise(1))
            eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);

        // Zero out ymm_zero register to be used in next loop
        if (jcp.dst_dt == data_type::u8) vpxor(ymm_zero, ymm_zero, ymm_zero);
        for (int i_ur = 0; i_ur < ur; ++i_ur) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                const bool mask_flag
                        = mask_flag_in && i_load == load_loop_blk - 1;
                auto r = vreg_accum(i_load, i_ur);
                if (jcp.dst_dt == data_type::u8) {
                    // Assumption: ymm_zero register is already zeroed!
                    vmaxps(r, ymm_zero, r);
                }
                if (jcp.dst_dt != data_type::f32) vcvtps2dq(r, r);

                store_data(jcp.dst_dt, r, aux_reg_output_data,
                        jcp.typesize_out
                                * (jcp.oc_without_padding * i_ur
                                        + i_load * jcp.load_block),
                        mask_flag ? get_tail_size() : simd_w);
            }
        }
        mov(reg_bcast_data, ptr[rsp + reg_bcast_data_off]);
        if (p_sum_scale && *p_sum_scale != 1.f)
            mov(reg_load_data, ptr[rsp + reg_load_data_off]);
    };

    auto compute = [&](Ymm vreg_acc, Ymm vreg_wei, Ymm vreg_src) {
        uni_vpmaddubsw(ymm_tmp, vreg_src, vreg_wei);
        uni_vpmaddwd(ymm_tmp, ymm_tmp, ymm_one);
        uni_vpaddd(vreg_acc, vreg_acc, ymm_tmp);
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
                    load_bytes(ymm_bcast, aux_reg_bcast_data,
                            jcp.ic_without_padding * i_ur + i_reduce,
                            ic_tail_size);
                    vpbroadcastd(ymm_bcast, Xmm(ymm_bcast.getIdx()));
                } else {
                    vpbroadcastd(ymm_bcast, bcast_ptr(i_reduce, i_ur));
                }
                if (jcp.signed_input) vpsubb(ymm_bcast, ymm_bcast, ymm_shift);
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    compute(vreg_accum(i_load, i_ur), vreg_load(i_load),
                            ymm_bcast);
                }
            }
        }
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reg_reduce_loop_iter, reg_reduce_loop_work);
    sub(reg_reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop);
    {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reg_reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(jcp.ic != jcp.ic_without_padding);

    if (jcp.oc_without_padding != jcp.oc) {
        Label end_store, common_store;
        mov(ptr[rsp + reg_bcast_data_off], reg_bcast_data);

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

void jit_avx2_x8s8s32x_1x1_conv_kernel::generate() {
    preamble();

    mov(reg_init_bcast, 0x1);
    auto xmm_one = Xmm(ymm_one.getIdx());
    vmovq(xmm_one, reg_init_bcast);
    vpbroadcastw(ymm_one, xmm_one);

    sub(rsp, stack_space_needed);

    if (jcp.with_bias) mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    if (jcp.signed_input) {
        mov(ptr[rsp + reg_bias_data_off], reg_bias_data);
        mov(reg_comp_data, ptr[param1 + GET_OFF(compensation)]);
        mov(ptr[rsp + reg_comp_data_off], reg_comp_data);
    }
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
    mov(ptr[rsp + reg_ptr_sum_scale_off], reg_ptr_scales);
    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(ptr[rsp + bcast_loop_work_off], reg_bcast_loop_work);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(first_last_flag)]);

    auto load_loop_body = [&](int load_loop_blk) {
        bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        if (jcp.with_bias) {
            if (jcp.signed_input)
                mov(reg_bias_data, ptr[rsp + reg_bias_data_off]);
            add(reg_bias_data,
                    load_loop_blk * jcp.load_block * jcp.typesize_bia);
            if (jcp.signed_input)
                mov(ptr[rsp + reg_bias_data_off], reg_bias_data);
        }
        if (jcp.signed_input) {
            mov(reg_comp_data, ptr[rsp + reg_comp_data_off]);
            add(reg_comp_data,
                    load_loop_blk * jcp.load_block * sizeof(int32_t));
            mov(ptr[rsp + reg_comp_data_off], reg_comp_data);
        }
        mov(ptr[rsp + reg_bcast_data_off], reg_bcast_data);
        mov(reg_ptr_scales, ptr[rsp + reg_ptr_sum_scale_off]);
        add(reg_ptr_scales,
                jcp.is_oc_scale * load_loop_blk * jcp.load_block
                        * sizeof(float));
        mov(ptr[rsp + reg_ptr_sum_scale_off], reg_ptr_scales);
        mov(reg_bcast_data, ptr[rsp + reg_bcast_data_off]);
        add(reg_output_data, load_loop_blk * jcp.load_block * jcp.typesize_out);
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    };

    static const int ur_cases[] = {2, 3, 5, 12};
    constexpr int num_ur_cases = sizeof(ur_cases) / sizeof(*ur_cases);
    Label load_loop_blk[num_ur_cases + 1];

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

    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
}

bool jit_avx2_x8s8s32x_1x1_conv_kernel::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    switch (p.len_) {
        case 0: return true;
        case 1: return is_eltwise(0) || p.contain(sum, 0);
        case 2:
            return (p.contain(sum, 0) && is_eltwise(1))
                    || (p.contain(sum, 1) && is_eltwise(0));
        default: return false;
    }

    return false;
}

status_t jit_avx2_x8s8s32x_1x1_conv_kernel::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const memory_desc_wrapper &bias_d, const primitive_attr_t &attr,
        int nthreads, bool reduce_src) {
    if (!mayiuse(avx2)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!one_of(src_d.data_type(), data_type::u8, data_type::s8)
            || weights_d.data_type() != data_type::s8
            || !one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8))
        return status::unimplemented;

    const bool is_2d = src_d.ndims() == 4;
    if (!is_2d) return status::unimplemented;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.signed_input = (src_d.data_type() == data_type::s8);

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    format_tag_t dat_tag = format_tag::nhwc;
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);

    bool args_ok = true && jcp.ngroups == 1 && jcp.src_tag == dat_tag
            && jcp.dst_tag == dat_tag;
    if (!args_ok) return status::unimplemented;

    jcp.oc = rnd_up(jcp.oc, simd_w);
    jcp.ic = rnd_up(jcp.ic, simd_w);

    args_ok = true && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
            && jcp.t_pad == 0 && jcp.l_pad == 0 && jcp.stride_w == 1
            && jcp.stride_h == 1 && jcp.ow == jcp.iw
            && jcp.oh == jcp.ih // enforce rpad=0
            && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    const int SMALL_SPATIAL = 7 * 7;
    const int BIG_REDUCE_DIM = 512;

    int load_blocking = 0;
    int load_blocking_max = 0;
    int bcast_blocking = 0;
    int bcast_blocking_max = 0;
    int reduce_blocking = 0;
    int reduce_blocking_max = 0;
    jcp.load_grp_count = 1;

    const int L2_size = get_cache_size(2, true) / sizeof(jcp.typesize_in);
    const int L2_capacity = (L2_size * 3) / 4;

    int size_threshold = 28;

    int min_regs = 3;
    int max_regs = 5;

    if (jcp.mb == 1 && jcp.ic > 128
            && (jcp.oh <= size_threshold && jcp.ow <= size_threshold)) {
        if (jcp.os <= SMALL_SPATIAL && jcp.oc * jcp.ic < L2_size)
            max_regs = min_regs = 3;
        jcp.ur = nstl::min(max_regs, jcp.os);
    } else {
        const int spatial = jcp.oh;
        jcp.ur = 1;
        for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
            if ((spatial >= size_threshold && spatial % ur_w == 0)
                    || (spatial < size_threshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = nstl::min(max_regs, jcp.os);
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
            = jcp.ur * jcp.oc_without_padding * jcp.typesize_out;
    jcp.bcast_loop_bcast_step
            = jcp.ur * jcp.ic_without_padding * jcp.typesize_in;

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

    jcp.load_grp_count = div_up(nthreads, jcp.mb * jcp.ngroups * nb_bcast);
    jcp.load_grp_count = best_divider(
            nthreads, jcp.load_grp_count, 2 * jcp.load_grp_count, false);

    if (jcp.bcast_dim <= SMALL_SPATIAL
            && jcp.load_dim * jcp.reduce_dim >= L2_size) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 4);
    } else if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.mb <= nthreads
            && jcp.load_dim > 256 && jcp.load_dim / jcp.reduce_dim >= 4) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2);
        load_blocking = jcp.load_block;
    }

    bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                             div_up(nthreads, jcp.load_grp_count))
            * jcp.bcast_block;
    bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
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

    const bool params_ok = true && load_blocking > 0 && load_blocking_max > 0
            && bcast_blocking > 0 && bcast_blocking_max > 0
            && reduce_blocking > 0 && reduce_blocking_max > 0
            && load_blocking % jcp.load_block == 0
            && reduce_blocking % jcp.reduce_block == 0
            && load_blocking_max % jcp.load_block == 0
            && reduce_blocking_max % jcp.reduce_block == 0
            && jcp.reduce_loop_unroll % 4 == 0
            && jcp.reduce_dim % jcp.reduce_loop_unroll == 0
            && jcp.bcast_block % jcp.ur == 0
            && jcp.reduce_dim % jcp.reduce_block == 0;

    assert(params_ok && "parameter values are inconsistent");
    if (!params_ok) return status::unimplemented;

    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

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

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;
    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    jcp.wei_adj_scale
            = (weights_d.extra().flags & memory_extra_flags::scale_adjust)
            ? weights_d.extra().scale_adjust
            : 1.f;

    return status::success;
}

void jit_avx2_x8s8s32x_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace dnnl::impl::memory_tracking::names;

    if (jcp.signed_input) {
        dim_t count = nstl::max<dim_t>(attr.output_scales_.count_, 8);
        scratchpad.book(key_conv_adjusted_scales, sizeof(float) * count);
    }
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
