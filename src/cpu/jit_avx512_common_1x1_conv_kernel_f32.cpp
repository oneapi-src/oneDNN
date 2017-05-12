/*******************************************************************************
* Copyright 2017 Intel Corporation
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
#include <float.h>
#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "jit_avx512_common_1x1_conv_kernel_f32.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

void jit_avx512_common_1x1_conv_kernel_f32::bcast_loop(int load_loop_blk)
{
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);

    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, EVEX_compress_addr(rsp, bcast_loop_work_offt));

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop); {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_substep);
            } else {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step
                        - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_step
                        - (num_substeps - 1) * jcp.bcast_loop_output_substep);
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
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        L(bcast_loop_tail_out);
    }
}

void jit_avx512_common_1x1_conv_kernel_f32::reduce_loop(int load_loop_blk,
         int ur, int substep, bool wraparound)
{
    auto vreg_load = [=](int i_load) {
        return Zmm(ur * load_loop_blk + i_load);
    };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return Zmm(i_ur * load_loop_blk + i_load);
    };

    auto bias_ptr = [=](int i_load) {
        return EVEX_compress_addr(reg_bias_data,
                                  sizeof(float) * jcp.oc_block * i_load);
    };

    auto bcast_ptr = [=](int i_reduce, int i_ur) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        size_t offt;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data)) {
            assert(jcp.reduce_loop_unroll == jcp.reduce_block);
            offt = (i_reduce == jcp.reduce_loop_unroll)
                           ? (jcp.bcast_dim + i_ur) * jcp.reduce_loop_unroll
                           : i_ur * jcp.reduce_loop_unroll + i_reduce;
        } else
            offt = i_reduce * jcp.ic_block + i_ur;
        return EVEX_compress_addr(aux_reg_bcast_data, sizeof(float) * offt,
                                  true);
    };

    auto load_ptr = [=](int i_reduce, int i_load) {
        size_t offt;
        size_t u0 = i_reduce % jcp.reduce_loop_unroll;
        size_t u1 = i_reduce / jcp.reduce_loop_unroll;
        if (jcp.prop_kind == backward_data)
            offt = (i_load * jcp.reduce_block + u0) * jcp.load_block;
        else
            offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;
        return EVEX_compress_addr(aux_reg_load_data,
                                  u1 * jcp.reduce_loop_load_step
                                  + sizeof(float) * offt);
    };

    auto output_ptr = [=](int i_load, int i_ur) {
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                   backward_data))
            return EVEX_compress_addr(aux_reg_output_data,
                    (i_load * jcp.bcast_dim + i_ur) * jcp.load_block
                    * sizeof(float));
        else
            return ptr[aux_reg_output_data +
                       (i_load
                            ? reg_output_stride * i_load
                            : 0) // TODO: Xbyak should allow 0 scale
                       + sizeof(float) * jcp.load_block * i_ur];
    };

    auto init = [=]() {
        Label init_done;
        Label init_zero;

        if (jcp.with_bias
            && one_of(jcp.prop_kind, forward_training, forward_inference)) {
            test(reg_reduce_pos_flag, REDUCE_FLAG_FIRST);
            jz(init_zero, T_NEAR);

            for (int i_load = 0; i_load < load_loop_blk; i_load++)
                for (int i_ur = 0; i_ur < ur; ++i_ur)
                    vmovups(vreg_accum(i_load, i_ur), bias_ptr(i_load));
            jmp(init_done, T_NEAR);
        }

        L(init_zero);
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                vpxord(r, r, r);
            }

        L(init_done);
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            vmovups(vreg_load(i_load), load_ptr(0, i_load));
    };
    auto store = [=]() {

        Label store_noadd;

        test(reg_reduce_pos_flag, REDUCE_FLAG_FIRST);
        jnz(store_noadd, T_NEAR);
        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto r = vreg_accum(i_load, i_ur);
                vaddps(r, r, output_ptr(i_load, i_ur));
            }

        L(store_noadd);

        if (jcp.with_relu) {
            const unsigned char _cmp_lt_os = 1;
            assert(ur * load_loop_blk < 14);

            Label store_norelu;
            test(reg_reduce_pos_flag, REDUCE_FLAG_LAST);
            jz(store_norelu, T_NEAR);

            for (int i_ur = 0; i_ur < ur; ++i_ur)
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    vcmpps(vmask, vreg_accum(i_load, i_ur), zmm_zero,
                        _cmp_lt_os);
                    vmulps(vreg_accum(i_load, i_ur) | vmask,
                        vreg_accum(i_load, i_ur), zmm_relu_ns);
            }
            L(store_norelu);
        }

        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                vmovntps(output_ptr(i_load, i_ur), vreg_accum(i_load, i_ur));

    };

    auto prefetch_callback = [=](int ur, int i_reduce, int i_ur, int i_load,
                                      bool last_block, bool wraparound) {
        bool pf_ker_l1 = true;
        bool pf_ker_l2 = wraparound;
        int n_ops = jcp.reduce_loop_unroll * ur * load_loop_blk;
        int i_op =
            i_reduce * ur * load_loop_blk + i_ur * load_loop_blk + i_load;

        int n_pf_ker_l1 = pf_ker_l1 ? jcp.reduce_block : 0;
        int n_pf_ker_l2 = pf_ker_l2 && wraparound ? jcp.reduce_block : 0;
        int n_pf_out_l1 = 0;

        int pf_inp_ops = n_ops / 2; // # of operations during which to pf input
        int pf_inp_trigger;
        if (jcp.prop_kind == backward_weights)
            pf_inp_trigger = nstl::max(1, pf_inp_ops / jcp.reduce_block);
        else
            pf_inp_trigger = nstl::max(1, pf_inp_ops / ur);

        int n_other_pf =
            load_loop_blk * (n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1);
        int n_other_pf_ops = n_ops - pf_inp_ops;
        int other_pf_trigger
                = n_other_pf ? nstl::max(1, n_other_pf_ops / n_other_pf) : 0;

        if (i_op < pf_inp_ops && i_op % pf_inp_trigger == 0) {
            // input prefetches have the highest priority b/c the
            // first iteration of the kernel block touches all the
            // cache lines
            int i_pf = i_op / pf_inp_trigger;
            auto pf_reg = wraparound && last_block
                                  ? reg_bcast_data
                                  : (last_block ? aux1_reg_bcast_data
                                                : aux_reg_bcast_data);
            int offt = i_pf;
            if (jcp.prop_kind == backward_weights) {
                offt += wraparound && last_block
                                    ? 0
                                    : (last_block ? jcp.is : jcp.reduce_block);
                offt *= jcp.bcast_block;
            } else {
                offt += wraparound && last_block
                                    ? 0
                                    : (last_block ? jcp.ur : jcp.bcast_dim);
                offt *= jcp.reduce_block;
            }
            mic_prefetcht0(ptr[pf_reg + offt * sizeof(float)]);
        } else if (i_op >= pf_inp_ops && n_other_pf) {
            // remaining prefetches are spread among the rest of the
            // operations; prefetches for output take priority
            // TODO: spread L2 prefetches among L1 prefetches
            i_op -= pf_inp_ops;
            if (i_op % other_pf_trigger == 0) {
                int i_pf = i_op / (load_loop_blk * other_pf_trigger);
                if (i_pf < n_pf_ker_l2) {
                    int offt = (i_pf + (i_load + 1) * jcp.reduce_dim)
                                    * jcp.load_block;
                    if (jcp.prop_kind == backward_data)
                        offt = (i_pf + (i_load + 1) * jcp.reduce_block)
                                * jcp.load_block;
                    mic_prefetcht1(ptr[aux_reg_load_data
                                    + offt * sizeof(float)]);
                } else if (i_pf < n_pf_ker_l2 + n_pf_ker_l1) {
                    i_pf -= n_pf_ker_l2;
                    auto pf_reg = last_block ? reg_load_data
                                             : aux_reg_load_data;
                    int offt = (i_pf + i_load * jcp.reduce_dim
                                    + (last_block
                                        ? (wraparound ? jcp.reduce_dim : 0)
                                        : jcp.reduce_block))
                                * jcp.load_block;
                    if (jcp.prop_kind == backward_data) {
                        offt = (i_pf + i_load * jcp.reduce_block + (last_block
                                ? (wraparound ? jcp.load_block : 0)
                                : jcp.load_dim)) * jcp.reduce_block;
                    }
                    mic_prefetcht0(ptr[pf_reg + offt * sizeof(float)]);
                } else if (i_pf < n_pf_ker_l1 + n_pf_ker_l2 + n_pf_out_l1) {
                    i_pf -= n_pf_ker_l1 + n_pf_ker_l2;
                    int offt = i_pf * jcp.load_block;
                    mic_prefetcht0(ptr[aux_reg_output_data
                                    + offt * sizeof(float)]);
                }
            }
        }
    };

    auto fma_block = [=](bool last_block) {
        for (int i_reduce = 0; i_reduce < jcp.reduce_loop_unroll; ++i_reduce) {
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    vfmadd231ps(vreg_accum(i_load, i_ur), vreg_load(i_load),
                                bcast_ptr(i_reduce, i_ur));

                    if (i_ur == ur - 1
                        && !(last_block
                            && i_reduce == jcp.reduce_loop_unroll - 1))
                        vmovups(vreg_load(i_load),
                                load_ptr(i_reduce + 1,
                                i_load));

                    prefetch_callback(ur, i_reduce, i_ur, i_load,
                                        last_block, wraparound);
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

    L(reduce_loop); {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
}

void jit_avx512_common_1x1_conv_kernel_f32::diff_bias_loop(int load_loop_blk)
{
    // TODO: This function is obsolete because diff_bias calculated in harness.
    if (!jcp.with_bias || jcp.prop_kind != backward_weights)
        return;

    Label diff_bias_loop;
    Label diff_bias_loop_out;
    Label diff_bias_init_out;
    Label diff_bias_load;

    auto diff_bias_ptr = [=](int i_load) {
        return EVEX_compress_addr(reg_diff_bias_data,
                                  i_load * jcp.oc_block * sizeof(float));
    };

    auto load_ptr = [=](int i_reduce, int i_load) {
        return EVEX_compress_addr(aux_reg_load_data,
                (i_load * jcp.os + i_reduce) * jcp.oc_block * sizeof(float));
    };

    auto diff_bias_reg = [=](int i_load) { return Zmm(i_load); };

    mov(reg_diff_bias_data,
        EVEX_compress_addr(rsp, reg_diff_bias_data_stack_offt));

    cmp(reg_diff_bias_data, 0);
    je(diff_bias_loop_out, T_NEAR);

    test(reg_reduce_pos_flag, REDUCE_FLAG_FIRST);
    jz(diff_bias_load, T_NEAR);

    for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
        auto r = diff_bias_reg(i_load);
        vpxord(r, r, r);
    }
    jmp(diff_bias_init_out, T_NEAR);

    L(diff_bias_load);
    for (int i_load = 0; i_load < load_loop_blk; ++i_load)
        vmovups(diff_bias_reg(i_load), diff_bias_ptr(i_load));

    L(diff_bias_init_out);
    mov(aux_reg_load_data, reg_load_data);
    mov(reduce_loop_iter, reg_reduce_loop_work);
    L(diff_bias_loop);
    {
        for (int i_reduce = 0; i_reduce < jcp.reduce_loop_unroll; ++i_reduce)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                vaddps(diff_bias_reg(i_load),
                        diff_bias_reg(i_load),
                        load_ptr(i_reduce, i_load));
        assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jnz(diff_bias_loop, T_NEAR);
    }

    for (int i_load = 0; i_load < load_loop_blk; i_load++)
        vmovups(diff_bias_ptr(i_load), diff_bias_reg(i_load));
    add(reg_diff_bias_data, load_loop_blk * jcp.oc_block * sizeof(float));
    mov(EVEX_compress_addr(rsp, reg_diff_bias_data_stack_offt),
        reg_diff_bias_data);

    L(diff_bias_loop_out);
}

static int ur_cases[] = {2, 4, 5, 8, 14, 28};
constexpr int ur_num = sizeof(ur_cases) / sizeof(ur_cases[0]);

void jit_avx512_common_1x1_conv_kernel_f32::generate()
{
    preamble();

    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);

    sub(rsp, stack_space_needed);

    if (jcp.with_bias) {
        if (jcp.prop_kind == backward_weights) {
            mov(reg_diff_bias_data, ptr[param1 + GET_OFF(bias_data)]);
            mov(ptr[rsp + reg_diff_bias_data_stack_offt], reg_diff_bias_data);
        } else
            mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    }

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(EVEX_compress_addr(rsp, bcast_loop_work_offt), reg_bcast_loop_work);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(reduce_pos_flag)]);
    if (jcp.prop_kind == backward_weights)
        mov(reg_output_stride, ptr[param1 + GET_OFF(output_stride)]);
    mov(reg_relu_ns, reinterpret_cast<size_t>(&jcp.relu_negative_slope));

    vbroadcastss(zmm_relu_ns, ptr[reg_relu_ns]);
    vpxord(zmm_zero, zmm_zero, zmm_zero);

    auto load_loop_body = [=](int load_loop_blk) {
        bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        switch (jcp.prop_kind) {
        case forward_training:
        case forward_inference:
            add(reg_bias_data, load_loop_blk * jcp.load_block * sizeof(float));
            add(reg_output_data,
                load_loop_blk * jcp.bcast_dim * jcp.load_block * sizeof(float));
            break;
        case backward_data:
            add(reg_output_data,
                load_loop_blk * jcp.bcast_dim * jcp.load_block * sizeof(float));
            break;
        case backward_weights:
            for (int i_load = 0; i_load < load_loop_blk; i_load++)
                add(reg_output_data, reg_output_stride);
            break;
        default:
            assert(!"invalid prop_kind");
        }
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    };

    const int simd_w = 16;

    Label load_loop_blk[7];

    for (int ur_idx = ur_num - 1; ur_idx > 0; ur_idx--) {
        int label_idx = ur_num - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            jle(load_loop_blk[label_idx], T_NEAR);
        }
    }

    for (int ur_idx = 0; ur_idx < ur_num; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = ur_num - ur_idx - 1;
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    cmp(reg_load_loop_work, 0);
                    je(load_loop_blk[ur_num], T_NEAR);
                }
                diff_bias_loop(label_idx + 1);
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
            if (ur_idx < ur_num - 2) {
                cmp(reg_load_loop_work, simd_w);
                jle(load_loop_blk[0], T_NEAR);
            }
        }
    }
    L(load_loop_blk[ur_num]);

    add(rsp, stack_space_needed);

    postamble();
}

namespace {

float loss_ratio(int amount, int divider)
{
    return float(rnd_up(amount, divider) - amount) / rnd_up(amount, divider);
}

int best_divider(int value, int min_divider, int max_divider,
                        bool find_max, int step = 1)
{
    max_divider = nstl::max(1, nstl::min(max_divider, value));
    min_divider = nstl::max(1, nstl::min(min_divider, max_divider));

    float min_loss = FLT_MAX;
    int x_divider = max_divider;
    for (int divider = max_divider; divider >= min_divider; divider -= step) {
        const float loss = loss_ratio(value, divider);
        if ((find_max && loss < min_loss) || (!find_max && loss <= min_loss)) {
            min_loss = loss;
            x_divider = divider;
        }
    }
    return x_divider;
}

}

status_t jit_avx512_common_1x1_conv_kernel_f32::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, bool with_relu,
        double relu_negative_slope)
{
    if (!mayiuse(avx512_common)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

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

    jcp.src_fmt = src_d.format();
    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;

    constexpr memory_format_t weights_formats[2][2] = {
            { OIhw16i16o, OIhw16o16i },
            { gOIhw16i16o, gOIhw16o16i }
    };
    memory_format_t weights_format
            = weights_formats[with_groups][jcp.prop_kind == backward_data];

    bool args_ok = true
        && jcp.ngroups == 1
        && src_d.format() == nChw16c
        && weights_d.format() == weights_format
        && one_of(cd.bias_desc.format, memory_format::undef, any, x)
        && dst_d.format() == nChw16c;
    if (!args_ok) return status::unimplemented;

    const int simd_w = 16;

    args_ok = true
        && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
        && jcp.t_pad == 0 && jcp.l_pad == 0
        && jcp.stride_w == 1 && jcp.stride_h == 1 // TODO: support some strides
        && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.ur = 1;

    int max_regs = 28;
    int min_regs = 8;

    for (int ur_w = min_regs; ur_w <= max_regs; ur_w++) {
        if (jcp.os % ur_w == 0) {
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
                if (i_tail == 0)
                    break;
            }
        }
    }

    int load_blocking{ 0 };
    int load_blocking_max{ 0 };
    int bcast_blocking{ 0 };
    int bcast_blocking_max{ 0 };
    int reduce_blocking{ 0 };
    int reduce_blocking_max{ 0 };

    const int KNL_L2_capacity = (512 * 1024) / sizeof(float);
    const int KNL_L1_capacity = (64 * 1024) / sizeof(float);

    if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
        jcp.reduce_dim = jcp.ic;
        jcp.reduce_block = jcp.ic_block;

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.is;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.is * sizeof(float);
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.oc_block * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur * jcp.oc_block * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.ic * jcp.oc_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        load_blocking = nstl::min(64, jcp.load_dim);
        load_blocking_max = load_blocking;

        reduce_blocking = (KNL_L2_capacity / 2) / load_blocking;
        reduce_blocking = jcp.reduce_block *
            (reduce_blocking / jcp.reduce_block); // round by jcp.reduce_block

        reduce_blocking = nstl::min(reduce_blocking, jcp.reduce_dim);

        int reduce_tail = jcp.reduce_dim % reduce_blocking;
        if (reduce_tail) {
            for (int i = reduce_blocking; i > jcp.reduce_block;
                                                i -= jcp.reduce_block)
                if (jcp.reduce_dim % i == 0) {
                    reduce_blocking = i;
                    reduce_tail = 0;
                    break;
                }
        }
        if (reduce_tail && reduce_tail < 64) {
            for (int i = reduce_blocking; i >= reduce_blocking - 64;
                                                    i -= jcp.reduce_block)
                if (jcp.reduce_dim % i >= 64) {
                    reduce_blocking = i;
                    break;
                }
        }

        const int USABLE_KNL_L2 =
            (KNL_L2_capacity - reduce_blocking * load_blocking) * sizeof(float);
        int os_block_size = sizeof(float) * reduce_blocking * jcp.ur;
        bcast_blocking
                = nstl::min(jcp.os, jcp.ur * (USABLE_KNL_L2 / os_block_size));
        int bcast_tail = jcp.os % bcast_blocking;
        if (bcast_tail) {
            for (int i = bcast_blocking; i > jcp.ur; i -= jcp.ur)
                if (jcp.os % i == 0) {
                    bcast_blocking = i;
                    break;
                }
        }
        bcast_blocking_max = bcast_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;

    } else if (jcp.prop_kind == backward_data) {
        // TODO: update evristic blocking

        int kernel_treshold = 192 * 1024;
        jcp.ur = nstl::min(28, jcp.os);
        if (jcp.iw <= 14 && jcp.ic * jcp.oc < kernel_treshold)
            jcp.ur = nstl::min(14, jcp.os);

        jcp.reduce_dim = jcp.oc;
        jcp.reduce_block = jcp.oc_block;

        jcp.load_dim = jcp.ic;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.os;
        jcp.bcast_block = jcp.ur;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.os * sizeof(float);
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.ic * sizeof(float);

        jcp.bcast_loop_output_step = jcp.ur * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = -1; // unused
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.oc_block * sizeof(float);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_load_step = jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.load_loop_iter_step = jcp.ic_block;

        load_blocking = nstl::min(528, jcp.load_dim);

        reduce_blocking = nstl::min(256, jcp.reduce_dim);

        const int USABLE_KNL_L2 = 400ULL * 1024;
        int os_block_size = sizeof(float) * reduce_blocking * jcp.ur;
        bcast_blocking
                = nstl::min(jcp.os, jcp.ur * (USABLE_KNL_L2 / os_block_size));

        bcast_blocking = nstl::min(196, bcast_blocking);
        bcast_blocking_max = bcast_blocking * 3 / 2;
        load_blocking_max = load_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;

    } else if (jcp.prop_kind == backward_weights) {
        jcp.reduce_dim = jcp.os;
        jcp.reduce_block = best_divider(jcp.is, 7, 16, true);

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        jcp.ur = jcp.bcast_block;

        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.ic_block * sizeof(float);
        jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.oc_block * sizeof(float);

        jcp.bcast_loop_output_step =
                                jcp.oc_block * jcp.ic_block * sizeof(float);
        jcp.bcast_loop_output_substep = jcp.oc_block * jcp.ur * sizeof(float);
        jcp.bcast_loop_bcast_step = jcp.ic_block * jcp.is * sizeof(float);
        jcp.bcast_loop_bcast_substep = jcp.ur * sizeof(float);

        jcp.load_loop_load_step = jcp.oc_block * jcp.os * sizeof(float);
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        load_blocking = best_divider(load_blocking, 16, load_blocking, false);
        load_blocking *= jcp.load_block;
        load_blocking_max = load_blocking;
        assert(jcp.load_dim % load_blocking == 0);

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        bcast_blocking = best_divider(bcast_blocking, 5, bcast_blocking, false);
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(jcp.bcast_dim % bcast_blocking == 0);

        // for reduction balance
        int num_jobs = div_up(jcp.load_dim, load_blocking) *
                        div_up(jcp.bcast_dim, bcast_blocking);
        const int nthreads = omp_get_max_threads();
        int threads_per_job = nstl::max(1, nthreads / num_jobs);
        reduce_blocking = div_up(jcp.mb * jcp.reduce_dim, jcp.reduce_block);
        reduce_blocking = div_up(reduce_blocking, threads_per_job);

        int max_reduce_blocking = KNL_L2_capacity /
                    ((bcast_blocking + load_blocking) * jcp.reduce_block);
        max_reduce_blocking = nstl::min(max_reduce_blocking,
                    (KNL_L1_capacity / (jcp.bcast_block)) / jcp.reduce_block);
        reduce_blocking = best_divider(reduce_blocking,
                    max_reduce_blocking - 2, max_reduce_blocking, true);
        reduce_blocking *= jcp.reduce_block;

        reduce_blocking_max = reduce_blocking * 3 / 2;
        assert(reduce_blocking % jcp.reduce_block == 0);
        assert(reduce_blocking_max % jcp.reduce_block == 0);

    } else
        return status::unimplemented;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);

    assert(jcp.bcast_block % jcp.ur == 0);
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

    return status::success;
}
}
}
}
