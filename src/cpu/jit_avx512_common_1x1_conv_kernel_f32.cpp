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
    auto vreg_load = [=](int i_load, int i_fma) {
        return Zmm(utils::rnd_up(ur * load_loop_blk, jcp.fma_step)
                    + jcp.fma_step * i_load + i_fma);
    };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return Zmm(i_ur * load_loop_blk + i_load);
    };

    auto bias_ptr = [=](int i_load) {
        return EVEX_compress_addr(reg_bias_data,
                                  sizeof(float) * jcp.oc_block * i_load);
    };

    auto bcast_ptr = [=](int i_reduce, int i_ur, bool bcast) {
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
                                bcast);
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
            for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++)
                vmovups(vreg_load(i_load, i_fma), load_ptr(i_fma, i_load));
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
            assert(ur * load_loop_blk < 30);

            Label store_norelu;
            test(reg_reduce_pos_flag, REDUCE_FLAG_LAST);
            jz(store_norelu, T_NEAR);

            vpxord(zmm_zero, zmm_zero, zmm_zero);
            if (jcp.relu_negative_slope == 0) {
                zmm_relu_ns = zmm_zero;
            } else {
                mov(reg_relu_ns,
                        reinterpret_cast<size_t>(&jcp.relu_negative_slope));
                vbroadcastss(zmm_relu_ns, ptr[reg_relu_ns]);
            }

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
                if (jcp.use_vmovntps)
                    vmovntps(output_ptr(i_load, i_ur),
                        vreg_accum(i_load, i_ur));
                else
                    vmovups(output_ptr(i_load, i_ur),
                        vreg_accum(i_load, i_ur));
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
        int n_pf_out_l1 = jcp.use_vmovntps ? 0 : ur;

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
        assert(jcp.reduce_loop_unroll % jcp.fma_step == 0);
        for (int i_reduce = 0; i_reduce < jcp.reduce_loop_unroll;
                i_reduce += jcp.fma_step) {
            bool last_reduce = last_block
                        && i_reduce == jcp.reduce_loop_unroll - jcp.fma_step;
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    if (jcp.ver == ver_4fma)
                        v4fmaddps(vreg_accum(i_load, i_ur),
                                    vreg_load(i_load, 0),
                                    bcast_ptr(i_reduce, i_ur, false));
                    else
                        vfmadd231ps(vreg_accum(i_load, i_ur),
                                    vreg_load(i_load, 0),
                                    bcast_ptr(i_reduce, i_ur, true));
                    if (i_ur == ur - 1 && !last_reduce) {
                        for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++) {
                            vmovups(vreg_load(i_load, i_fma),
                                    load_ptr(i_reduce + jcp.fma_step + i_fma,
                                    i_load));
                        }
                    }
                    for (int i_fma = 0; i_fma < jcp.fma_step; i_fma++)
                        prefetch_callback(ur, i_reduce + i_fma, i_ur, i_load,
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

    static const int ur_cases_fma[] = {2, 4, 5, 8, 14, 28};
    static const int ur_cases_4fma[] = {2, 4, 6, 12, 28};
    const int *ur_cases = (jcp.ver == ver_4fma) ? ur_cases_4fma : ur_cases_fma;
    const int num_ur_cases
        = (jcp.ver == ver_4fma ? sizeof(ur_cases_4fma) : sizeof(ur_cases_fma))
        / sizeof(*ur_cases);

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
            if (ur_idx < num_ur_cases - 2) {
                cmp(reg_load_loop_work, simd_w);
                jle(load_loop_blk[0], T_NEAR);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    add(rsp, stack_space_needed);

    postamble();
}

status_t jit_avx512_common_1x1_conv_kernel_f32::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, bool with_relu,
        double relu_negative_slope, int nthreads, bool reduce_src)
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

    if (jcp.prop_kind != backward_weights && mayiuse(avx512_mic_4ops)
        && ((jcp.prop_kind == backward_data) ? jcp.oc_block : jcp.ic_block)
            % 4 == 0) {
        jcp.ver = ver_4fma;
        jcp.fma_step = 4;
    } else {
        jcp.ver = ver_fma;
        jcp.fma_step = 1;
    }

    jcp.ur = 1;

    int max_regs = 28;
    int min_regs = 8;

    int size_treshold;
    if (jcp.ver == ver_4fma)
        size_treshold = 28;
    else
        size_treshold = 14;

    for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
        if ((jcp.ih >= size_treshold && jcp.ih % ur_w == 0)
            || (jcp.ih < size_treshold && jcp.os % ur_w == 0)) {
            jcp.ur = ur_w;
            break;
        }
    }
    const int SMALL_SPATIAL = 7 * 7;
    const int BIG_REDUCE_DIM = 1024;

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

    jcp.load_grp_count = 1;
    jcp.use_vmovntps = true;

    const int L2_capacity = (512 * 1024) / sizeof(float);
    const int L1_capacity = (32 * 1024) / sizeof(float);

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

        jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

        load_blocking = jcp.load_dim;

        int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
        int nb_load = div_up(jcp.load_dim, jcp.load_block);
        int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

        int reduce_divider = 1;
        if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.reduce_dim > BIG_REDUCE_DIM)
            reduce_divider = 4;
        else if (jcp.bcast_dim > SMALL_SPATIAL
            && jcp.reduce_dim >= BIG_REDUCE_DIM)
            reduce_divider = 2;

        if (reduce_divider > 1) {
            jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
            jcp.use_vmovntps = false;
        }
        reduce_blocking = nstl::max(1, nb_reduce / reduce_divider);
        reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
        reduce_blocking *= jcp.reduce_block;

        int kernel_size = reduce_blocking * jcp.load_dim;
        int bcast_block_size = reduce_blocking * jcp.bcast_block;
        int L2_for_kernel = L2_capacity - bcast_block_size;
        jcp.load_grp_count = utils::div_up(kernel_size, L2_for_kernel);
        load_blocking = div_up(nb_load, jcp.load_grp_count) * jcp.load_block;
        kernel_size = utils::div_up(kernel_size, jcp.load_grp_count);

        bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
            div_up(nthreads, jcp.load_grp_count)) * jcp.bcast_block;
        bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
        int space_for_bcast = nstl::min(L1_capacity - 4 * jcp.load_block,
            L2_capacity - kernel_size);
        int bcast_in_cache =
            nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
        bcast_blocking = nstl::min(bcast_blocking,
            rnd_dn(bcast_in_cache, jcp.bcast_block));

        load_blocking_max = load_blocking;
        bcast_blocking_max = bcast_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;

    } else if (jcp.prop_kind == backward_data) {
        // TODO: update heuristic blocking

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

        jcp.loop_order = loop_lbr;

        load_blocking = jcp.load_dim;

        reduce_blocking = nstl::min(256, jcp.reduce_dim);

        const int USABLE_L2 = 400ULL * 1024;
        int os_block_size = sizeof(float) * reduce_blocking * jcp.ur;
        bcast_blocking = nstl::min(jcp.os,
            jcp.ur * nstl::max(1, USABLE_L2 / os_block_size));
        bcast_blocking = nstl::min(196, bcast_blocking);
        bcast_blocking = rnd_dn(bcast_blocking, jcp.bcast_block);

        bcast_blocking_max = bcast_blocking * 3 / 2;
        load_blocking_max = rnd_dn(load_blocking * 3 / 2, jcp.load_block);
        reduce_blocking_max = reduce_blocking;

        if (jcp.oh < 8 && jcp.mb < nthreads) {
            // XXX: there must be a function that does this...
            // Something similar to balance211 should work best
            // XXX probably not the best choice...
            const int nb_ic_div =
                nstl::min(div_up(jcp.load_dim, jcp.load_block),
                utils::div_up(nthreads, jcp.mb));
            int load_grp_size = utils::div_up(nthreads, nb_ic_div);
            jcp.load_grp_count = utils::div_up(nthreads, load_grp_size);
        }

        if (jcp.ver == ver_4fma && jcp.is < 15 * 15 && !reduce_src)
            jcp.loop_order = loop_rlb;

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

        int max_reduce_blocking = L2_capacity /
                    ((bcast_blocking + load_blocking) * jcp.reduce_block);
        max_reduce_blocking = nstl::min(max_reduce_blocking,
                    (L1_capacity / (jcp.bcast_block)) / jcp.reduce_block);
        reduce_blocking = best_divider(reduce_blocking,
                    max_reduce_blocking - 2, max_reduce_blocking, true);
        reduce_blocking *= jcp.reduce_block;

        reduce_blocking_max = rnd_dn(reduce_blocking * 3 / 2, jcp.reduce_block);
    } else
        return status::unimplemented;

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
    if (jcp.ver == ver_4fma) {
        assert(jcp.reduce_loop_unroll % jcp.fma_step == 0);
        assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);
    }

    assert(jcp.bcast_block % jcp.ur == 0);
    assert(jcp.load_dim % load_blocking == 0);

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
