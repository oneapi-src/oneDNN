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

#include "bfloat16.hpp"
#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_barrier.hpp"

#include "jit_avx512_core_bf16_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {

using namespace format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace {

constexpr auto small_spatial = 14;

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(
            jcp.prop_kind, forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    // ow-threading is currently implemented for forward only
    // TODO: single code for fwd and bwd after ow-thr for bwd
    // meaningless switch was removed
    if (jcp.prop_kind == backward_data) {
        if (jcp.ndims < 5)
            jcp.loop_order = (w <= small_spatial && h <= small_spatial)
                    ? loop_cwgn
                    : loop_gncw;
        else
            jcp.loop_order = (w <= small_spatial && h <= small_spatial)
                    ? loop_cgn
                    : loop_gnc;
    } else {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial) ? loop_cwgn
                                                                    : loop_gncw;
    }
}
inline bool is_ow_threading_available(const jit_conv_conf_t &jcp) {
    /*is 1D conv */
    return (jcp.id == 1 && jcp.ih == 1 && jcp.kd == 1 && jcp.kh == 1);
}
inline bool is_ow_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}
inline bool is_iw_threading_available(const jit_conv_conf_t &jcp) {
    return one_of(jcp.ndims, 3, 4);
}
inline bool is_iw_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_iw > 1);
}
} // namespace

void jit_avx512_core_bf16_fwd_kernel::prepare_output(int ur_w) {
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
        }
}

void jit_avx512_core_bf16_fwd_kernel::store_output(int ur_w) {
    Label store_label;
    if (!isa_has_bf16(jcp.isa)) bf16_emu_->init_vcvtneps2bf16();

    if (jcp.with_sum) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                size_t aux_output_offset = get_output_offset(j, k);
                if (jcp.dst_dt == data_type::bf16) {
                    vpmovzxwd(zmm_prev_dst,
                            make_safe_addr(reg_out, aux_output_offset,
                                    reg_out_long_offt));
                    vpslld(zmm_prev_dst, zmm_prev_dst, 16);
                    vaddps(zmm, zmm_prev_dst);
                } else {
                    vaddps(zmm,
                            make_safe_addr(reg_out, aux_output_offset,
                                    reg_out_long_offt));
                }
            }
        }
    }

    if (jcp.with_bias) {
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            int bias_offset = jcp.typesize_bia * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                if (jcp.bia_dt == data_type::bf16) {
                    vpmovzxwd(zmm_bias,
                            EVEX_compress_addr(reg_bias, bias_offset));
                    vpslld(zmm_bias, zmm_bias, 16);
                    vaddps(zmm, zmm_bias);
                } else
                    vaddps(zmm, EVEX_compress_addr(reg_bias, bias_offset));
            }
        }
    }

    if (jcp.with_eltwise) {
        if (ur_w == jcp.ur_w) {
            eltwise_injector_->compute_vector_range(
                    0, jcp.nb_oc_blocking * jcp.ur_w);
        } else {
            for (int k = 0; k < jcp.nb_oc_blocking; k++)
                eltwise_injector_->compute_vector_range(
                        k * jcp.ur_w, k * jcp.ur_w + ur_w);
        }
    }

    L(store_label);
    if (jcp.dst_dt == data_type::f32) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++)
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                size_t aux_output_offset = jcp.typesize_out
                        * ((size_t)k * jcp.od * jcp.oh * jcp.ow + j)
                        * jcp.oc_block;
                auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

                vmovups(addr, zmm);
            }
    } else if (jcp.dst_dt == data_type::bf16) {
        if (isa_has_bf16(jcp.isa)) {
            for (int k = 0; k < jcp.nb_oc_blocking; k++) {
                int n_2bf2ps = (ur_w / 2) * 2, j = 0;
                for (j = 0; j < n_2bf2ps; j += 2) {
                    size_t aux_output_offset = jcp.typesize_out
                            * ((size_t)k * jcp.od * jcp.oh * jcp.ow + j)
                            * jcp.oc_block;
                    auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

                    auto zmm_str = zmm_inp(j, jcp.nb_oc_blocking);
                    vcvtne2ps2bf16(zmm_str, zmm_out(j + 1, k), zmm_out(j, k));
                    vmovups(addr, zmm_str);
                }
                if (j < ur_w) {
                    size_t aux_output_offset = jcp.typesize_out
                            * ((size_t)k * jcp.od * jcp.oh * jcp.ow + j)
                            * jcp.oc_block;
                    auto addr = EVEX_compress_addr(reg_out, aux_output_offset);
                    auto ymm_str = ymm_inp(j, jcp.nb_oc_blocking);
                    vcvtneps2bf16(ymm_str, zmm_out(j, k));
                    vmovups(addr, ymm_str);
                }
            }
        } else {
            for (int k = 0; k < jcp.nb_oc_blocking; k++)
                for (int j = 0; j < ur_w; j++) {
                    Zmm zmm = zmm_out(j, k);
                    size_t aux_output_offset = jcp.typesize_out
                            * ((size_t)k * jcp.od * jcp.oh * jcp.ow + j)
                            * jcp.oc_block;
                    auto addr = EVEX_compress_addr(reg_out, aux_output_offset);
                    Ymm ymm = ymm_inp(0, jcp.nb_oc_blocking);
                    bf16_emu_->vcvtneps2bf16(ymm, zmm);
                    vmovups(addr, ymm);
                }
        }
    } else
        assert(!"unsupported destination type");
}

void jit_avx512_core_bf16_fwd_kernel::compute_loop(
        int ur_w, int pad_l, int pad_r) {
    Label kh_label, kd_label;
    const size_t shift_kernel_ptr
            = (size_t)jcp.typesize_in * jcp.kw * jcp.oc_block * jcp.ic_block;
    const size_t shift_input_ptr = (size_t)jcp.typesize_in * (jcp.dilate_h + 1)
            * jcp.iw * jcp.ic_block;

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        mov(reg_kj, ptr[param1 + GET_OFF(kd_padding)]);
        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1)
                        < nstl::max(jcp.f_pad, jcp.back_pad)) {
            cmp(reg_kj, 0);
            je(skip_compute_loop, T_NEAR);
        }
    }
    mov(reg_kj, reg_kh);
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_compute_loop, T_NEAR);
    }

    // IC loop
    Label icb_label;
    mov(reg_icb, jcp.nb_ic);
    L(icb_label);

    if (jcp.ndims == 5) {
        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        mov(aux_reg_ker_d, reg_ker);
        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    } else {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
    }

    mov(reg_kj, reg_kh);

    L(kh_label);
    {
        for (int ki = 0; ki < jcp.kw; ki++) {
            int ow_start = get_ow_start(ki, pad_l);
            int ow_end = get_ow_end(ur_w, ki, pad_r);
            for (int ic = 0; ic < div_up(nstl::min(jcp.ic_block, jcp.ic), 2);
                    ic++) {
                for (int oi = ow_start; oi < ow_end; oi++) {
                    size_t input_offset = get_input_offset(ki, ic, oi, pad_l);
                    vpbroadcastd(zmm_inp(oi, jcp.nb_oc_blocking),
                            EVEX_compress_addr(aux_reg_inp, input_offset));
                }
                for (int kk = 0; kk < jcp.nb_oc_blocking; kk++) {
                    size_t kernel_offset = get_kernel_offset(ki, ic, kk, 0);
                    vmovups(zmm_wei,
                            EVEX_compress_addr(aux_reg_ker, kernel_offset));
                    for (int oi = ow_start; oi < ow_end; oi++) {
                        auto acc = zmm_out(oi, kk);
                        auto inp = zmm_inp(oi, jcp.nb_oc_blocking);
                        if (isa_has_bf16(jcp.isa)) {
                            vdpbf16ps(acc, zmm_wei, inp);
                        } else
                            bf16_emu_->vdpbf16ps(acc, zmm_wei, inp);
                    }
                }
            }
        }
        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.iw
                        * jcp.ic_block);
        add(aux_reg_ker_d,
                jcp.typesize_in * jcp.kw * jcp.kh * jcp.oc_block
                        * jcp.ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
    }

    // End of IC Loop
    size_t inp_step = (size_t)jcp.id * jcp.ih * jcp.iw * jcp.ic_block;
    size_t ker_step
            = (size_t)jcp.kd * jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;
    add(reg_inp, jcp.typesize_in * inp_step);
    add(reg_ker, jcp.typesize_in * ker_step);

    dec(reg_icb);
    cmp(reg_icb, 0);
    jg(icb_label, T_NEAR);

    sub(reg_inp, jcp.typesize_in * inp_step * jcp.nb_ic);
    sub(reg_ker, jcp.typesize_in * ker_step * jcp.nb_ic);

    L(skip_compute_loop);
    store_output(ur_w);
}

void jit_avx512_core_bf16_fwd_kernel::generate() {
    int iw = jcp.iw;
    int ow = jcp.ow;
    int ow_block = jcp.ow_block;
    int nb_ow = jcp.nb_ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    int inp_mult = jcp.ic_block;

    size_t inp_shift = (size_t)jcp.typesize_in * ur_w * stride_w * inp_mult;
    size_t out_shift = (size_t)jcp.typesize_out * ur_w * jcp.oc_block;

    int inp_shift_pad = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    int inp_shift_pad_second_block = -1 * jcp.typesize_in * l_pad * inp_mult;

    preamble();
    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    if (!is_ow_threading_on(jcp)) {
        // ow is being processed as a whole - with left and right paddings
        if (r_pad1 > 0) n_oi--;

        xor_(reg_oi, reg_oi);
        if (ow == ur_w) {
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            if (n_oi == 0) {
                compute_loop(ur_w, l_pad, r_pad1);
                add(reg_inp, inp_shift_pad);
                add(reg_out, out_shift);
                if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_pad); }
            } else {
                if (l_pad > 0) {
                    compute_loop(ur_w, l_pad, 0);
                    add(reg_inp, inp_shift_pad);
                    add(reg_out, out_shift);
                    inc(reg_oi);
                }
                if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        compute_loop(ur_w, 0, 0);
                        add(reg_inp, inp_shift);
                        add(reg_out, out_shift);

                        inc(reg_oi);
                        cmp(reg_oi, n_oi);
                        jl(ow_loop_label, T_NEAR);
                    }
                }
                if (r_pad1 > 0) {
                    compute_loop(ur_w, 0, r_pad1);
                    add(reg_inp, inp_shift);
                    add(reg_out, out_shift);
                }
                if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_pad); }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.

        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label;
        Label oi_loop_label, oi_loop_start_label, oi_loop_end_label;

        assert(ow_block % ur_w == 0);
        int n_oi_not_last_ow_block = ow_block / ur_w;
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;

        int n_oi_last_ow_block = (ow - ow_block * (nb_ow - 1)) / ur_w;

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded
                = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded)
            n_oi_last_ow_block--;
        else if (first_ow_block_padded)
            n_oi_first_ow_block--;
        else if (next_last_ow_block_padded)
            n_oi_next_last_ow_block--;

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, 0); // is that the first ow-block ?
        jg(middle_ow_blocks_label, T_NEAR);

        // the first ow block, compute left padding

        mov(reg_oi, n_oi_first_ow_block);
        if (l_pad > 0) {
            compute_loop(ur_w, l_pad, 0);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);
            dec(reg_oi);
        }
        jmp(oi_loop_label, T_NEAR);

        // middle or last ow block entry

        L(middle_ow_blocks_label);

        if (l_pad > 0) {
            // just to consider left padding, not compute
            add(reg_inp, inp_shift_pad_second_block);
        }

        // set number of iteration for oi-loop
        cmp(reg_owb, jcp.nb_ow - 1); // last ow-block ?
        mov(reg_oi, n_oi_last_ow_block);
        je(oi_loop_label, T_NEAR);
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        mov(reg_oi, n_oi_next_last_ow_block);
        je(oi_loop_label, T_NEAR);
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label);
        L(oi_loop_start_label);
        cmp(reg_oi, 0);
        jle(oi_loop_end_label, T_NEAR);

        compute_loop(ur_w, 0, 0);
        add(reg_inp, inp_shift);
        add(reg_out, out_shift);
        dec(reg_oi);
        jmp(oi_loop_start_label, T_NEAR);
        L(oi_loop_end_label);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);

        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        jl(end_label, T_NEAR);
        if (next_last_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        // that is last block
        if (!last_ow_block_padded) { jmp(tail_label, T_NEAR); }

        // last oi block with right padding
        L(last_oi_label);
        compute_loop(ur_w, 0, r_pad1);
        add(reg_inp, inp_shift);
        add(reg_out, out_shift);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        jl(end_label, T_NEAR);

        L(tail_label);
        if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_pad); }
        L(end_label);
    }
    postamble();

    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
}

bool jit_avx512_core_bf16_fwd_kernel::post_ops_ok(
        jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(); };

    switch (p.len_) {
        case 0: return true; // no post_ops
        case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
        case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
        default: return false;
    }

    return false;
}

void jit_avx512_core_bf16_fwd_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    using namespace memory_tracking::names;
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding) {
        assert(jcp.ngroups == 1);
        scratchpad.book(key_conv_padded_bias, jcp.typesize_bia * jcp.oc);
    }
}

status_t jit_avx512_core_bf16_fwd_kernel::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const memory_desc_wrapper &bias_d, const primitive_attr_t &attr,
        int nthreads) {
    using namespace prop_kind;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                        : bf16_emulation_t::get_isa();
    jcp.ver = ver_vnni;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];
    jcp.dst_dt = dst_d.data_type();

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
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

    const int regs = isa_has_bf16(jcp.isa) ? 31 /* expl_bcast case */ : 26;

    const int simd_w = cpu_isa_traits<avx512_core>::vlen / sizeof(float);

    auto dat_tag = utils::pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_tag
            = utils::pick(2 * ndims - 6 + with_groups, OIw8i16o2i, gOIw8i16o2i,
                    OIhw8i16o2i, gOIhw8i16o2i, OIdhw8i16o2i, gOIdhw8i16o2i);

    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);

    jcp.oc_block = simd_w;
    jcp.ic_block = simd_w;
    jcp.aligned_threads = 0;

    bool ok_to_pad_channels = jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    bool args_ok = true && jcp.src_tag == dat_tag && jcp.dst_tag == dat_tag
            && jcp.wei_tag == wei_tag && jcp.oc % jcp.oc_block == 0
            && jcp.ic % jcp.ic_block == 0;
    if (!args_ok) return status::unimplemented;

    args_ok = true && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
    }

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());

    jcp.bia_dt = jcp.with_bias ? bias_d.data_type() : data_type::undef;
    jcp.typesize_bia = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    jcp.kernel_kind = expl_bcast;
    jcp.nb_oc_blocking = nstl::min(4, jcp.nb_oc);
    for (; jcp.nb_oc_blocking > 1; jcp.nb_oc_blocking--) {
        int ur_w = regs / (jcp.nb_oc_blocking + 1);
        if (jcp.nb_oc % jcp.nb_oc_blocking == 0
                && (jcp.l_pad <= ur_w
                        && IMPLICATION(jcp.ow != 1, jcp.ow % ur_w != 1)))
            break;
    }

    jcp.ur_w = regs / (jcp.nb_oc_blocking + 1);
    if (jcp.ow < jcp.ur_w) jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ow_block = jcp.ow;
    if (is_ow_threading_available(jcp)) {
        const int L1_part = get_cache_size(1) * 5 / 8;
        int size_src_chunk = jcp.typesize_in * jcp.ic_block * jcp.ur_w;
        int size_dst_chunk = jcp.typesize_out * jcp.oc_block
                * jcp.nb_oc_blocking * jcp.ur_w;
        int size_wei_chunk = jcp.typesize_in * jcp.oc_block * jcp.ic_block
                * jcp.nb_oc_blocking * jcp.kw;
        int nurw = (L1_part - size_wei_chunk)
                / (size_dst_chunk + size_src_chunk);
        // current design of generate() requires ow_block >= 2 * ur_w
        jcp.ow_block = jcp.ur_w * nstl::max(2, nurw);
    }
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    int r_pad_no_tail = nstl::max(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw));
    if (jcp.l_pad > jcp.ur_w || r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    return status::success;
}

void jit_avx512_core_bf16_bwd_data_kernel::prepare_output(int ur_w) {
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
        }
    }
}

void jit_avx512_core_bf16_bwd_data_kernel::store_output(int ur_w) {
    if (!isa_has_bf16(jcp.isa)) bf16_emu_->init_vcvtneps2bf16();

    if (jcp.dst_dt == data_type::f32) {
        for (int k = 0; k < jcp.nb_ic_blocking; k++)
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                size_t aux_diff_src_offset = jcp.typesize_out
                        * ((size_t)k * jcp.id * jcp.ih * jcp.iw + j)
                        * jcp.ic_block;
                auto addr = EVEX_compress_addr(reg_src, aux_diff_src_offset);

                vmovups(addr, zmm);
            }
    } else if (jcp.dst_dt == data_type::bf16) {
        if (isa_has_bf16(jcp.isa)) {
            int store_idx = 0;
            const int max_regs = 32;
            const int free_regs_start_idx = jcp.ur_w * jcp.nb_ic_blocking;
            const int num_regs_available = max_regs - free_regs_start_idx;
            int reg_idx = 0;
            for (int k = 0; k < jcp.nb_ic_blocking; k++) {
                int n_2bf2ps = (ur_w / 2) * 2, j = 0;
                for (j = 0; j < n_2bf2ps; j += 2) {
                    reg_idx = free_regs_start_idx
                            + store_idx % num_regs_available;
                    assert(reg_idx < max_regs);
                    size_t aux_diff_src_offset = jcp.typesize_out
                            * ((size_t)k * jcp.id * jcp.ih * jcp.iw + j)
                            * jcp.ic_block;
                    auto addr
                            = EVEX_compress_addr(reg_src, aux_diff_src_offset);

                    auto zmm_str = Zmm(reg_idx);
                    vcvtne2ps2bf16(zmm_str, zmm_out(j + 1, k), zmm_out(j, k));
                    vmovups(addr, zmm_str);
                    store_idx++;
                }
                if (j < ur_w) {
                    reg_idx = free_regs_start_idx
                            + store_idx % num_regs_available;
                    assert(reg_idx < max_regs);

                    size_t aux_diff_src_offset = jcp.typesize_out
                            * ((size_t)k * jcp.id * jcp.ih * jcp.iw + j)
                            * jcp.ic_block;
                    auto addr
                            = EVEX_compress_addr(reg_src, aux_diff_src_offset);
                    auto ymm_str = Ymm(reg_idx);
                    vcvtneps2bf16(ymm_str, zmm_out(j, k));
                    vmovups(addr, ymm_str);
                    store_idx++;
                }
            }
        } else {
            for (int k = 0; k < jcp.nb_ic_blocking; k++)
                for (int j = 0; j < ur_w; j++) {
                    Zmm zmm = zmm_out(j, k);
                    size_t aux_diff_src_offset = jcp.typesize_out
                            * ((size_t)k * jcp.id * jcp.ih * jcp.iw + j)
                            * jcp.ic_block;
                    auto addr
                            = EVEX_compress_addr(reg_src, aux_diff_src_offset);
                    Ymm ymm = ymm_inp(0);
                    bf16_emu_->vcvtneps2bf16(ymm, zmm);
                    vmovups(addr, ymm);
                }
        }
    } else
        assert(!"unsupported diff_src type");
}

void jit_avx512_core_bf16_bwd_data_kernel::compute_loop(
        int ur_w, int l_overflow, int r_overflow) {
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;
    Label kh_label, skip_compute_label;

    auto kernel_offset = [=](int icb, int oc, int ki) {
        size_t blk_idx = (size_t)icb * jcp.kd * jcp.kh * jcp.kw + ki;
        size_t blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        size_t oc_offset = (size_t)oc * jcp.oc_block;
        return jcp.typesize_in * (blk_offset + oc_offset);
    };

    prepare_output(ur_w);

    if (jcp.ndims == 5) {
        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        cmp(reg_ki, 0);
        jle(skip_compute_label, T_NEAR);
    }

    cmp(reg_kh, 0);
    jle(skip_compute_label, T_NEAR);

    // OC loop
    Label ocb_label;
    mov(reg_ocb, jcp.nb_oc);
    L(ocb_label);

    if (jcp.ndims < 5) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
    }
    Label kd_label;
    if (jcp.ndims == 5) {
        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        mov(aux_reg_ker_d, reg_ker);

        L(kd_label);
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    mov(reg_kj, reg_kh);
    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            assert(stride_w != 1
                    || jj_start
                            == nstl::max(
                                    0, l_overflow - (kw - 1 - ki) * dilate_w));
            assert(stride_w != 1
                    || jj_end
                            == ur_w - nstl::max(0, r_overflow - ki * dilate_w));

            for (int oc = 0; oc < div_up(nstl::min(oc_block, jcp.oc), 2);
                    oc++) {
                for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                    assert((jj + jcp.l_pad - ki * dilate_w) % stride_w == 0);
                    size_t aux_dst_offset = jcp.typesize_in
                            * ((jj + jcp.l_pad - ki * dilate_w) / stride_w
                                            * oc_block
                                    + 2 * oc);
                    auto inp = zmm_inp(jj / stride_w);
                    vpbroadcastd(inp, ptr[aux_reg_dst + aux_dst_offset]);
                }
                for (int kk = 0; kk < jcp.nb_ic_blocking; kk++) {
                    size_t aux_kernel_offset = kernel_offset(kk, 2 * oc, ki);
                    vmovups(zmm_wei,
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));

                    for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                        auto inp = zmm_inp(jj / stride_w);
                        auto acc = zmm_out(jj, kk);

                        if (isa_has_bf16(jcp.isa)) {
                            vdpbf16ps(acc, zmm_wei, inp);
                        } else
                            bf16_emu_->vdpbf16ps(acc, zmm_wei, inp);
                    }
                }
            }
        }

        add(aux_reg_ker, jcp.typesize_in * stride_h * kw * oc_block * ic_block);
        sub(aux_reg_dst, jcp.typesize_in * (jcp.dilate_h + 1) * ow * oc_block);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.oh * jcp.ow
                        * ic_block);
        add(aux_reg_ker_d,
                jcp.typesize_in * jcp.stride_d * jcp.kw * jcp.kh * oc_block
                        * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
    }

    // End of OC Loop
    size_t diff_dst_step = (size_t)jcp.od * jcp.oh * jcp.ow * jcp.oc_block;
    size_t ker_step = (size_t)jcp.ic * jcp.kd * jcp.kh * jcp.kw * jcp.oc_block;
    add(reg_dst, jcp.typesize_in * diff_dst_step);
    add(reg_ker, jcp.typesize_in * ker_step);

    dec(reg_ocb);
    cmp(reg_ocb, 0);
    jg(ocb_label, T_NEAR);

    sub(reg_dst, jcp.typesize_in * diff_dst_step * jcp.nb_oc);
    sub(reg_ker, jcp.typesize_in * ker_step * jcp.nb_oc);

    L(skip_compute_label);
    store_output(ur_w);
}

void jit_avx512_core_bf16_bwd_data_kernel::generate() {
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int nb_iw = jcp.nb_iw;
    int iw_block = jcp.iw_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    size_t dst_shift = jcp.typesize_in * (ur_w / stride_w) * ic_block;
    size_t src_shift = jcp.typesize_out * ur_w * oc_block;

    preamble();

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);

    int l_overflow = nstl::max(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w);
    int r_overflow = nstl::max(
            0, ((kw - 1) * dilate_w - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow1 = nstl::max(0,
            ((kw - 1) * dilate_w - nstl::max(0, jcp.r_pad + ur_w_tail))
                    / stride_w);

    int body_l_overflow = 0, body_r_overflow = 0;
    int n_oi = iw / ur_w;
    int head_n_oi = 0, body_n_oi = 0, pretail_n_oi = 0, tail_n_oi = 0;
    int head_thread = 0, pretail_thread = 0, tail_thread = 0;
    bool threaded = is_iw_threading_on(jcp);
    Label head_label, body_label, pretail_label, tail_label, end_label;
    assert(n_oi > 0);

    if (r_overflow1 > 0) n_oi--;
    if (l_overflow > 0) n_oi--;
    if (n_oi < 0) {
        // l_overflow and r_overflow1 are handled in the same compute_loop.
        // Perform one iteration of body handling l_overflow and r_overflow1.
        body_l_overflow = l_overflow;
        body_r_overflow = r_overflow1;
        n_oi = 1;
        l_overflow = 0;
        r_overflow1 = 0;
    }

    if (!threaded) {
        if (n_oi > 1) { mov(reg_oi, n_oi); }
    } else {
        // Setup for threaded code generation, and jump into the correct
        // portion of code for execution.
        head_thread = 0;
        tail_thread = nb_iw - 1;
        pretail_thread = tail_thread;

        int base_n_oi = iw_block / ur_w;
        head_n_oi = l_overflow > 0 ? base_n_oi - 1 : base_n_oi;
        tail_n_oi = (iw - iw_block * (nb_iw - 1)) / ur_w;
        pretail_n_oi = tail_n_oi;
        if (r_overflow1 > 0) {
            if (tail_n_oi > 0) {
                pretail_n_oi--;
                tail_n_oi = pretail_n_oi;
            } else {
                // pretail_thread and tail_thread are different
                pretail_n_oi = base_n_oi - 1;
                pretail_thread = tail_thread - 1;
            }
            if (head_thread == pretail_thread) {
                head_n_oi--;
                pretail_n_oi = 0;
                tail_n_oi = 0;
            }
        }
        body_n_oi = (head_thread < pretail_thread - 1) ? base_n_oi : 0;

        // n_oi is used to determine how much control flow in the body portion
        // of the code needs generated. As such, n_oi needs to be set to the
        // maximum number of iterations it will be used the body code section.
        n_oi = nstl::max(body_n_oi, head_n_oi);
        n_oi = nstl::max(n_oi, pretail_n_oi);

        assert(iw_block % ur_w == 0);
        mov(reg_iwb, ptr[param1 + GET_OFF(iwb)]);

        if (head_n_oi != 0) mov(reg_oi, head_n_oi);
        cmp(reg_iwb, head_thread);
        je(head_label, T_NEAR);

        cmp(reg_iwb, pretail_thread);
        if (pretail_n_oi == 0) {
            je(pretail_label, T_NEAR);
        } else {
            mov(reg_oi, pretail_n_oi);
            je(body_label, T_NEAR);
        }
        if (pretail_thread != tail_thread) {
            cmp(reg_iwb, tail_thread);
            je(tail_label, T_NEAR);
        }
        if (body_n_oi != 0) {
            mov(reg_oi, body_n_oi);
            jmp(body_label, T_NEAR);
        } else {
            jmp(end_label, T_NEAR);
        }
    }
    L(head_label);
    if (l_overflow > 0) {
        compute_loop(ur_w, l_overflow, 0);
        if (threaded && head_n_oi == 0 && head_thread != pretail_thread)
            jmp(end_label, T_NEAR);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
    }
    L(body_label);
    if (n_oi > 0) {
        Label ow_loop_label;
        L(ow_loop_label);
        {
            compute_loop(ur_w, body_l_overflow, body_r_overflow);
            if (n_oi > 1 || r_overflow1 > 0 || ur_w_tail != 0) {
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
            }
            if (n_oi > 1) {
                sub(reg_oi, 1);
                jg(ow_loop_label, T_NEAR);
            }
        }
    }
    if (threaded) {
        cmp(reg_iwb, pretail_thread);
        jne(end_label, T_NEAR);
    }
    L(pretail_label);
    if (r_overflow1 > 0) {
        compute_loop(ur_w, 0, r_overflow1);
        if (ur_w_tail != 0) {
            if (threaded && tail_thread != pretail_thread)
                jmp(end_label, T_NEAR);
            else {
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);
            }
        }
    }
    L(tail_label);
    if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_overflow); }
    L(end_label);

    postamble();
}

status_t jit_avx512_core_bf16_bwd_data_kernel::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d, int nthreads) {
    const int simd_w = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                        : bf16_emulation_t::get_isa();
    jcp.ver = ver_vnni;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims - 2];
    jcp.iw = diff_src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];
    jcp.dst_dt = cd.diff_src_desc.data_type;
    jcp.nb_iw = 1;
    jcp.iw_block = jcp.iw;

    /* Dilated convolutions supported with unit strides only */
    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

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

    jcp.aligned_threads = 0;

    jcp.oc_block = simd_w;
    jcp.ic_block = simd_w;

    bool ok_to_pad_channels = jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    auto dat_tag = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_tag = with_groups
            ? pick(ndims - 3, gOIw8o16i2o, gOIhw8o16i2o, gOIdhw8o16i2o)
            : pick(ndims - 3, OIw8o16i2o, OIhw8o16i2o, OIdhw8o16i2o);
    jcp.src_tag = diff_src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = diff_dst_d.matches_one_of_tag(dat_tag);
    jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);
    bool args_ok = true && jcp.oc % jcp.oc_block == 0
            && jcp.ic % jcp.ic_block == 0 && jcp.src_tag == dat_tag
            && jcp.dst_tag == dat_tag && jcp.wei_tag == wei_tag;
    if (!args_ok) return status::unimplemented;

    args_ok = true && jcp.ic <= diff_src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_w = jcp.stride_w;

    /* Maximum number of registers available for result accumulation and delta
       dst data. One additional register is reserved for weights data. */
    const int max_regs
            = isa_has_bf16(jcp.isa) ? 31 : 26; /* In case of cpx emulation
                                                  additional 5 registers are
                                                  reserved */
    int l_overflow = nstl::max(
            0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad + jcp.iw % jcp.ur_w))
                    / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    jcp.typesize_in = types::data_type_size(diff_dst_d.data_type());
    jcp.typesize_out = types::data_type_size(diff_src_d.data_type());

    /* Find the best blocking with maximum number of compute instructions
       per ur_w * nb_ic_blocking compute loops. Number of required registers
       is num_regs = ur_w * nb_ic_blocking + ur_w / stride_w <= max_regs.
       ur_w must be divisible by stride_w */
    if (jcp.stride_w + 1 > max_regs) /* Minimal possible registers
                                         distribution exceeds max_regs */
        return status::unimplemented;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    {
        jcp.kernel_kind = expl_bcast;
        int best_compute_pipeline_length = 0;
        const int max_ic_blocks = 4;
        for (int b = 1; b <= max_ic_blocks; b++) {
            if (jcp.nb_ic % b != 0) continue;

            for (int u = jcp.stride_w; u * b + u / jcp.stride_w <= max_regs
                    && u < jcp.iw + jcp.stride_w;
                    u += jcp.stride_w) {
                int ur_w = nstl::min(u, jcp.iw);
                /* maximum 1 step with l_overflow so far */
                if (l_overflow * jcp.stride_w > ur_w && ur_w != jcp.iw)
                    continue;
                int pipeline_length = utils::div_up(ur_w, jcp.stride_w) * b;
                if (pipeline_length > best_compute_pipeline_length
                        || (pipeline_length == best_compute_pipeline_length
                                && jcp.ur_w < ur_w)) {
                    jcp.ur_w = ur_w;
                    jcp.nb_ic_blocking = b;
                    best_compute_pipeline_length = pipeline_length;
                }
            }
        }
        if (best_compute_pipeline_length == 0) /* can't find
                                                  appropriate blocking */
            return status::unimplemented;
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (is_iw_threading_available(jcp)) {
        int ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
        int work_units = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih;
        float no_iw_block_eff
                = (float)work_units / rnd_up(work_units, nthreads);

        // current design of generate() requires iw_block >= 2 * ur_w
        const int min_iw_block = jcp.ur_w * 2;
        int iw_threads = nthreads / math::gcd(work_units, nthreads);
        int iw_block = nstl::max(min_iw_block,
                rnd_up(jcp.iw, jcp.ur_w * iw_threads) / iw_threads);
        int nb_iw = div_up(jcp.iw, iw_block);

        float block_eff = (float)jcp.iw / rnd_up(jcp.iw, iw_block);
        work_units = jcp.ngroups * jcp.mb * ic_chunks * jcp.ih * nb_iw;
        float work_eff = (float)work_units / rnd_up(work_units, nthreads);
        float iw_block_eff = block_eff * work_eff;

        const int iw_thread_min_size = 16 * 128;
        const float iw_block_cost = 20.0;
        float block_overhead = nstl::max(0.0f, 1.0f - iw_block_cost / iw_block);

        bool iw_thread_useful = no_iw_block_eff < block_overhead * iw_block_eff
                && jcp.ic_block * jcp.iw > iw_thread_min_size;

        if (iw_thread_useful) {
            jcp.iw_block = iw_block;
            jcp.nb_iw = nb_iw;
        }
    }

    if (l_overflow * jcp.stride_w > jcp.ur_w) return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad + jcp.ur_w_tail))
                    / jcp.stride_w);
    bool tails_not_ok = false
            /* maximum 1 ur_w block with r_overflow so far */
            || r_overflow_no_tail * jcp.stride_w > jcp.ur_w
            /* ur_w must be a multiple of stride */
            || ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
            /* r_pad must not extend beyond ur_w_tail */
            || ((jcp.iw > jcp.ur_w) && (jcp.r_pad + jcp.ur_w_tail < 0));
    if (tails_not_ok) return status::unimplemented;

    pick_loop_order(jcp);

    return status::success;
}

const int jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::max_ur_w = 28;

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        od_step_comeback_pointers() {
    Label kd_comeback_label;
    mov(kj, reg_kd_count);
    L(kd_comeback_label);
    {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = jcp.tr_iw;
        sub(reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mult);
        sub(reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * jcp.ic_block
                        * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kd_comeback_label, T_NEAR);
    }
}
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        oh_step_comeback_pointers() {
    Label kh_comeback_label;
    mov(kj, reg_kh);
    L(kh_comeback_label);
    {
        int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
        int iw = jcp.tr_iw;
        sub(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mult);
        sub(reg_kernel,
                jcp.typesize_out * jcp.kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        compute_ic_block_step_extern(int ur_w, int pad_l, int pad_r,
                int ic_block_step, int input_offset, int kernel_offset,
                int output_offset, bool is_tail) {
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    auto zmm_ker = [=](int i_kw, int i_ic) {
        return Zmm(i_kw * ic_block_step + i_ic);
    };
    auto zmm_out = [=](int i_iw) {
        // TODO: move reg calc to global member funcs
        const int out_zmm_base_idx = 24;
        const int num_out_zmm_regs = !isa_has_bf16(jcp.isa) ? 2 : 4;
        return Zmm(out_zmm_base_idx + i_iw % num_out_zmm_regs);
    };

    auto ker_addr = [=](int i_kw, int i_ic) {
        size_t local_offset
                = jcp.typesize_out * (i_kw * ic_block + i_ic) * jcp.oc_block;
        return EVEX_compress_addr(reg_kernel, local_offset + kernel_offset);
    };
    auto inp_addr = [=](int i_iw, int i_ic, ptrdiff_t extra_offset = 0,
                            bool vnni_bcast = false) {
        int stride = jcp.tr_iw;
        int local_offset = jcp.typesize_in * (i_iw + i_ic * stride);
        if (vnni_bcast)
            return EVEX_compress_addr(reg_input,
                    local_offset + input_offset + extra_offset, true);
        else
            return EVEX_compress_addr(
                    reg_input, local_offset + input_offset + extra_offset);
    };
    auto out_addr = [=](int i_ur) {
        auto ow_per_oc = 2;
        return EVEX_compress_addr(reg_output,
                jcp.typesize_in * i_ur * oc_block * ow_per_oc + output_offset);
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto zmm = zmm_ker(i_kw, i_ic);
            vpxord(zmm, zmm, zmm);
        }
    assert(ur_w % 2 == 0);
    auto steps = ur_w / 2;

    const int str_w = jcp.stride_w;
    for (int s = 0; s < str_w; s++) {
        const int kw_start = s;
        assert(jcp.tr_iw % str_w == 0);
        const int inp_stride_w_shift = jcp.tr_iw / str_w;
        for (int i_ur = 0; i_ur < steps; i_ur++) {
            auto zmm = zmm_out(i_ur);
            vmovdqu16(zmm, out_addr(i_ur));

            for (int i_kw = kw_start; i_kw < kw; i_kw += str_w)
                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    int i_iw = 2 * i_ur + (i_kw * (jcp.dilate_w + 1)) / str_w
                            + s * inp_stride_w_shift;
                    if (!isa_has_bf16(jcp.isa)) {
                        auto inp = Zmm(26);
                        vpbroadcastd(inp, inp_addr(i_iw, i_ic, 0));
                        auto acc = zmm_ker(i_kw, i_ic);
                        auto wei = zmm_out(i_ur);
                        bf16_emu_->vdpbf16ps(acc, wei, inp);
                    } else
                        vdpbf16ps(zmm_ker(i_kw, i_ic), zmm_out(i_ur),
                                inp_addr(i_iw, i_ic, 0, true));
                }
        }
        for (int i_kw = kw_start; i_kw < kw; i_kw += str_w) {
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                auto addr = ker_addr(i_kw, i_ic);
                auto zmm = zmm_ker(i_kw, i_ic);
                vaddps(zmm, zmm, addr);
                vmovups(addr, zmm);
            }
        }
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::
        compute_ic_block_step_vpermw(int ur_w, int pad_l, int pad_r,
                int ic_block_step, int input_offset, int kernel_offset,
                int output_offset, bool is_tail) {
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;

    int load_buf_count = 0;
    int use_buf_count = 0;

    int dst_count = 0;
    int src_count = 0;

    int prv_i_offset = -1024;
    int prv_u_offset = -1024;

    int pipeline_length = (isa_has_bf16(jcp.isa))
            ? nstl::max(1, nstl::min(4, ur_w / 2))
            : 1;

    Reg64 reg_trans_tmp = r11;

    const int dst_off_reg = (!isa_has_bf16(jcp.isa)) ? 26 : 31;

    auto get_inp_offset = [=](int i_ur, int i_kw) {
        int local_offset = i_ur + i_kw - pad_l;
        return input_offset + jcp.typesize_in * local_offset * ic_block;
    };
    auto get_w_positions = [=](int i_ur, int i_kw, int &iw_1, int &iw_2) {
        iw_1 = (i_ur + i_kw);
        iw_2 = (i_ur + 1 == ur_w) ? -1 : (i_ur + 1) + i_kw;

        iw_1 = (iw_1 - pad_l < 0 || iw_1 > (ur_w - 1) + (kw - 1) - pad_r)
                ? -1
                : iw_1 - pad_l;
        iw_2 = (iw_2 - pad_l < 0 || iw_2 > (ur_w - 1) + (kw - 1) - pad_r)
                ? -1
                : iw_2 - pad_l;
    };
    auto check_borders = [=](int i_ur, int i_kw) {
        int iw_1, iw_2;
        get_w_positions(i_ur, i_kw, iw_1, iw_2);

        return (iw_1 == -1 && iw_2 == -1) ? false : true;
    };
    auto get_load_mask = [=](int i_ur, int i_kw, Opmask &load_mask) {
        int iw_1, iw_2;
        get_w_positions(i_ur, i_kw, iw_1, iw_2);

        bool rt = true;
        if (iw_1 != -1 && iw_2 != -1)
            load_mask = full_mask;
        else if (iw_1 != -1 && iw_2 == -1)
            load_mask = low_mask;
        else if (iw_1 == -1 && iw_2 != -1)
            load_mask = high_mask;
        else
            rt = false;

        return rt;
    };
    auto load_src = [=](int i_ur, int i_kw, int buf_offset, int count) {
        auto bcast_values = Zmm(25 + count % pipeline_length);

        Opmask load_mask;
        get_load_mask(i_ur, i_kw, load_mask);

        int inp_offset = get_inp_offset(i_ur, i_kw);
        vmovdqu16(bcast_values | load_mask | T_z, ptr[reg_input + inp_offset]);
        vpermw(bcast_values, perm, bcast_values);
        vmovups(EVEX_compress_addr(rsp, buf_offset), bcast_values);
    };
    auto load_dst = [=](int c) {
        int offset = jcp.typesize_in * c * 2 * oc_block + output_offset;

        Opmask load_mask;
        if (ur_w % 2 && c * 2 + 2 >= ur_w)
            load_mask = m_0000ffff;
        else
            load_mask = m_ffffffff;

        vmovdqu16(Zmm(dst_off_reg - c % pipeline_length) | load_mask | T_z,
                EVEX_compress_addr(reg_output, offset));
        vpermw(Zmm(dst_off_reg - c % pipeline_length), perm,
                Zmm(dst_off_reg - c % pipeline_length));
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vmovups(Zmm(i_kw * ic_block_step + i_ic),
                    EVEX_compress_addr(reg_kernel,
                            sizeof(float) * (i_kw * ic_block + i_ic)
                                            * jcp.oc_block
                                    + kernel_offset));

    if (jcp.ndims == 5)
        mov(EVEX_compress_addr(rsp, trans_tmp_offset), reg_trans_tmp);
    mov(reg_trans_tmp, dst_prm_table);
    vmovups(perm, ptr[reg_trans_tmp]);
    if (jcp.ndims == 5)
        mov(reg_trans_tmp, EVEX_compress_addr(rsp, trans_tmp_offset));

    for (src_count = 0; src_count < pipeline_length; src_count++) {
        int _i_ur = (src_count / kw) * 2;
        int _i_kw = src_count % kw;

        if (!check_borders(_i_ur, _i_kw)) continue;

        int i_offset = get_inp_offset(_i_ur, _i_kw);
        if (i_offset != prv_i_offset) {
            int load_buffer_offset = (load_buf_count++ % pipeline_length)
                    * jcp.typesize_in * 32;
            load_src(_i_ur, _i_kw, load_buffer_offset,
                    src_count % pipeline_length);
            prv_i_offset = i_offset;
        }
    }
    for (dst_count = 0; dst_count < pipeline_length; dst_count++) {
        load_dst(dst_count);
    }
    int use_buffer_offset = 0;
    for (int i_ur = 0; i_ur < ur_w; i_ur += 2) {
        for (int i_kw = 0; i_kw < kw; i_kw++) {
            int _i_ur = (src_count / kw) * 2;
            int _i_kw = src_count % kw;
            src_count++;

            int i_offset = get_inp_offset(_i_ur, _i_kw);

            if (check_borders(i_ur, i_kw)) {
                int u_offset = get_inp_offset(i_ur, i_kw);
                if (prv_u_offset != u_offset) {
                    use_buffer_offset = (use_buf_count++ % pipeline_length)
                            * jcp.typesize_in * 32;
                    prv_u_offset = u_offset;
                }
                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    if (!isa_has_bf16(jcp.isa)) {
                        auto zmm_src = Zmm(28);
                        vpbroadcastd(zmm_src,
                                ptr[rsp + use_buffer_offset
                                        + jcp.typesize_in * 2 * i_ic]);
                        bf16_emu_->vdpbf16ps(Zmm(i_kw * ic_block_step + i_ic),
                                Zmm(dst_off_reg - dst_count % pipeline_length),
                                zmm_src);
                    } else {
                        vdpbf16ps(Zmm(i_kw * ic_block_step + i_ic),
                                Zmm(dst_off_reg - dst_count % pipeline_length),
                                zword_b[rsp + use_buffer_offset
                                        + jcp.typesize_in * 2 * i_ic]);
                    }
                }
            }
            if ((_i_ur < ur_w && _i_kw < kw) && (check_borders(_i_ur, _i_kw))
                    && (i_offset != prv_i_offset)) {
                int load_buffer_offset = (load_buf_count++ % pipeline_length)
                        * jcp.typesize_in * 32;
                load_src(_i_ur, _i_kw, load_buffer_offset, src_count % 3);
                prv_i_offset = i_offset;
            }
        }
        if (dst_count * 2 < ur_w) load_dst(dst_count);
        dst_count++;
    }
    for (int i_kw = 0; i_kw < kw; i_kw++) {
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            int l_offset = jcp.typesize_out * (i_kw * ic_block + i_ic)
                    * jcp.oc_block;
            vmovups(EVEX_compress_addr(reg_kernel, l_offset + kernel_offset),
                    Zmm(i_kw * ic_block_step + i_ic));
        }
    }
}
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_ic_block_step(
        int ur_w, int pad_l, int pad_r, int ic_block_step, int input_offset,
        int kernel_offset, int output_offset, bool is_tail) {

    if (jcp.uses_permw_transposition)
        compute_ic_block_step_vpermw(ur_w, pad_l, pad_r, ic_block_step,
                input_offset, kernel_offset, output_offset, is_tail);
    else
        compute_ic_block_step_extern(ur_w, pad_l, pad_r, ic_block_step,
                input_offset, kernel_offset, output_offset, is_tail);
}
void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::
        compute_oh_step_unroll_ow_icblock(int ic_block_step) {
    Label kh_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;
    int iw = jcp.tr_iw;
    int ow = jcp.tr_ow;
    // XXX: is it possible to use jcp.r_pad here?
    int r_pad = (jcp.uses_permw_transposition)
            ? nstl::max(0,
                    calculate_end_padding(jcp.l_pad, ow, jcp.iw, jcp.stride_w,
                            calculate_extended_filter_size(
                                    jcp.kw, jcp.dilate_w)))
            : 0;
    int l_pad = (jcp.uses_permw_transposition) ? jcp.l_pad : 0;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step) {
            const int input_offset = (jcp.uses_permw_transposition)
                    ? jcp.typesize_in * i_b_ic
                    : jcp.typesize_in * i_b_ic * iw;
            compute_ic_block_step(jcp.ur_w, l_pad, r_pad, ic_block_step,
                    input_offset, jcp.typesize_out * i_b_ic * jcp.oc_block, 0,
                    i_b_ic + ic_block_step >= jcp.ic_block);
        }
        add(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul);
        add(reg_kernel, jcp.typesize_out * jcp.kw * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mul);
        add(aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::
        compute_oh_step_unroll_ow(int ic_block_step) {
    Label kh_label, ic_block_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;

    int ow = jcp.tr_ow;

    // XXX: is it possible to use jcp.r_pad here?
    int r_pad = (jcp.uses_permw_transposition) ? nstl::max(0,
                        calculate_end_padding(jcp.l_pad, ow, jcp.tr_iw,
                                jcp.stride_w,
                                calculate_extended_filter_size(
                                        jcp.kw, jcp.dilate_w)))
                                               : 0;
    int l_pad = (jcp.uses_permw_transposition) ? jcp.l_pad : 0;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        xor_(b_ic, b_ic);
        L(ic_block_label);
        {
            compute_ic_block_step(ow, l_pad, r_pad, ic_block_step, 0, 0, 0);
            size_t inp_icblk_stride = (jcp.uses_permw_transposition)
                    ? jcp.is_1stconv ? (size_t)jcp.ih * jcp.tr_iw * jcp.id : 1
                    : jcp.tr_iw;
            size_t input_offset
                    = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            safe_add(reg_input, input_offset, reg_long_offt);
            add(reg_kernel, jcp.typesize_out * ic_block_step * oc_block);
            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
        if (jcp.uses_permw_transposition) {
            if (jcp.is_1stconv) {
                size_t input_offset = (size_t)jcp.typesize_in * jcp.id * jcp.ih
                        * jcp.tr_iw * ic_block;
                safe_sub(reg_input, input_offset, reg_long_offt);
                add(reg_input,
                        jcp.typesize_in * (jcp.dilate_h + 1) * jcp.tr_iw);
            } else {
                add(reg_input,
                        jcp.typesize_in * ((jcp.dilate_h + 1) * jcp.tr_iw - 1)
                                * ic_block);
            }
        } else {
            if (jcp.dilate_h > 0)
                add(reg_input,
                        jcp.typesize_in * jcp.tr_iw * jcp.dilate_h * ic_block);
        }
        add(reg_kernel, jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.tr_iw
                        * inp_mul);
        add(aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_oh_step_common(
        int ic_block_step) {
    Label kh_label, ic_block_label, ow_block_label, kd_label;

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int inp_mul = !jcp.is_1stconv ? ic_block : 1;

    int ow = jcp.tr_ow;
    int l_pad = (jcp.uses_permw_transposition) ? jcp.l_pad : 0;
    // XXX: is it possible to use jcp.r_pad here?
    int r_pad = (jcp.uses_permw_transposition) ? nstl::max(0,
                        calculate_end_padding(jcp.l_pad, ow, jcp.tr_iw,
                                jcp.stride_w,
                                calculate_extended_filter_size(
                                        jcp.kw, jcp.dilate_w)))
                                               : 0;
    int stride_w = (jcp.uses_permw_transposition) ? jcp.stride_w : 1;

    int ur_w = nstl::min(ow, max_ur_w);
    int ur_w_trips = ow / ur_w;
    int ur_w_tail = ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0) || r_pad >= ur_w_tail) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }
    int inp_mult = (jcp.uses_permw_transposition)
            ? (jcp.is_1stconv) ? 1 : ic_block
            : 1;
    int input_comeback = (ur_w_trips * ur_w * stride_w - l_pad) * inp_mult;
    int output_comeback = ur_w_trips * ur_w * oc_block;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        xor_(b_ic, b_ic);
        L(ic_block_label);
        {
            if (l_pad != 0) {
                ur_w_trips--;
                compute_ic_block_step(ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
                add(reg_input,
                        jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult);
                add(reg_output, jcp.typesize_in * ur_w * oc_block);
            }

            if (ur_w_trips > 0) {
                xor_(reg_ur_w_trips, reg_ur_w_trips);
                L(ow_block_label);
                {
                    compute_ic_block_step(ur_w, 0, 0, ic_block_step, 0, 0, 0);
                    add(reg_input,
                            jcp.typesize_in * ur_w * stride_w * inp_mult);
                    add(reg_output, jcp.typesize_in * ur_w * oc_block);

                    inc(reg_ur_w_trips);
                    cmp(reg_ur_w_trips, ur_w_trips);
                    jl(ow_block_label, T_NEAR);
                }
            }

            if (ur_w_tail > 0) {
                compute_ic_block_step(
                        ur_w_tail, 0, r_pad, ic_block_step, 0, 0, 0, true);
            }

            sub(reg_input, jcp.typesize_in * input_comeback);
            sub(reg_output, jcp.typesize_in * output_comeback);

            int inp_icblk_stride = (jcp.uses_permw_transposition)
                    ? jcp.is_1stconv ? jcp.ih * jcp.tr_iw * jcp.id : 1
                    : jcp.tr_iw;

            size_t input_offset
                    = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            safe_add(reg_input, input_offset, reg_long_offt);
            add(reg_kernel, jcp.typesize_out * ic_block_step * oc_block);

            add(b_ic, ic_block_step);
            cmp(b_ic, jcp.ic_block);
            jl(ic_block_label, T_NEAR);
        }
        if (jcp.uses_permw_transposition) {
            if (jcp.is_1stconv) {
                size_t input_offset = (size_t)jcp.typesize_in * jcp.id * jcp.ih
                        * jcp.tr_iw * ic_block;
                safe_sub(reg_input, input_offset, reg_long_offt);
                add(reg_input,
                        jcp.typesize_in * (jcp.dilate_h + 1) * jcp.tr_iw);
            } else {
                add(reg_input,
                        jcp.typesize_in * ((jcp.dilate_h + 1) * jcp.tr_iw - 1)
                                * ic_block);
            }
        } else {
            if (jcp.dilate_h > 0)
                add(reg_input,
                        jcp.typesize_in * jcp.tr_iw * jcp.dilate_h * ic_block);
        }
        add(reg_kernel, jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.tr_iw
                        * inp_mul);
        add(aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_oh_step_disp() {
    int ic_block_step = jcp.ic_block_step;

    bool too_large_to_unroll = (jcp.kw > 1 || jcp.kh > 1 || jcp.kd > 1)
            && (jcp.stride_w > 1 || jcp.stride_h > 1 || jcp.stride_d > 1);

    int ow = jcp.tr_ow;
    if (jcp.ndims == 5) {
        /* NOTE: reg_kd_count = aux_reg_input = r12. The following order of
         * 'movs' must be guaranteed. */
        mov(ki, reg_kd_count);
        mov(EVEX_compress_addr(rsp, kd_count_offset), reg_kd_count);
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
    }
    if (jcp.kw <= 3 && ow <= 16 && !too_large_to_unroll) {
        compute_oh_step_unroll_ow_icblock(ic_block_step);
    } else if (ow <= max_ur_w) {
        compute_oh_step_unroll_ow(ic_block_step);
    } else {
        compute_oh_step_common(ic_block_step);
    }

    if (jcp.ndims == 5) {
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
        mov(reg_kd_count, EVEX_compress_addr(rsp, kd_count_offset));
        od_step_comeback_pointers();
    } else {
        oh_step_comeback_pointers();
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::maybe_zero_kernel() {
    Label skip_zeroing, zeroing_loop;

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jz(skip_zeroing, T_NEAR);

    Zmm zero = Zmm(0);
    vpxord(zero, zero, zero);
    xor_(reg_tmp, reg_tmp);
    L(zeroing_loop);
    {
        assert(jcp.oc_block * jcp.typesize_out
                == cpu_isa_traits<avx512_core>::vlen);
        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
            vmovups(ptr[reg_kernel + reg_tmp
                            + ic1 * jcp.oc_block * jcp.typesize_out],
                    zero);
        add(reg_tmp, jcp.ic_block * jcp.oc_block * jcp.typesize_out);
        cmp(reg_tmp,
                jcp.ic_block * jcp.oc_block * jcp.kw * jcp.kh * jcp.kd
                        * jcp.typesize_out);
        jnz(zeroing_loop);
    }

    L(skip_zeroing);
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::
        compute_oh_loop_common() {
    int b_pad = jcp.b_pad;
    int t_pad = jcp.t_pad;
    bool is_dilated = jcp.dilate_h != 0;
    int dilate_h = jcp.dilate_h + 1;
    int stride_h = jcp.stride_h;
    const int inp_mult = jcp.is_1stconv ? 1 : jcp.ic_block;
    int iw = jcp.tr_iw;
    int ow = jcp.tr_ow;
    Label oh_label, oh_label_end, oh_tpad_label, oh_tpad_tail_label,
            oh_bpad_label, oh_bpad_label_end, oh_dilate_label_shift,
            oh_dilate_label_noshift, oh_dilate_label_end;

    mov(reg_kh, jcp.kh);
    xor_(reg_ih_count, reg_ih_count);
    xor_(reg_oj, reg_oj);
    /* Compute 'top' edge */
    if (t_pad > 0) {
        const int kh_range = 1 + (jcp.kh - 1) * dilate_h;
        const int overflow
                = nstl::max(0, jcp.kh - div_up(t_pad + jcp.ih, dilate_h));
        const int underflow = div_up(t_pad, dilate_h);
        const int initial_inp_ker_overlap = jcp.kh - overflow - underflow;
        mov(reg_kh, initial_inp_ker_overlap);
        add(reg_kernel,
                jcp.typesize_out * underflow * jcp.kw * jcp.ic_block
                        * jcp.oc_block);
        // generate loop to process kernel while it remains within t_pad + ih
        if (kh_range < t_pad + jcp.ih) {
            if (is_dilated) {
                const int tail = t_pad % dilate_h;
                const int shift = tail == 0 ? 0 : dilate_h - tail;
                mov(reg_tmp, shift);
                if (tail != 0)
                    add(reg_input, jcp.typesize_in * shift * iw * inp_mult);
            }
            L(oh_tpad_label);
            {
                cmp(reg_oj, jcp.oh);
                jge(oh_label_end, T_NEAR);

                compute_oh_step_disp();
                add(reg_output, jcp.typesize_in * ow * jcp.oc_block);
                if (is_dilated) {
                    inc(reg_tmp);
                    cmp(reg_tmp, dilate_h);
                    jl(oh_dilate_label_shift, T_NEAR);
                    // unshift input as new kernel element enters
                    sub(reg_input,
                            jcp.typesize_in * (dilate_h - 1) * iw * inp_mult);
                    xor_(reg_tmp, reg_tmp);
                }
                // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
                sub(reg_kernel,
                        jcp.typesize_out * stride_h * jcp.kw * jcp.ic_block
                                * jcp.oc_block);
                add(reg_kh, stride_h);
                if (is_dilated) {
                    jmp(oh_dilate_label_noshift, T_NEAR);
                    L(oh_dilate_label_shift);
                    // shift input as old kernel element progresses
                    add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
                    L(oh_dilate_label_noshift);
                }
                inc(reg_oj);
                add(reg_ih_count, stride_h);

                // final number of kernel elements that overlap with input
                const int final_inp_ker_overlap
                        = nstl::min(jcp.kh, div_up(jcp.ih, dilate_h));
                cmp(reg_kh, final_inp_ker_overlap);
                jl(oh_tpad_label, T_NEAR);
            }
        }
        // need second loop to process kernel if it is larger than the input
        // (does not apply to dilations as they must have unit stride)
        if (kh_range >= jcp.ih
                        + (t_pad % stride_h == 0 ? stride_h
                                                 : t_pad % stride_h)) {
            assert(!is_dilated);
            mov(reg_kh, jcp.ih);
            L(oh_tpad_tail_label);
            {
                cmp(reg_oj, jcp.oh);
                jge(oh_label_end, T_NEAR);

                compute_oh_step_disp();
                add(reg_output, jcp.typesize_in * ow * jcp.oc_block);
                sub(reg_kernel,
                        jcp.typesize_out * stride_h * jcp.kw * jcp.ic_block
                                * jcp.oc_block);

                inc(reg_oj);
                add(reg_ih_count, stride_h);

                cmp(reg_ih_count, nstl::min(t_pad, jcp.oh * stride_h));
                jl(oh_tpad_tail_label, T_NEAR);
            }
        }
        // correct any excess shifts to kernel and input
        // (does not apply to dilations as they must have unit stride,
        //  kernel must fit inside input, and padding is smaller than input)
        if (t_pad <= jcp.oh * stride_h) {
            // kernel has moved beyond padding (adjust for stride effects)
            if (t_pad % stride_h != 0) {
                assert(!is_dilated);
                int inp_corr = stride_h - t_pad % stride_h;
                add(reg_kernel,
                        jcp.typesize_out * inp_corr * jcp.kw * jcp.ic_block
                                * jcp.oc_block);
                add(reg_input, jcp.typesize_in * inp_corr * iw * inp_mult);
            }
        } else {
            // kernel still overlaps padding (complete reset)
            assert(!is_dilated);
            sub(reg_kernel,
                    jcp.typesize_out * (t_pad - jcp.oh * stride_h) * jcp.kw
                            * jcp.ic_block * jcp.oc_block);
        }
    }

    cmp(reg_ih_count, jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h);
    jge(oh_label_end, T_NEAR);
    cmp(reg_oj, jcp.oh);
    jge(oh_label_end, T_NEAR);

    /* Compute middle block(s) */
    mov(reg_kh, jcp.kh);
    L(oh_label);
    {
        compute_oh_step_disp();
        add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
        add(reg_output, jcp.typesize_in * ow * jcp.oc_block);

        inc(reg_oj);
        add(reg_ih_count, stride_h);

        cmp(reg_ih_count, jcp.ihp - b_pad - (jcp.kh - 1) * dilate_h);
        jge(oh_label_end, T_NEAR);

        cmp(reg_oj, jcp.oh);
        jl(oh_label, T_NEAR);
    }
    L(oh_label_end);

    /* Compute bottom edge */
    if (b_pad > 0) {
        cmp(reg_oj, jcp.oh);
        jge(oh_bpad_label_end, T_NEAR);

        if (is_dilated) {
            mov(reg_kh, jcp.kh - 1); // assumes unit stride for dilations
            mov(reg_tmp, 0);
        } else {
            mov(reg_kh, jcp.ihp - b_pad);
            sub(reg_kh, reg_ih_count);
        }
        L(oh_bpad_label);
        {
            compute_oh_step_disp();
            add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
            add(reg_output, jcp.typesize_in * ow * jcp.oc_block);
            if (is_dilated) {
                inc(reg_tmp);
                cmp(reg_tmp, dilate_h);
                jl(oh_dilate_label_end, T_NEAR);
                xor_(reg_tmp, reg_tmp);
            }
            sub(reg_kh, stride_h);
            cmp(reg_kh, 0);
            jle(oh_bpad_label_end, T_NEAR);
            if (is_dilated) L(oh_dilate_label_end);

            inc(reg_oj);
            cmp(reg_oj, jcp.oh);
            jl(oh_bpad_label, T_NEAR);
        }
        L(oh_bpad_label_end);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32 ::
        compute_od_loop_common() {
    assert(jcp.harness == harness_3d_reduction);

    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    const int inp_mult = jcp.is_1stconv ? 1 : ic_block;
    int iw = jcp.tr_iw;
    int ow = jcp.tr_ow;

    const int input_backpad_overlap
            = div_up(jcp.id + jcp.f_pad - (jcp.kd - 1), jcp.stride_d);

    const size_t filter_shift
            = jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block;
    const size_t input_shift = jcp.typesize_in * jcp.ih * iw * inp_mult;
    const size_t output_shift = jcp.typesize_in * jcp.oh * ow * oc_block;

    const int kd_front_pad = nstl::max(0, jcp.f_pad);
    const int kd_back_pad = nstl::max(0, jcp.kd - jcp.f_pad - jcp.id);

    const int kd_padding = jcp.kd - kd_front_pad - kd_back_pad;
    const int kd_offset = nstl::min(jcp.kd - 1, kd_front_pad) * jcp.kh * jcp.kw
            * jcp.ic_block * jcp.oc_block * jcp.typesize_out;

    Label d_loop_label, loop_end_label, common_block_label, fpad_end_label,
            backpad_end_label, backpad_label;

    /* initially offset 'kd' by f_pad */
    add(reg_kernel, kd_offset);

    mov(reg_input_d, ptr[param + GET_OFF(src)]);
    mov(reg_output_d, ptr[param + GET_OFF(dst)]);

    mov(reg_kd_count, kd_padding);
    xor_(reg_d_index, reg_d_index);

    cmp(reg_kd_count, 0);
    jle(loop_end_label, T_NEAR); // no iterations along kd
    cmp(reg_d_index, jcp.od);
    jge(loop_end_label, T_NEAR); // no iterations along depth dimension

    L(d_loop_label);

    mov(reg_input, reg_input_d);
    mov(reg_output, reg_output_d);

    mov(EVEX_compress_addr(rsp, input_d_offset), reg_input_d);
    mov(EVEX_compress_addr(rsp, output_d_offset), reg_output_d);
    mov(EVEX_compress_addr(rsp, d_index_offset), reg_d_index);

    compute_oh_loop_common();

    mov(reg_input_d, EVEX_compress_addr(rsp, input_d_offset));
    mov(reg_output_d, EVEX_compress_addr(rsp, output_d_offset));
    mov(reg_d_index, EVEX_compress_addr(rsp, d_index_offset));

    /* Compute 'front' edge */
    if (jcp.f_pad > 0) {
        /* Check if within fpad region */
        cmp(reg_d_index, div_up(jcp.f_pad, jcp.stride_d));
        jge(fpad_end_label, T_NEAR);

        /* Fpad steps */
        sub(reg_kernel, filter_shift * jcp.stride_d);
        add(reg_kd_count, jcp.stride_d);

        /* Final number of kernel elements that overlap with input */
        const int inp_ker_overlap = nstl::min(jcp.kd, jcp.id);
        cmp(reg_kd_count, inp_ker_overlap);
        jle(common_block_label, T_NEAR);

        /* Correct any excess shifts to kernel and input */
        if (jcp.f_pad <= jcp.od * jcp.stride_d) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.f_pad % jcp.stride_d != 0) {
                int inp_corr = jcp.stride_d - jcp.f_pad % jcp.stride_d;
                add(reg_kernel, filter_shift * inp_corr);
                add(reg_input_d, input_shift * inp_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub(reg_kernel, (jcp.f_pad - jcp.od * jcp.stride_d) * filter_shift);
        }

        /* Apply correction */
        mov(reg_kd_count, inp_ker_overlap);
        jmp(common_block_label);

        L(fpad_end_label);
    }

    /* Compute bottom edge */
    if (jcp.back_pad > 0) {

        /* Check if within back_pad region */
        cmp(reg_d_index, input_backpad_overlap - 1);
        jl(backpad_end_label, T_NEAR);
        jg(backpad_label, T_NEAR);

        /* Execute overlap correction between the filter and the initial
         * back_pad region. */
        mov(reg_kd_count,
                jcp.id + jcp.f_pad - input_backpad_overlap * jcp.stride_d);
        jmp(backpad_end_label, T_NEAR);

        L(backpad_label);
        sub(reg_kd_count, jcp.stride_d);
        cmp(reg_kd_count, 0);
        jle(loop_end_label, T_NEAR);

        L(backpad_end_label);
    }

    /* Compute middle block */
    add(reg_input_d, input_shift * jcp.stride_d);

    /* Execute common block and loop */
    L(common_block_label);
    add(reg_output_d, output_shift);
    inc(reg_d_index);
    cmp(reg_d_index, jcp.od);
    jl(d_loop_label, T_NEAR);

    L(loop_end_label);
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::compute_loop() {
    maybe_zero_kernel();

    switch (jcp.harness) {
        case harness_3d_reduction: compute_od_loop_common(); break;
        case harness_mb_reduction: compute_oh_loop_common(); break;
        default: assert(!"Invalid harness type");
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::generate() {
    preamble();

    sub(rsp, stack_space_needed);

    Reg64 reg_mask_load = r11;

    mov(reg_input, ptr[param + GET_OFF(src)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    if (jcp.uses_permw_transposition) {
        int ilow_mask = (1 << jcp.ic_block_step) - 1;
        int ihigh_mask = ilow_mask << 16;
        int ifull_mask = ihigh_mask | ilow_mask;

        mov(reg_mask_load.cvt32(), ifull_mask);
        kmovd(full_mask, reg_mask_load.cvt32());

        mov(reg_mask_load.cvt32(), ilow_mask);
        kmovd(low_mask, reg_mask_load.cvt32());

        mov(reg_mask_load.cvt32(), ihigh_mask);
        kmovd(high_mask, reg_mask_load.cvt32());

        mov(reg_mask_load.cvt32(), 0xffffffff);
        kmovd(m_ffffffff, reg_mask_load.cvt32());

        mov(reg_mask_load.cvt32(), 0x0000ffff);
        kmovd(m_0000ffff, reg_mask_load.cvt32());
    }

    mov(reg_input, ptr[param + GET_OFF(src)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    compute_loop();

    add(rsp, stack_space_needed);

    postamble();

    if (jcp.uses_permw_transposition) {
        align(64);
        L(dst_prm_table);
        const uint16_t dst_prm_array[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20,
                5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13,
                29, 14, 30, 15, 31};

        for (size_t i = 0; i < 32; ++i)
            dw(dst_prm_array[i]);
    }
}

status_t jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &diff_weights_d,
        const memory_desc_wrapper &diff_bias_d,
        const memory_desc_wrapper &diff_dst_d) {
    const int simd_w = cpu_isa_traits<avx512_core>::vlen / sizeof(float);

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16
                                        : bf16_emulation_t::get_isa();
    jcp.ver = ver_vnni;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : diff_weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    int ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);

    bool ok = true
            // general condition to simplify dilations
            && IMPLICATION(jcp.dilate_d != 0, jcp.stride_d == 1)
            && IMPLICATION(jcp.dilate_h != 0, jcp.stride_h == 1)
            && IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1)
            // special condition to simplify dilations in compute_oh_loop_common
            && IMPLICATION(jcp.dilate_h != 0, ext_kh <= jcp.ih);
    if (!ok) return status::unimplemented;

    jcp.r_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw));
    jcp.b_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh));
    jcp.back_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd));

    /* XXX: currently, does not support stride_d > 1 or dilation_d > 0 */
    if (ndims == 5)
        if (jcp.stride_d > 1 || jcp.dilate_d > 0) return status::unimplemented;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.aligned_threads = 0;

    jcp.oc_block = simd_w;

    bool ok_to_pad_channels = jcp.ngroups == 1;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    auto dat_tag = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_tag = with_groups
            ? pick(ndims - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
            : pick(ndims - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o);
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = diff_dst_d.matches_one_of_tag(dat_tag);
    jcp.wei_tag = diff_weights_d.matches_one_of_tag(wei_tag);

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;
    jcp.bia_dt = jcp.with_bias ? diff_bias_d.data_type() : data_type::undef;
    jcp.typesize_bia = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    jcp.nb_oc = jcp.oc / jcp.oc_block;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_pad_h = ext_kh / 2;
    const bool boundaries_ok = true && jcp.l_pad < ext_kw && jcp.r_pad < ext_kw
            && jcp.t_pad <= max_pad_h && jcp.b_pad <= max_pad_h
            && jcp.f_pad < ext_kd && jcp.back_pad < ext_kd;
    if (!boundaries_ok) return status::unimplemented;

    /* yet another common check */
    if (jcp.kw > 14) return status::unimplemented;

    /* setting register strategy */
    for (int ur_w = nstl::min(max_ur_w, jcp.ow); ur_w > 0; --ur_w) {
        if (jcp.ow % ur_w == 0) {
            jcp.ur_w = ur_w;
            break;
        }
    }

    ok = true && jcp.src_tag == dat_tag && jcp.dst_tag == dat_tag
            && jcp.wei_tag == wei_tag;
    if (!ok) return status::unimplemented;
    jcp.wei_dt = diff_weights_d.data_type();

    jcp.ic_block = simd_w;
    if (ok_to_pad_channels) jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    if (true && one_of(ndims, 3, 4, 5)
            && everyone_is(
                    data_type::bf16, src_d.data_type(), diff_dst_d.data_type())
            && one_of(diff_weights_d.data_type(), data_type::f32,
                    data_type::bf16)) {
    } else {
        return status::unimplemented;
    }

    jcp.ic_block_step = jcp.kw <= 3 ? 8 : (jcp.kw < 7 ? 4 : 2);
    jcp.uses_permw_transposition
            = (jcp.stride_w != 1 || jcp.dilate_w != 0 || jcp.ic_block_step <= 4)
            ? false
            : true;
    const int tr_round = 4;
    // TODO: try to optimize required memory size
    int tr_pad
            = rnd_up(nstl::max(1, nstl::max(jcp.l_pad, jcp.r_pad)), tr_round);
    jcp.tr_iw = (jcp.uses_permw_transposition)
            ? jcp.iw
            : rnd_up(div_up(jcp.iw, jcp.stride_w) + tr_pad, tr_round)
                    * jcp.stride_w;

    jcp.tr_src_num_guard_elems = tr_pad; // upper bound
    jcp.tr_ow = (jcp.uses_permw_transposition) ? jcp.ow : rnd_up(jcp.ow, 2);
    jcp.ur_w = (jcp.uses_permw_transposition) ? jcp.ow : jcp.tr_ow;

    jcp.typesize_in = sizeof(bfloat16_t);
    jcp.typesize_out = sizeof(float);

    jcp.harness = ndims == 5 ? harness_3d_reduction : harness_mb_reduction;
    bool args_ok = true && jcp.ic % jcp.ic_block == 0
            && jcp.oc % jcp.oc_block == 0 && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= diff_weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= diff_weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    { // balancing
        int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
        balance(jcp, nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b);
        jcp.nthr = nthr;
        jcp.nthr_mb = nthr_mb;
        jcp.nthr_g = nthr_g;
        jcp.nthr_oc_b = nthr_oc_b;
        jcp.nthr_ic_b = nthr_ic_b;
    }

    return status::success;
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {

    if (!jcp.uses_permw_transposition) {
        // XXX: See the comment about tr_iw and guarding elements in
        // jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::init_conf()
#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        const size_t max_nthr = jcp.nthr_mb * jcp.ngroups * jcp.nb_ic;
#else
        const size_t max_nthr = jcp.nthr;
#endif // defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        const size_t min_tr_src_size_per_thr
                = jcp.id * jcp.ih * jcp.ic_block * jcp.tr_iw;
        const size_t tr_src_size = max_nthr * min_tr_src_size_per_thr
                + jcp.tr_src_num_guard_elems;
        scratchpad.book(key_conv_tr_src, jcp.typesize_in * tr_src_size);

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        /* prepare synchronization contexts */
        if (jcp.nthr_oc_b > 1) {
            const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
            scratchpad.book(key_conv_tr_src_bctx,
                    sizeof(simple_barrier::ctx_t) * tr_src_bctx_size);
        }
#endif // !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        const size_t tr_diff_dst_size = jcp.nthr_mb * jcp.ngroups * jcp.nb_oc
                * jcp.oc_block * jcp.tr_ow * jcp.oh * jcp.od;
#else
        const size_t tr_diff_dst_size
                = jcp.nthr * jcp.oc_block * jcp.tr_ow * jcp.oh * jcp.od;
#endif // defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        scratchpad.book(
                key_conv_tr_diff_dst, jcp.typesize_in * tr_diff_dst_size);

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        /* prepare synchronization contexts */
        if (jcp.nthr_ic_b > 1) {
            const size_t tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
            scratchpad.book(key_conv_tr_diff_dst_bctx,
                    sizeof(simple_barrier::ctx_t) * tr_diff_dst_bctx_size);
        }
#endif // defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    }

    if (jcp.nthr_mb > 1 || jcp.wei_dt == data_type::bf16) {
        const size_t wei_size
                = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw * jcp.kd;
        const size_t bia_size = jcp.ngroups * jcp.oc;

        const int num_wei_buffers
                = jcp.wei_dt == data_type::bf16 ? jcp.nthr_mb : jcp.nthr_mb - 1;

        const size_t wei_bia_reduction_size = wei_size + bia_size;

        scratchpad.book(key_conv_wei_bia_reduction,
                sizeof(float) * wei_bia_reduction_size * num_wei_buffers);

#if !defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
        scratchpad.book(
                key_conv_wei_bia_reduction_bctx, sizeof(simple_barrier::ctx_t));
#endif // defined(BF16_CONV_BWD_W_DOES_NOT_USE_BARRIERS)
    }

    if (jcp.with_bias) {
        const size_t dst_f32_size = (size_t)jcp.od * jcp.oh * jcp.ow
                * jcp.oc_block * jcp.typesize_out;
        scratchpad.book(key_conv_dst_bf16_convert_wsp, jcp.nthr * dst_f32_size);

        if (jcp.oc != jcp.oc_without_padding && jcp.bia_dt == data_type::f32)
            scratchpad.book(key_conv_padded_bias, jcp.typesize_bia * jcp.oc);
        else if (jcp.bia_dt == data_type::bf16)
            scratchpad.book(key_conv_bias_bf16_convert_wsp,
                    sizeof(float) * jcp.oc * jcp.ngroups);
    }
}

void jit_avx512_core_bf16_conv_bwd_weights_kernel_f32::balance(
        const jit_conv_conf_t &j, int &nthr_, int &nthr_mb_, int &nthr_g_,
        int &nthr_oc_b_, int &nthr_ic_b_) {
    nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

    const int max_threads = dnnl_get_max_threads();

    if (max_threads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }

    nthr_g_ = j.ngroups;
    const int nthr = max_threads / nthr_g_;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level optimizer
         * tries to minimize memory consumption. few notes:
         *  (n1) unclear why, but that essentially helps first convolution...
         *  (n2) assuming the reduction over minibatch is always there:
         *    - instead of 8 it should be 5 here (write ~= 2 read):
         *      kernel: temporal workspace 1 write
         *      reduction: 1 read from workspace and 1 write to the diff_wei
         *    - but experiments showed 8 works better than 5 or 6... */

        const dim_t src_type_size = 2;
        const dim_t wei_type_size = 4;
        const dim_t balance_threshold = 16;

        dim_t src_size
                = (dim_t)j.mb * j.ic * j.id * j.ih * j.iw * src_type_size;
        dim_t wei_size
                = (dim_t)j.oc * j.ic * j.kd * j.kh * j.kw * wei_type_size;

        dim_t r2 = nstl::min(balance_threshold,
                nstl::max(div_up(src_size, wei_size), (dim_t)1));
        dim_t r1 = nstl::min(balance_threshold,
                nstl::max(div_up(wei_size, src_size), (dim_t)1));

        const dim_t src_coef = (src_size <= wei_size) ? r2 : r1;
        const dim_t dst_coef = 1;
        const dim_t wei_coef = (src_size <= wei_size) ? r1 : r2;

        dim_t src_v = src_coef * div_up(j.mb, nthr_mb)
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_ic, nthr_ic_b)
                * j.ic_block * j.ih * j.iw * j.id / j.stride_d / j.stride_h
                / j.stride_w;
        dim_t wei_v = wei_coef * div_up(j.ngroups, nthr_g_)
                * div_up(j.nb_oc, nthr_oc_b) * div_up(j.nb_ic, nthr_ic_b) * j.kh
                * j.kw * j.kd * j.ic_block * j.oc_block;
        dim_t dst_v = dst_coef * div_up(j.mb, nthr_mb)
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_oc, nthr_oc_b)
                * j.oc_block * j.oh * j.ow * j.od;

        return src_v + dst_v + wei_v;
    };

    dim_t best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, j.mb * j.od);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, j.nb_oc);
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, j.nb_ic);

            dim_t mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

    if (nthr_mb_ > max_threads / 2 && nthr_mb_ < max_threads)
        nthr_mb_ = nstl::min(j.mb * j.od, max_threads);
    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;

    assert(nthr_ <= max_threads);
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
