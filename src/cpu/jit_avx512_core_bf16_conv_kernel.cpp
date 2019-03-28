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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "bfloat16.hpp"

#include "cpu_barrier.hpp"

#include "jit_avx512_core_bf16_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace format_tag;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

namespace {

constexpr auto small_spatial = 14;

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(jcp.prop_kind,
                forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    // ow-threading is currently implemented for forward only
    // TODO: single code for fwd and bwd after ow-thr for bwd
    // meaningless switch was removed
    if (jcp.prop_kind == backward_data) {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial)
            ? loop_cgn : loop_gnc;
    } else {
        jcp.loop_order = (w <= small_spatial && h <= small_spatial)
            ? loop_cwgn : loop_gncw;
    }
}
inline bool is_ow_threading_available(const jit_conv_conf_t &jcp) {
    /*is 1D conv */
    return (jcp.ih == 1 && jcp.kh == 1);
}
inline bool is_ow_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}
}

void jit_avx512_core_bf16_fwd_kernel::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
        }
}

void jit_avx512_core_bf16_fwd_kernel::store_output(int ur_w)
{
    Label store_label;
    if (!isa_has_bf16(jcp.isa))
        bf16_emu_->init_vcvtneps2bf16();

    if (jcp.with_sum) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                size_t aux_output_offset = get_output_offset(j, k);
                if (jcp.dst_dt == data_type::bf16) {
                    vpmovzxwd(zmm_prev_dst,
                        make_safe_addr(reg_out, aux_output_offset, reg_out_long_offt));
                    vpslld(zmm_prev_dst, zmm_prev_dst, 16);
                    vaddps(zmm, zmm_prev_dst);
                } else {
                    vaddps(zmm,
                        make_safe_addr(reg_out, aux_output_offset, reg_out_long_offt));
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
            eltwise_injector_->compute_vector_range(0,
                    jcp.nb_oc_blocking * jcp.ur_w);
        } else {
            for (int k = 0; k < jcp.nb_oc_blocking; k++)
                eltwise_injector_->compute_vector_range(k * jcp.ur_w,
                        k * jcp.ur_w + ur_w);
        }
    }

    L(store_label);
    if (jcp.dst_dt == data_type::f32) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++)
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                size_t aux_output_offset = jcp.typesize_out *
                    ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
                auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

                vmovups(addr, zmm);
            }
    } else if (jcp.dst_dt == data_type::bf16) {
        if (isa_has_bf16(jcp.isa)) {
            for (int k = 0; k < jcp.nb_oc_blocking; k++) {
                int n_2bf2ps = (ur_w / 2) * 2, j = 0;
                for (j = 0; j < n_2bf2ps; j += 2) {
                    size_t aux_output_offset = jcp.typesize_out *
                        ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
                    auto addr = EVEX_compress_addr(reg_out, aux_output_offset);

                    auto zmm_str = zmm_inp(j, jcp.nb_oc_blocking);
                    vcvtne2ps2bf16(zmm_str, zmm_out(j+1, k),
                                            zmm_out(j, k));
                    vmovups(addr, zmm_str);
                }
                if (j < ur_w) {
                    size_t aux_output_offset = jcp.typesize_out *
                        ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
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
                    size_t aux_output_offset = jcp.typesize_out *
                        ((size_t)k * jcp.od * jcp.oh * jcp.ow + j) * jcp.oc_block;
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
        int ur_w, int pad_l, int pad_r)
{
    Label kh_label, kd_label;
    const size_t shift_kernel_ptr = (size_t)jcp.typesize_in * jcp.kw
                               * jcp.oc_block * jcp.ic_block;
    const size_t shift_input_ptr
            = (size_t)jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * jcp.ic_block;

    prepare_output(ur_w);

    // IC loop
    Label icb_label;
    mov(reg_icb, jcp.nb_ic);
    L(icb_label);

    mov(aux_reg_inp, reg_inp);
    mov(aux_reg_ker, reg_ker);

    Label skip_kh_loop, skip_kd_loop;

    mov(reg_kj, reg_kh);
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        cmp(reg_kj, 0);
        je(skip_kh_loop, T_NEAR);
    }

    L(kh_label); {
        for (int ki = 0; ki < jcp.kw; ki++) {
            int ow_start = get_ow_start(ki, pad_l);
            int ow_end = get_ow_end(ur_w, ki, pad_r);
            for (int ic = 0;
                 ic < div_up(nstl::min(jcp.ic_block, jcp.ic), 2); ic++) {
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

    L(skip_kh_loop);

    // End of IC Loop
    size_t inp_step = (size_t)jcp.ih * jcp.iw * jcp.ic_block;
    size_t ker_step = (size_t)jcp.kh * jcp.kw * jcp.oc_block * jcp.ic_block;
    add(reg_inp, jcp.typesize_in * inp_step);
    add(reg_ker, jcp.typesize_in * ker_step);

    dec(reg_icb);
    cmp(reg_icb, 0);
    jg(icb_label, T_NEAR);

    sub(reg_inp, jcp.typesize_in * inp_step * jcp.nb_ic);
    sub(reg_ker, jcp.typesize_in * ker_step * jcp.nb_ic);

    store_output(ur_w);
}

void jit_avx512_core_bf16_fwd_kernel::generate()
{
    int iw = jcp.iw;
    int ow = jcp.ow;
    int ow_block = jcp.ow_block;
    int nb_ow = jcp.nb_ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = jcp.dilate_w + 1;
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

    int r_pad = nstl::max(
            0, (ow - 1) * stride_w + (kw - 1) * dilate_w - (iw + l_pad - 1));
    int n_oi = ow / ur_w;
    int r_pad1 = (ur_w * n_oi - 1) * stride_w + (kw - 1) * dilate_w
            - (iw + l_pad - 1);

    if (!is_ow_threading_on(jcp)) {
        // ow is being processed as a whole - with left and right paddings
        if (r_pad1 > 0)
            n_oi--;

        xor_(reg_oi, reg_oi);
        if (ow == ur_w) {
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            if (n_oi == 0) {
                compute_loop(ur_w, l_pad, r_pad1);
                add(reg_inp, inp_shift_pad);
                add(reg_out, out_shift);
                if (ur_w_tail != 0) {
                    compute_loop(ur_w_tail, 0, r_pad);
                }
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
                if (ur_w_tail != 0) {
                    compute_loop(ur_w_tail, 0, r_pad);
                }
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

        int n_oi_last_ow_block = (ow - ow_block * (nb_ow-1)) / ur_w;

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded) n_oi_last_ow_block--;
        else if (first_ow_block_padded) n_oi_first_ow_block--;
        else if (next_last_ow_block_padded) n_oi_next_last_ow_block--;

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
        if (!last_ow_block_padded) {
            jmp(tail_label, T_NEAR);
        }

        // last oi block with right padding
        L(last_oi_label);
        compute_loop(ur_w, 0, r_pad1);
        add(reg_inp, inp_shift);
        add(reg_out, out_shift);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        jl(end_label, T_NEAR);

        L(tail_label);
        if (ur_w_tail != 0) {
            compute_loop(ur_w_tail, 0, r_pad);
        }
        L(end_label);
    }
    postamble();

    if (jcp.with_eltwise)
        eltwise_injector_->prepare_table();
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
        scratchpad.book(
                key_conv_padded_bias, jcp.typesize_bia * jcp.oc);
    }
}

status_t jit_avx512_core_bf16_fwd_kernel::init_conf(
            jit_conv_conf_t &jcp, const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
            const primitive_attr_t &attr, int nthreads)
{
    using namespace prop_kind;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    /*TODO: add 3d*/
    if (ndims > 4)
        return status::unimplemented;


    jcp = zero<decltype(jcp)>();
    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16 : avx512_core;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims-2];
    jcp.iw = src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims-2];
    jcp.ow = dst_d.dims()[ndims-1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims-2];
    jcp.kw = weights_d.dims()[with_groups + ndims-1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];
    jcp.dst_dt = dst_d.data_type();

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];

    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

    const int regs = isa_has_bf16(jcp.isa) ? 31 /* expl_bcast case */ : 26;

    const int simd_w = cpu_isa_traits<avx512_core>::vlen / sizeof(float);

    auto dat_tag = utils::pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto wei_tag = utils::pick(2 * ndims - 6 + with_groups,
            OIw8i16o2i, gOIw8i16o2i, OIhw8i16o2i, gOIhw8i16o2i,
            OIdhw8i16o2i, gOIdhw8i16o2i);

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
    bool args_ok = true
        && jcp.src_tag == dat_tag
        && jcp.dst_tag == dat_tag
        && jcp.wei_tag == wei_tag
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0;
    if (!args_ok)
        return status::unimplemented;

    args_ok = true
        && jcp.ic <= src_d.padded_dims()[1]
        && jcp.oc <= dst_d.padded_dims()[1]
        && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
        && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok)
        return status::unimplemented;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
        if (dst_d.data_type() == data_type::s32) return status::unimplemented;
    }

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.ver = ver_vnni;
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
    if (jcp.ow < jcp.ur_w)
        jcp.ur_w = jcp.ow;
    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    jcp.ow_block = jcp.ow;
    if (is_ow_threading_available(jcp)) {
        const int L1_part = get_cache_size(1) * 5 / 8;
        int size_src_chunk = jcp.typesize_in * jcp.ic_block * jcp.ur_w;
        int size_dst_chunk = jcp.typesize_out
            * jcp.oc_block * jcp.nb_oc_blocking * jcp.ur_w;
        int size_wei_chunk = jcp.typesize_in
            * jcp.oc_block * jcp.ic_block * jcp.nb_oc_blocking * jcp.kw;
        int nurw = (L1_part - size_wei_chunk)
            / (size_dst_chunk + size_src_chunk);
        // current design of generate() requires ow_block >= 2 * ur_w
        jcp.ow_block = jcp.ur_w * nstl::max(2, nurw);
    }
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

    int r_pad_no_tail = nstl::max(0, (jcp.ow - jcp.ur_w_tail - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1)
                    - (jcp.iw + jcp.l_pad - 1));
    if (jcp.l_pad > jcp.ur_w
        || r_pad_no_tail > jcp.ur_w)
        return status::unimplemented;

    pick_loop_order(jcp);

    return status::success;
}

void jit_avx512_core_bf16_bwd_data_kernel::prepare_output(int ur_w)
{
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Zmm zmm = zmm_out(j, k);
            vpxord(zmm, zmm, zmm);
        }
    }
}

void jit_avx512_core_bf16_bwd_data_kernel::store_output(int ur_w)
{
    if (!isa_has_bf16(jcp.isa))
        bf16_emu_->init_vcvtneps2bf16();

    if (jcp.dst_dt == data_type::f32) {
        for (int k = 0; k < jcp.nb_ic_blocking; k++)
            for (int j = 0; j < ur_w; j++) {
                Zmm zmm = zmm_out(j, k);
                size_t aux_diff_src_offset = jcp.typesize_out *
                    ((size_t)k * jcp.id * jcp.ih * jcp.iw + j) * jcp.ic_block;
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
                    size_t aux_diff_src_offset = jcp.typesize_out *
                        ((size_t)k * jcp.id * jcp.ih * jcp.iw + j) *
                        jcp.ic_block;
                    auto addr = EVEX_compress_addr(reg_src,
                                    aux_diff_src_offset);

                    auto zmm_str = Zmm(reg_idx);
                    vcvtne2ps2bf16(zmm_str, zmm_out(j+1, k),
                                            zmm_out(j, k));
                    vmovups(addr, zmm_str);
                    store_idx++;
                }
                if (j < ur_w) {
                    reg_idx = free_regs_start_idx
                        + store_idx % num_regs_available;
                    assert(reg_idx < max_regs);

                    size_t aux_diff_src_offset = jcp.typesize_out *
                        ((size_t)k * jcp.id * jcp.ih * jcp.iw + j) * jcp.ic_block;
                    auto addr = EVEX_compress_addr(reg_src, aux_diff_src_offset);
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
                    size_t aux_diff_src_offset = jcp.typesize_out *
                        ((size_t)k * jcp.id * jcp.ih * jcp.iw + j) * jcp.ic_block;
                    auto addr = EVEX_compress_addr(reg_src, aux_diff_src_offset);
                    Ymm ymm = ymm_inp(0);
                    bf16_emu_->vcvtneps2bf16(ymm, zmm);
                    vmovups(addr, ymm);
                }
        }
    } else
        assert(!"unsupported diff_src type");
}

void jit_avx512_core_bf16_bwd_data_kernel::compute_loop(
        int ur_w, int l_overflow, int r_overflow)
{
    int ow = jcp.ow;
    int kw = jcp.kw;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;
    int stride_h = jcp.stride_h;
    Label kh_label, skip_compute_label;

    auto kernel_offset = [=](int icb, int oc, int ki) {
        size_t blk_idx = (size_t)icb * jcp.kh * jcp.kw + ki;
        size_t blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        size_t oc_offset = (size_t)oc * jcp.oc_block;
        return jcp.typesize_in * (blk_offset + oc_offset);
    };

    prepare_output(ur_w);
    cmp(reg_kh, 0);
    jle(skip_compute_label, T_NEAR);

    // OC loop
    Label ocb_label;
    mov(reg_ocb, jcp.nb_oc);
    L(ocb_label);

    mov(aux_reg_dst, reg_dst);
    mov(aux_reg_ker, reg_ker);

    mov(reg_kj, reg_kh);
    L(kh_label); {
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = get_iw_start(ki, l_overflow);
            int jj_end = get_iw_end(ur_w, ki, r_overflow);
            assert(stride_w != 1
                    || jj_start == nstl::max(0,
                        l_overflow - (kw - 1 - ki) * dilate_w));
            assert(stride_w != 1
                    || jj_end == ur_w - nstl::max(0,
                        r_overflow - ki * dilate_w));

            for (int oc = 0;
                 oc < div_up(nstl::min(oc_block, jcp.oc), 2); oc++) {
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

    // End of OC Loop
    size_t diff_dst_step = (size_t)jcp.oh * jcp.ow * jcp.oc_block;
    size_t ker_step = (size_t)jcp.ic * jcp.kh * jcp.kw * jcp.oc_block;
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

void jit_avx512_core_bf16_bwd_data_kernel::generate()
{
    int iw = jcp.iw;
    int kw = jcp.kw;
    int ur_w = jcp.ur_w;
    int ic_block = jcp.ic_block;
    int oc_block = jcp.oc_block;
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
    int r_overflow = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad)) / stride_w);
    int r_overflow1 = nstl::max(0, ((kw - 1) * dilate_w
                    - nstl::max(0, jcp.r_pad) - ur_w_tail) / stride_w);

    int n_oi = iw / ur_w;
    if (r_overflow1 > 0) n_oi--;

    if (ur_w == iw) {
        compute_loop(ur_w, l_overflow, r_overflow);
    } else if (n_oi == 0) {
        compute_loop(ur_w, l_overflow, r_overflow1);
        add(reg_src, src_shift);
        add(reg_dst, dst_shift);
        if (ur_w_tail != 0)
            compute_loop(ur_w_tail, 0, r_overflow);
    } else {
        xor_(reg_oi, reg_oi);
        if (l_overflow > 0) {
            compute_loop(ur_w, l_overflow, 0);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);

            inc(reg_oi);
        }
        if ((l_overflow <= 0 && n_oi > 0)
            || (l_overflow > 0 && n_oi > 1)) {
            Label ow_loop_label;
            L(ow_loop_label); {
                compute_loop(ur_w, 0, 0);
                add(reg_src, src_shift);
                add(reg_dst, dst_shift);

                inc(reg_oi);
                cmp(reg_oi, n_oi);
                jl(ow_loop_label, T_NEAR);
            }
        }
        if (r_overflow1 > 0) {
            compute_loop(ur_w, 0, r_overflow1);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
        }
        if (ur_w_tail != 0) {
            compute_loop(ur_w_tail, 0, r_overflow);
        }
    }

    postamble();
}

status_t jit_avx512_core_bf16_bwd_data_kernel::init_conf(
        jit_conv_conf_t &jcp,
        const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d)
{
    const int simd_w = cpu_isa_traits<avx512_core>::vlen / sizeof(float);
    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    /*TODO: add 3d*/
    if (ndims > 4)
        return status::unimplemented;

    jcp.isa = mayiuse(avx512_core_bf16) ? avx512_core_bf16 : avx512_core;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims-2];
    jcp.iw = diff_src_d.dims()[ndims-1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims-2];
    jcp.ow = diff_dst_d.dims()[ndims-1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims-4];
    jcp.l_pad = cd.padding[0][ndims-3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims-4];
    jcp.stride_w = cd.strides[ndims-3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims-4];
    jcp.dilate_w = cd.dilates[ndims-3];
    jcp.dst_dt = cd.diff_src_desc.data_type;

    /* Dilated convolutions supported with unit strides only */
    if ((jcp.dilate_w != 0 && jcp.stride_w != 1)
            || (jcp.dilate_d != 0 && jcp.stride_d != 1)
            || (jcp.dilate_h != 0 && jcp.stride_h != 1))
        return status::unimplemented;

    jcp.r_pad = (jcp.ow - 1) * jcp.stride_w + (jcp.kw - 1) * (jcp.dilate_w + 1)
            - (jcp.iw + jcp.l_pad - 1);
    jcp.b_pad = (jcp.oh - 1) * jcp.stride_h + (jcp.kh - 1) * (jcp.dilate_h + 1)
            - (jcp.ih + jcp.t_pad - 1);
    jcp.back_pad = (jcp.od - 1) * jcp.stride_d
            + (jcp.kd - 1) * (jcp.dilate_d + 1) - (jcp.id + jcp.f_pad - 1);

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
    bool args_ok = true
        && jcp.oc % jcp.oc_block == 0
        && jcp.ic % jcp.ic_block == 0
        && jcp.src_tag == dat_tag
        && jcp.dst_tag == dat_tag
        && jcp.wei_tag == wei_tag;
    if (!args_ok)
        return status::unimplemented;

    args_ok = true
        && jcp.ic <= diff_src_d.padded_dims()[1]
        && jcp.oc <= diff_dst_d.padded_dims()[1]
        && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
        && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ur_w = jcp.stride_w;

    /* Maximun number of registers available for result accumulation and delta
       dst data. One additional register is reserved for weights data. */
    const int max_regs = isa_has_bf16(jcp.isa) ? 31 : 26; /* In case of cpx emulation
                                                  additional 5 registers are
                                                  reserved */
    int l_overflow = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - jcp.l_pad) / jcp.stride_w);
    int r_overflow1 = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad) - jcp.iw % jcp.ur_w) / jcp.stride_w);
    int n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow1 > 0) n_oi--;

    jcp.ver = ver_vnni;
    jcp.typesize_in = types::data_type_size(diff_dst_d.data_type());
    jcp.typesize_out = types::data_type_size(diff_src_d.data_type());

    /* Find the best blocking with maximum number of compute instructions
       per ur_w * nb_ic_blocking compute loops. Number of required registers
       is num_regs = ur_w * nb_ic_blocking + ur_w / stride_w <= max_regs.
       ur_w must be divisible by stride_w */
    if (jcp.stride_w + 1 > max_regs)  /* Minimal possible registers
                                         distribution exceeds max_regs */
        return status::unimplemented;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
    {
        jcp.kernel_kind = expl_bcast;
        int best_compute_pipeline_length = 0;
        const int max_ic_blocks = 4;
        for (int b = 1; b <= max_ic_blocks; b++)
        {
            if (jcp.nb_ic % b != 0)
                continue;

            for (int u = jcp.stride_w;
                 u * b + u / jcp.stride_w <= max_regs
                     && u < jcp.iw + jcp.stride_w;
                 u += jcp.stride_w)
            {
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

    jcp.loop_order = loop_gnc;

    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    if (l_overflow * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    int r_overflow_no_tail = nstl::max(0, ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max(0, jcp.r_pad) - jcp.ur_w_tail) / jcp.stride_w);
    if (r_overflow_no_tail * jcp.stride_w > jcp.ur_w)
        return status::unimplemented;
    if ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
        return status::unimplemented;

    pick_loop_order(jcp);

    return status::success;
}

}
}
}
// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
