/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#include "cpu/aarch64/jit_brgemm_conv_trans_kernel.hpp"
#include "cpu/aarch64/jit_brgemm_conv_utils.hpp"

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;
using namespace nstl;
using namespace data_type;

namespace jit_sve_core_brgemm_conv_trans_kernel {

#define GET_OFF(field) offsetof(jit_brgemm_conv_trans_kernel_call_s, field)

jit_sve_core_brgemm_conv_trans_kernel_t::
        jit_sve_core_brgemm_conv_trans_kernel_t(
                const jit_brgemm_conv_conf_t &ajcp)
    : jcp(ajcp) {
    inp_dsz = jcp.src_dsz;
    ic_block_sz = inp_dsz * jcp.ic_block;
    dst_w_block = dst_w(jcp, jcp.ow_block);
    dst_stride = jcp.copy_block_only ? dst_w_block : jcp.iwp;
    dst_w_offset = jcp.kh_sets * jcp.kw_sets * ic_block_sz;
    dst_h_offset = dst_stride * dst_w_offset;
    iw_size = inp_dsz * jcp.ngroups * jcp.ic_without_padding;
    VL = cpu_isa_traits<sve_512>::vlen;
    n_vec = jcp.ic_block / jcp.simd_w;
    n_tail_vec = (jcp.ic_without_padding % jcp.ic_block) / jcp.simd_w;
}

int get_inp_size(int dst_size, int ext_k, int stride, int dilate) {
    const auto res = calculate_end_padding(0, dst_size, 0, stride, ext_k);
    return res;
}

int get_inp_start(int b, int b_size, int stride, int pad) {
    return b * b_size * stride - pad;
}

int jit_sve_core_brgemm_conv_trans_kernel_t::inp_w(int out_w, int kw) const {
    return get_inp_size(out_w, kw, jcp.stride_w, jcp.dilate_w);
}

int jit_sve_core_brgemm_conv_trans_kernel_t::inp_w(int out_w) const {
    return inp_w(out_w, jcp.ext_kw);
}

int jit_sve_core_brgemm_conv_trans_kernel_t::dst_w(
        const jit_brgemm_conv_conf_t &ajcp, int out_w) {
    int res = 0;
    if (ajcp.kw_sets > 1)
        res = get_inp_size(out_w, 1, 1, ajcp.dilate_w);
    else
        res = get_inp_size(out_w, ajcp.ext_kw, ajcp.stride_w, ajcp.dilate_w);
    if (ajcp.is_os_blocking) res = rnd_up(res, ajcp.stride_w);
    return res;
}

int jit_sve_core_brgemm_conv_trans_kernel_t::inp_w_start(int owb) const {
    return get_inp_start(owb, jcp.ow_block, jcp.stride_w, jcp.l_pad);
}

// use different vmovdqu32/16/8 due to case when tail mask used
void jit_sve_core_brgemm_conv_trans_kernel_t::load(
        const ZReg &x, const AdrNoOfs &addr) {
    if (one_of(jcp.src_dt, f32, s32))
        ldr(QReg(x.getIdx()), addr);
    else if (one_of(jcp.src_dt, bf16, f16)) {
        assert(!"Unknown type!\n");
    } else if (one_of(jcp.src_dt, data_type::s8, data_type::u8)) {
        assert(!"Unknown type!\n");
    } else
        assert(!"Unknown type!");
}

//for T_z
void jit_sve_core_brgemm_conv_trans_kernel_t::load(
        const ZReg &x, const PReg &p, const AdrNoOfs &addr) {
    if (one_of(jcp.src_dt, f32, s32)) {
        ld1rw(x.s, p, addr);
    } else if (one_of(jcp.src_dt, bf16, f16)) {
        assert(!"Unknown type!");
    } else if (one_of(jcp.src_dt, data_type::s8, data_type::u8)) {
        assert(!"Unknown type!");
    } else
        assert(!"Unknown type!");
}

void jit_sve_core_brgemm_conv_trans_kernel_t::store(
        const AdrNoOfs &addr, const ZReg &x) {
    if (one_of(jcp.src_dt, f32, s32))
        str(QReg(x.getIdx()), addr);
    else if (one_of(jcp.src_dt, bf16, f16))
        assert(!"Unknown type!");
    else if (one_of(jcp.src_dt, data_type::s8, data_type::u8))
        assert(!"Unknown type!");
    else
        assert(!"Unknown type!");
}

// for T_z
void jit_sve_core_brgemm_conv_trans_kernel_t::store(
        const AdrNoOfs &addr, const PReg &p, const ZReg &x) {
    if (one_of(jcp.src_dt, f32, s32)) {
        st1w(zmm_zero.s, p, addr);
    } else if (one_of(jcp.src_dt, bf16, f16))
        assert(!"Unknown type!");
    else if (one_of(jcp.src_dt, data_type::s8, data_type::u8))
        assert(!"Unknown type!");
    else
        assert(!"Unknown type!");
}

void jit_sve_core_brgemm_conv_trans_kernel_t::zero_ic_block(
        bool is_ic_tail, dim_t dst_off) {
    bool has_block_tail = (jcp.ic_block % jcp.simd_w);

    // TODO: use Xmm or Ymm moves for better small ic efficiency
    auto nvec = is_ic_tail ? n_tail_vec : n_vec;
    for (int iv = 0; iv < nvec; iv++) {
        add_imm(X_DEFAULT_ADDR, aux_dst_ptr, dst_off + iv * VL, X_TMP_0);
        store(ptr(X_DEFAULT_ADDR), zmm_zero);
    }
    //const auto last_dst_off = aux_dst_ptr + dst_off + nvec * VL;
    add_imm(X_DEFAULT_ADDR, aux_dst_ptr, dst_off + nvec * VL, X_TMP_0);
    if (is_ic_tail) {
        if (has_block_tail) {
            store(ptr(X_DEFAULT_ADDR), kblock_tail_mask, zmm_zero);
        } else {
            store(ptr(X_DEFAULT_ADDR), zmm_zero);
        }
    } else if (has_block_tail) {
        store(ptr(X_DEFAULT_ADDR), kblock_tail_mask, zmm_zero);
    }
}

void jit_sve_core_brgemm_conv_trans_kernel_t::copy_ic_block(bool is_ic_tail,
        dim_t inp_off = 0, dim_t dst_off = 0, bool do_load = true) {
    bool has_block_tail = (jcp.ic_block % jcp.simd_w);

    // TODO: use Xmm or Ymm moves for better small ic efficiency
    auto nvec = is_ic_tail ? n_tail_vec : n_vec;
    for (int iv = 0; iv < nvec; iv++) {
        if (do_load) {
            add_imm(X_DEFAULT_ADDR, aux_inp_ptr, inp_off + iv * VL, X_TMP_0);
            load(zmm_tmp, ptr(X_DEFAULT_ADDR));
        }
        add_imm(X_DEFAULT_ADDR, aux_dst_ptr, dst_off + iv * VL, X_TMP_0);
        store(ptr(X_DEFAULT_ADDR), zmm_tmp);
    }

    if (is_ic_tail) {
        if (do_load) {
            add_imm(X_DEFAULT_ADDR, aux_inp_ptr, inp_off + nvec * VL, X_TMP_0);
            load(zmm_tmp, ktail_mask, ptr(X_DEFAULT_ADDR));
        }
        if (has_block_tail) {
            add_imm(X_DEFAULT_ADDR, aux_dst_ptr, dst_off + nvec * VL, X_TMP_0);
            store(ptr(X_DEFAULT_ADDR), kblock_tail_mask, zmm_tmp);
        } else {
            add_imm(X_DEFAULT_ADDR, aux_dst_ptr, dst_off + nvec * VL, X_TMP_0);
            store(ptr(X_DEFAULT_ADDR), zmm_tmp);
        }
    } else if (has_block_tail) {
        if (do_load) {
            add_imm(X_DEFAULT_ADDR, aux_inp_ptr, inp_off + nvec * VL, X_TMP_0);
            load(zmm_tmp, kblock_tail_mask, ptr(X_DEFAULT_ADDR));
        }
        add_imm(X_DEFAULT_ADDR, aux_dst_ptr, dst_off + nvec * VL, X_TMP_0);
        store(ptr(X_DEFAULT_ADDR), kblock_tail_mask, zmm_tmp);
    }
}

void jit_sve_core_brgemm_conv_trans_kernel_t::generate() {
    preamble();

    ldr(inp_ptr, ptr(param1, uint32_t(GET_OFF(src))));
    ldr(dst_ptr, ptr(param1, uint32_t(GET_OFF(dst))));
    ldr(reg_hc, ptr(param1, uint32_t(GET_OFF(h_count))));
    ldr(reg_t_pad, ptr(param1, uint32_t(GET_OFF(t_pad))));
    ldr(reg_b_pad, ptr(param1, uint32_t(GET_OFF(b_pad))));
    ldr(reg_owb, ptr(param1, uint32_t(GET_OFF(owb))));
    ldr(reg_ic, ptr(param1, uint32_t(GET_OFF(ic))));

    eor(zmm_zero.d, zmm_zero.d, zmm_zero.d);

    if (jcp.ic_without_padding % jcp.ic_block) {
        int tail_size = (jcp.ic_without_padding % jcp.ic_block) % jcp.simd_w;
        set_preg(ktail_mask.s, tail_size, X_TMP_0, X_TMP_1);
    }

    if (jcp.ic_block % jcp.simd_w) {
        int block_tail_size = jcp.ic_block % jcp.simd_w;
        set_preg(kblock_tail_mask.s, block_tail_size, X_TMP_0, X_TMP_1);
    }

    auto icb_loop_body = [&](bool is_ic_tail) {
        Label kh_label, no_kh_label, icb_label;
        Label kh_tover_label, kh_bover_label;
        Label no_kh_tover_label, no_kh_bover_label;

        mov(aux_inp_ptr, inp_ptr);
        mov(aux_dst_ptr, dst_ptr);

        cmp_imm(reg_hc, 0, X_TMP_0);
        b(LE, no_kh_bover_label);

        cmp_imm(reg_t_pad, 0, X_TMP_0);
        b(LE, no_kh_tover_label);

        mov(kh_over, reg_t_pad);
        L(kh_tover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small ic
            for_(dim_t iw = 0; iw < dst_w_block; iw++)
            for (int kw = 0; kw < jcp.kw_sets; kw++)
                zero_ic_block(is_ic_tail, iw * dst_w_offset + kw * ic_block_sz);
            add_imm(aux_dst_ptr, aux_dst_ptr, dst_h_offset, X_TMP_0);

            subs(kh_over, kh_over, 1);
            cmp_imm(kh_over, 0, X_TMP_0);
            b(NE, kh_tover_label);
        }
        sub(reg_hc, reg_hc, reg_t_pad);
        L(no_kh_tover_label);

        cmp(reg_hc, reg_b_pad);
        b(LE, no_kh_label);

        L(kh_label);
        {
            copy_ow_block(is_ic_tail);
            auto inp_h_offset = jcp.iw * iw_size;

            add_imm(aux_inp_ptr, aux_inp_ptr, inp_h_offset, X_TMP_0);
            add_imm(aux_dst_ptr, aux_dst_ptr, dst_h_offset, X_TMP_0);

            sub_imm(reg_hc, reg_hc, 1, X_TMP_0);
            cmp(reg_hc, reg_b_pad);
            b(GT, kh_label);
        }
        L(no_kh_label);

        cmp_imm(reg_hc, 0, X_TMP_0);
        b(LE, no_kh_bover_label);

        L(kh_bover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small ic
            for_(dim_t iw = 0; iw < dst_w_block; iw++)
            for (int kw = 0; kw < jcp.kw_sets; kw++)
                zero_ic_block(is_ic_tail, iw * dst_w_offset + kw * ic_block_sz);
            add_imm(aux_dst_ptr, aux_dst_ptr, dst_h_offset, X_TMP_0);

            sub_imm(reg_hc, reg_hc, 1, X_TMP_0);
            cmp_imm(reg_hc, 0, X_TMP_0);
            b(NE, kh_bover_label);
        }
        L(no_kh_bover_label);

        // End IC Loop
        auto inp_cb_offset = ic_block_sz;
        auto dst_cb_offset = jcp.ihp * dst_h_offset;

        add_imm(inp_ptr, inp_ptr, inp_cb_offset, X_TMP_0);
        add_imm(dst_ptr, dst_ptr, dst_cb_offset, X_TMP_0);
    };

    for (int icb = 0; icb < jcp.nb_ic_blocking; icb++) {
        Label ic_tail_label, icb_continue_label;
        add_imm(reg_ic, reg_ic, jcp.ic_block, X_TMP_0);
        cmp_imm(reg_ic, jcp.ic, X_TMP_0);
        b(GT, ic_tail_label);

        icb_loop_body(false);
        b(icb_continue_label);

        L(ic_tail_label);
        icb_loop_body(true);

        L(icb_continue_label);
    }

    postamble();
}

void jit_sve_core_brgemm_conv_trans_kernel_t::copy_ow_block(bool is_ic_tail) {
    if (jcp.nb_ow == 1) {
        copy_ow_block_body(jcp.l_pad, jcp.ow_block, jcp.iw, is_ic_tail);
        return;
    }

    Label copy_block_done_label;

    int start_first_zero_block = -1;
    int end_first_zero_block = -1;
    int start_first_partial_block = -1;
    int end_first_partial_block = -1;
    int start_full_block = -1;
    int end_full_block = -1;
    int start_last_partial_block = -1;
    int end_last_partial_block = -1;

    const auto adj_iw = nstl::min(jcp.iw, jcp.iwp - jcp.l_pad);

    int ow_block_tail = jcp.ow % jcp.ow_block;

    for (int owb = 0; owb < jcp.nb_ow; owb++) {
        const auto inp_block = inp_w(jcp.ow_block);
        const auto inp_start = inp_w_start(owb);
        const auto inp_end = inp_start + inp_block;
        if (inp_start + inp_block < 0) {
            if (start_first_zero_block == -1) start_first_zero_block = owb;
            end_first_zero_block = owb;
        } else if (inp_start < 0) {
            if (start_first_partial_block == -1)
                start_first_partial_block = owb;
            end_first_partial_block = owb;
        } else if (inp_start < adj_iw) {
            if (inp_end <= adj_iw) {
                if (start_full_block == -1) start_full_block = owb;
                end_full_block = owb;
            } else {
                if (start_last_partial_block == -1)
                    start_last_partial_block = owb;
                end_last_partial_block = owb;
            }
        }
    }

    if (start_first_zero_block != -1) {
        Label skip_first_zero_blocks;
        cmp_imm(reg_owb, end_first_zero_block, X_TMP_0);
        b(GT, skip_first_zero_blocks);
        // zero block
        copy_ow_block_body(0, jcp.ow_block, 0, is_ic_tail);
        b(copy_block_done_label);

        L(skip_first_zero_blocks);
    }
    if (start_first_partial_block != -1) {
        for (int c = start_first_partial_block; c <= end_first_partial_block;
                c++) {
            int cur_ow_block = (c == jcp.nb_ow - 1 && ow_block_tail > 0)
                    ? ow_block_tail
                    : jcp.ow_block;
            const auto inp_block = inp_w(cur_ow_block);
            const auto inp_start = inp_w_start(c);
            const auto inp_end = inp_start + inp_block;
            const auto block_lpad = -inp_start;
            const auto block_len = nstl::min(adj_iw, inp_end);
            Label skip_first_partial_block;
            cmp_imm(reg_owb, c, X_TMP_0);
            b(NE, skip_first_partial_block);
            copy_ow_block_body(block_lpad, jcp.ow_block, block_len, is_ic_tail);
            b(copy_block_done_label);
            L(skip_first_partial_block);
        }
    }
    if (start_full_block != -1) {
        Label skip_full_blocks;
        cmp_imm(reg_owb, end_full_block, X_TMP_0);
        b(GT, skip_full_blocks);
        copy_ow_block_body(0, jcp.ow_block, inp_w(jcp.ow_block), is_ic_tail);
        b(copy_block_done_label);

        L(skip_full_blocks);
    }
    if (start_last_partial_block != -1) {
        for (int c = start_last_partial_block; c <= end_last_partial_block;
                c++) {
            int cur_ow_block = (c == jcp.nb_ow - 1 && ow_block_tail > 0)
                    ? ow_block_tail
                    : jcp.ow_block;
            const auto inp_block = inp_w(cur_ow_block);
            const auto inp_start = inp_w_start(c);
            const auto inp_end = inp_start + inp_block;
            const auto block_lpad = 0;
            const auto block_len = nstl::min(adj_iw, inp_end) - inp_start;
            Label skip_last_partial_block;
            cmp_imm(reg_owb, c, X_TMP_0);
            b(NE, skip_last_partial_block);
            copy_ow_block_body(block_lpad, cur_ow_block, block_len, is_ic_tail);
            b(copy_block_done_label);

            L(skip_last_partial_block);
        }
    }

    // if not any above case then owb is among last zero blocks
    // check is this needed and check may it be partial
    copy_ow_block_body(0, jcp.ow_block, 0, is_ic_tail);

    L(copy_block_done_label);
}

void jit_sve_core_brgemm_conv_trans_kernel_t::copy_ow_block_body(
        int lpad, int ow_len, int iw_len, bool is_ic_tail) {
    const auto dst_width = dst_w(jcp, ow_len);
    const auto iw_stride = jcp.kw_sets > 1 ? jcp.stride_w : 1;
    for_(int kw = 0; kw < jcp.kw_sets; kw++)
    for (dim_t ind_w = 0; ind_w < dst_width; ind_w++) {
        auto iw_idx = ind_w * iw_stride - lpad + kw * (jcp.dilate_w + 1);
        auto dst_off = ind_w * dst_w_offset + kw * ic_block_sz;
        if (iw_idx < 0 || iw_idx >= iw_len) {
            // left or right padding
            zero_ic_block(is_ic_tail, dst_off);
        } else {
            auto inp_off = iw_idx * iw_size;
            copy_ic_block(is_ic_tail, inp_off, dst_off, true);
        }
    }
}

jit_sve_core_brgemm_conv_rtus_kernel_t::jit_sve_core_brgemm_conv_rtus_kernel_t(
        const jit_brgemm_conv_conf_t &ajcp)
    : jit_sve_core_brgemm_conv_trans_kernel_t(ajcp) {
    ic_block_sz = inp_dsz * jcp.LDA; // output may or may not be zero padded
    dst_h_offset = jcp.iwp * ic_block_sz;
}

void jit_sve_core_brgemm_conv_rtus_kernel_t::generate() {
    preamble();

    const XReg &reg_khp = reg_hc;
    const XReg &reg_kwp = reg_owb;

    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(src), X_TMP_0);
    ldr(inp_ptr, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(dst), X_TMP_0);
    ldr(dst_ptr, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(h_count), X_TMP_0);
    ldr(reg_khp, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(owb), X_TMP_0);
    ldr(reg_kwp, ptr(X_DEFAULT_ADDR));

    if (jcp.ic_without_padding % jcp.ic_block) {
        int tail_size = (jcp.ic_without_padding % jcp.ic_block) % jcp.simd_w;
        set_preg(ktail_mask.s, tail_size, X_TMP_0, X_TMP_1);
    }

    if (jcp.ic_block % jcp.simd_w) {
        int block_tail_size = jcp.ic_block % jcp.simd_w;
        set_preg(kblock_tail_mask.s, block_tail_size, X_TMP_0, X_TMP_1);
    }

    assert(jcp.nb_ic_blocking == 1 && "TODO: support multi-batch case");

    for (int icb = 0; icb < jcp.nb_ic_blocking; icb++) {
        const bool is_ic_tail
                = (icb + 1) * jcp.ic_block > jcp.ic_without_padding;
        mov(aux_inp_ptr, inp_ptr);
        mov(aux_dst_ptr, dst_ptr);

        // Section 1: copy nw spatial elements in a row
        Label label_kwp_begin, label_kwp_end;
        cmp_imm(reg_kwp, 0, X_TMP_0);
        b(LE, label_kwp_end);
        L(label_kwp_begin);
        {
            copy_ic_block(is_ic_tail);

            auto inp_w_step = jcp.stride_w * iw_size;
            auto out_w_step = ic_block_sz;
            add_imm(aux_inp_ptr, aux_inp_ptr, inp_w_step, X_TMP_0);
            add_imm(aux_dst_ptr, aux_dst_ptr, out_w_step, X_TMP_0);

            sub_imm(reg_kwp, reg_kwp, 1, X_TMP_0);
            cmp_imm(reg_kwp, 0, X_TMP_0);
            b(NE, label_kwp_begin);
        }
        L(label_kwp_end);

        // Section 2: copy nh whole rows of OW spatial elements
        Label label_khp_begin, label_khp_end;
        cmp_imm(reg_khp, 0, X_TMP_0);
        b(LE, label_khp_end);
        L(label_khp_begin);
        {
            for (int ow = 0; ow < jcp.ow; ow++) {
                auto inp_w_off = ow * jcp.stride_w * iw_size;
                auto out_w_off = ow * ic_block_sz;
                copy_ic_block(is_ic_tail, inp_w_off, out_w_off);
            }

            auto inp_h_step = jcp.stride_h * jcp.iw * iw_size;
            auto out_h_step = jcp.ow * ic_block_sz;
            add_imm(aux_inp_ptr, aux_inp_ptr, inp_h_step, X_TMP_0);
            add_imm(aux_dst_ptr, aux_dst_ptr, out_h_step, X_TMP_0);

            sub_imm(reg_khp, reg_khp, 1, X_TMP_0);
            cmp_imm(reg_khp, 0, X_TMP_0);
            b(NE, label_khp_begin);
        }
        L(label_khp_end);
    }

    postamble();
}

} // namespace jit_sve_core_brgemm_conv_trans_kernel

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
