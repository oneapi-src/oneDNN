/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "cpu/x64/jit_brgemm_conv_trans_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace nstl;
using namespace data_type;

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

jit_avx512_core_brgemm_conv_trans_kernel_t::
        jit_avx512_core_brgemm_conv_trans_kernel_t(
                const jit_brgemm_conv_conf_t &ajcp)
    : jcp(ajcp) {
    src_dsz = jcp.src_dsz;
    ic_block_sz = src_dsz * jcp.ic_block;
    out_h_offset = jcp.iwp * ic_block_sz;
    iw_size = src_dsz * jcp.ngroups * jcp.ic;
    VL = cpu_isa_traits<avx512_common>::vlen;
    n_vec = jcp.ic_block / jcp.simd_w;
}

// use different vmovdqu32/16/8 due to case when tail mask used
void jit_avx512_core_brgemm_conv_trans_kernel_t::load(
        const Xbyak::Xmm &x, const Xbyak::Address &addr) {
    if (one_of(jcp.src_dt, f32, s32))
        vmovdqu32(x, addr);
    else if (one_of(jcp.src_dt, bf16, f16))
        vmovdqu16(x, addr);
    else if (one_of(jcp.src_dt, s8, u8))
        vmovdqu8(x, addr);
    else
        assert(!"Unknown type!");
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::store(
        const Xbyak::Address &addr, const Xbyak::Xmm &x) {
    if (one_of(jcp.src_dt, f32, s32))
        vmovdqu32(addr, x);
    else if (one_of(jcp.src_dt, bf16, f16))
        vmovdqu16(addr, x);
    else if (one_of(jcp.src_dt, s8, u8))
        vmovdqu8(addr, x);
    else
        assert(!"Unknown type!");
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::zero_ic_block(
        int icb, dim_t out_off) {
    bool is_ic_tail = (icb + 1) * jcp.ic_block > jcp.ic;
    auto n_tail_vec = (jcp.ic % jcp.ic_block) / jcp.simd_w;
    bool has_block_tail = (jcp.ic_block % jcp.simd_w);

    // TODO: use Xmm or Ymm moves for better small ic efficiency
    auto nvec = is_ic_tail ? n_tail_vec : n_vec;
    for (int iv = 0; iv < nvec; iv++)
        store(ptr[aux_out_ptr + out_off + iv * VL], zmm_zero);
    // if is_ic_tail then not needed to process block tail
    if (is_ic_tail) {
        store(ptr[aux_out_ptr + out_off + nvec * VL] | ktail_mask | T_z,
                zmm_zero);
    } else if (has_block_tail) {
        store(ptr[aux_out_ptr + out_off + nvec * VL] | kblock_tail_mask | T_z,
                zmm_zero);
    }
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::copy_ic_block(
        int icb, dim_t inp_off, dim_t out_off) {
    bool is_ic_tail = (icb + 1) * jcp.ic_block > jcp.ic;
    auto n_tail_vec = (jcp.ic % jcp.ic_block) / jcp.simd_w;
    bool has_block_tail = (jcp.ic_block % jcp.simd_w);

    // TODO: use Xmm or Ymm moves for better small ic efficiency
    auto nvec = is_ic_tail ? n_tail_vec : n_vec;
    for (int iv = 0; iv < nvec; iv++) {
        load(zmm_tmp, ptr[aux_inp_ptr + inp_off + iv * VL]);
        store(ptr[aux_out_ptr + out_off + iv * VL], zmm_tmp);
    }
    // if is_ic_tail then not needed to process block tail
    if (is_ic_tail) {
        auto zmm_tmp_mask = zmm_tmp | ktail_mask | T_z;
        load(zmm_tmp_mask, ptr[aux_inp_ptr + inp_off + nvec * VL]);
        store(ptr[aux_out_ptr + out_off + nvec * VL] | ktail_mask | T_z,
                zmm_tmp);
    } else if (has_block_tail) {
        auto zmm_tmp_mask = zmm_tmp | kblock_tail_mask | T_z;
        load(zmm_tmp_mask, ptr[aux_inp_ptr + inp_off + nvec * VL]);
        store(ptr[aux_out_ptr + out_off + nvec * VL] | kblock_tail_mask | T_z,
                zmm_tmp);
    }
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::generate() {
    preamble();

    mov(inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(out_ptr, ptr[param1 + GET_OFF(dst)]);
    mov(khp, ptr[param1 + GET_OFF(kh_padding)]);
    mov(tover, ptr[param1 + GET_OFF(t_overflow)]);
    mov(bover, ptr[param1 + GET_OFF(b_overflow)]);
    mov(reg_owb, ptr[param1 + GET_OFF(owb)]);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    if (jcp.ic % jcp.ic_block) {
        int tail_size = (jcp.ic % jcp.ic_block) % jcp.simd_w;
        uint64_t mask = (UINT64_C(1) << tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(ktail_mask, reg_tmp);
    }

    if (jcp.ic_block % jcp.simd_w) {
        int block_tail_size = jcp.ic_block % jcp.simd_w;
        uint64_t mask = (UINT64_C(1) << block_tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(kblock_tail_mask, reg_tmp);
    }

    for (int icb = 0; icb < jcp.nb_ic_blocking; icb++) {
        Xbyak::Label kh_label, no_kh_label, icb_label;
        Xbyak::Label kh_tover_label, kh_bover_label;
        Xbyak::Label no_kh_tover_label, no_kh_bover_label;

        mov(aux_inp_ptr, inp_ptr);
        mov(aux_out_ptr, out_ptr);

        cmp(khp, 0);
        jle(no_kh_bover_label, T_NEAR); // nothing to do
        mov(khc, khp);

        cmp(tover, 0);
        jle(no_kh_tover_label, T_NEAR);

        mov(kh_over, tover);
        L(kh_tover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small ic
            for (dim_t iw = 0; iw < jcp.iwp; iw++)
                zero_ic_block(icb, iw * ic_block_sz);
            add(aux_out_ptr, out_h_offset);

            dec(kh_over);
            jnz(kh_tover_label, T_NEAR);
        }
        sub(khc, tover);
        L(no_kh_tover_label);

        cmp(khc, bover);
        jle(no_kh_label, T_NEAR);

        L(kh_label);
        {
            copy_row(icb);
            auto inp_h_offset = jcp.iw * iw_size;

            add(aux_inp_ptr, inp_h_offset);
            add(aux_out_ptr, out_h_offset);

            dec(khc);
            cmp(khc, bover);
            jg(kh_label, T_NEAR);
        }
        L(no_kh_label);

        cmp(khc, 0);
        jle(no_kh_bover_label, T_NEAR);

        L(kh_bover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small ic
            for (int iw = 0; iw < jcp.iwp; iw++)
                zero_ic_block(icb, iw * ic_block_sz);
            add(aux_out_ptr, out_h_offset);

            dec(khc);
            jnz(kh_bover_label, T_NEAR);
        }
        L(no_kh_bover_label);

        // End IC Loop
        auto inp_cb_offset = ic_block_sz;
        auto out_cb_offset = jcp.ihp * jcp.iwp * ic_block_sz;

        add(inp_ptr, inp_cb_offset);
        add(out_ptr, out_cb_offset);
    }

    postamble();
}

void jit_avx512_core_brgemm_conv_trans_kernel_t::copy_row(int icb) {
    if (jcp.nb_ow == 1) {
        copy_row_body(jcp.l_pad, jcp.ow_block, jcp.iw, icb);
    } else {
        auto get_iw_len_required = [&](int cur_ow_block, int cur_lpad) {
            return calculate_end_padding(cur_lpad, cur_ow_block, 0,
                    jcp.stride_w,
                    calculate_extended_filter_size(jcp.kw, jcp.dilate_w));
        };

        auto get_iw_len_limited = [&](int owb, int cur_ow_block, int cur_lpad) {
            auto len_req = get_iw_len_required(cur_ow_block, cur_lpad);
            if (owb < 0) return len_req;
            int ow_block_start = nstl::max(
                    0, owb * jcp.ow_block * jcp.stride_w - jcp.l_pad);
            return nstl::min(jcp.iw - ow_block_start, len_req);
        };

        int general_owb_cases = jcp.nb_ow;
        Xbyak::Label copy_row_done_label;
        bool special_first_block_case = jcp.l_pad > 0;
        if (special_first_block_case) {
            general_owb_cases--;
            Xbyak::Label skip_first_block_case_label;
            cmp(reg_owb, 0);
            jne(skip_first_block_case_label, T_NEAR);
            copy_row_body(jcp.l_pad, jcp.ow_block,
                    get_iw_len_limited(0, jcp.ow_block, jcp.l_pad), icb);
            jmp(copy_row_done_label, T_NEAR);
            L(skip_first_block_case_label);
        }
        bool special_last_block_case = false
                // has ow_block_tail
                || jcp.ow % jcp.ow_block != 0
                // there is no ow_block_tail but right padding exists
                || get_iw_len_limited(jcp.nb_ow - 1, jcp.ow_block, 0)
                        != get_iw_len_required(jcp.ow_block, 0);
        if (special_last_block_case) {
            general_owb_cases--;
            Xbyak::Label skip_last_block_case_label;
            cmp(reg_owb, jcp.nb_ow - 1);
            jne(skip_last_block_case_label, T_NEAR);
            int ow_block_tail = jcp.ow % jcp.ow_block;
            int cur_ow_block = ow_block_tail > 0 ? ow_block_tail : jcp.ow_block;
            copy_row_body(0, cur_ow_block,
                    get_iw_len_limited(jcp.nb_ow - 1, cur_ow_block, 0), icb);
            jmp(copy_row_done_label, T_NEAR);
            L(skip_last_block_case_label);
        }

        bool special_penult_block_case = true
                // if nb_ow = 2 and l_pad > 0 it's the same as
                // special_first_block_case
                && jcp.nb_ow >= (special_first_block_case ? 3 : 2)
                // right padding exists in penult block
                && get_iw_len_limited(jcp.nb_ow - 2, jcp.ow_block, 0)
                        != get_iw_len_required(jcp.ow_block, 0);
        if (special_penult_block_case) {
            general_owb_cases--;
            Xbyak::Label skip_penult_block_case_label;
            cmp(reg_owb, jcp.nb_ow - 2);
            jne(skip_penult_block_case_label, T_NEAR);
            copy_row_body(0, jcp.ow_block,
                    get_iw_len_limited(jcp.nb_ow - 2, jcp.ow_block, 0), icb);
            jmp(copy_row_done_label, T_NEAR);
            L(skip_penult_block_case_label);
        }

        if (general_owb_cases > 0) // general case
            copy_row_body(
                    0, jcp.ow_block, get_iw_len_required(jcp.ow_block, 0), icb);

        L(copy_row_done_label);
    }
}
void jit_avx512_core_brgemm_conv_trans_kernel_t::copy_row_body(
        int lpad, int ow_len, int iw_len, int icb) {
    // there are min(gen_kw, jcp.stride_w) continuous sets of input
    // data (for each stride idx), they are placed one by one
    // without additional padding
    const int gen_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    int gen_w = calculate_end_padding(0, ow_len, 0, jcp.stride_w, gen_kw);
    for (dim_t ind_w = 0; ind_w < gen_w; ind_w++) {
        auto iw_idx = ind_w - lpad;
        auto out_off = ind_w * ic_block_sz;
        if (iw_idx < 0 || iw_idx >= iw_len) {
            // left or right padding
            zero_ic_block(icb, out_off);
        } else {
            auto inp_off = iw_idx * iw_size;
            copy_ic_block(icb, inp_off, out_off);
        }
    }
}

} // namespace x64

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
