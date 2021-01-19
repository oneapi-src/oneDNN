/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/jit_avx512_core_amx_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_call_s, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::utils;
using namespace Xbyak;

void jit_avx512_core_amx_copy_to_wbuffer_t::generate() {

    const bool is_bf16 = jcp.src_dt == data_type::bf16;

    // required for use of VPERMB instruction
    assert(IMPLICATION(!is_bf16, cpu().has(Xbyak::util::Cpu::tAVX512_VBMI)));
    assert(jcp.ic_block_int * jcp.typesize_in == 64);

    preamble();

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[param1 + GET_OFF(dst)]);

    // load permute indices from data section
    Label permute_index_table;
    mov(reg_tmp, permute_index_table);
    if (is_bf16)
        vmovdqu16(zmm_idx, ptr[reg_tmp]);
    else
        vmovdqu8(zmm_idx, ptr[reg_tmp]);

    const int vnni_width = is_bf16 ? 2 : 4;
    const int r = jcp.kh * jcp.kw * jcp.ic_without_padding;
    const int nb_r = div_up(r, vnni_width);
    const int rtail = (r % vnni_width) * jcp.oc_block;
    if (rtail > 0) {
        uint64_t mask = (UINT64_C(1) << rtail) - 1;
        mov(reg_tmp, mask);
        kmovq(kmask_load, reg_tmp);
    }
    const int nb_z = rnd_up(nb_r, jcp.ic_block);
    if (nb_r < nb_z) vpxord(zmm_zero, zmm_zero, zmm_zero);

    const int tile_size = jcp.ic_block_int * jcp.oc_block * jcp.typesize_in;
    const int ocb_src_step = r * jcp.oc_block * jcp.typesize_in;
    const int ocb_dst_step = rnd_up(ocb_src_step, tile_size);

    // reorder from ~Owhi16o -> ~OR16oVr with r := whi and V := vnni_width
    for (int g = 0; g < jcp.ngroups; g++) {
        for (int ocb = 0; ocb < jcp.nb_oc; ocb++) {
            int offset = 0;
            int rb = 0;
            for (; rb < nb_r; offset += 64, rb++) {
                auto zmm_src_tmp = (rtail > 0 && rb == nb_r - 1)
                        ? zmm_src | kmask_load | T_z
                        : zmm_src;
                if (is_bf16) {
                    vmovdqu16(zmm_src_tmp, ptr[reg_src + offset]);
                    vpermw(zmm_dst, zmm_idx, zmm_src);
                    vmovdqu16(ptr[reg_dst + offset], zmm_dst);
                } else {
                    vmovdqu8(zmm_src_tmp, ptr[reg_src + offset]);
                    vpermb(zmm_dst, zmm_idx, zmm_src);
                    vmovdqu8(ptr[reg_dst + offset], zmm_dst);
                }
            }
            for (; rb < nb_z; offset += 64, rb++) {
                if (is_bf16)
                    vmovdqu16(ptr[reg_dst + offset], zmm_zero);
                else
                    vmovdqu8(ptr[reg_dst + offset], zmm_zero);
            }
            add(reg_src, ocb_src_step);
            add(reg_dst, ocb_dst_step);
        }
    }

    postamble();

    align(64);
    L(permute_index_table);
    const uint8_t no = 16; // 16o
    const uint8_t nr = is_bf16 ? 2 : 4; // 2r or 4r
    for (uint8_t o = 0; o < no; ++o) {
        for (uint8_t r = 0; r < nr; r++) {
            const uint8_t index = o + r * UINT8_C(no);
            if (is_bf16)
                dw(index);
            else
                db(index);
        }
    }
}

void jit_avx512_core_amx_copy_to_pbuffer_t::copy_row_body(
        int lpad, int iw_len, int icb) {

    const bool is_bf16 = jcp.src_dt == data_type::bf16;
    int iwp_idx = 0;
    // there are min(gen_kw, jcp.stride_w) continuous sets of input
    // data (for each stride idx), they are placed one by one
    // without additional padding
    const bool are_sets_interleaved
            = IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1);
    const int gen_kw = (jcp.kw - 1) * (jcp.dilate_w + 1) + 1;
    const int num_sets = are_sets_interleaved ? jcp.n_stride_sets : jcp.kw;
    for (int set_idx = 0; set_idx < num_sets; set_idx++) {
        int set_width_padded = !jcp.is_pbuffer_strided
                ? (jcp.ow_block - 1) * jcp.stride_w + gen_kw
                : are_sets_interleaved ? jcp.ow_block - 1 + gen_kw / num_sets
                                + (set_idx < gen_kw % num_sets ? 1 : 0)
                                       : jcp.ow_block;
        for (int set_shift = 0; set_shift < set_width_padded;
                set_shift++, iwp_idx++) {
            int iw_idx = set_idx * (jcp.dilate_w + 1)
                    + set_shift * (jcp.is_pbuffer_strided ? jcp.stride_w : 1)
                    - lpad;
            size_t out_base_offset
                    = (size_t)jcp.typesize_in * iwp_idx * jcp.ic_block_int_np;
            if (iw_idx < 0 || iw_idx >= iw_len) {
                // left or right padding
                vmovups(ptr[reg_aux_out_ptr + out_base_offset], zmm_zero);
            } else if (jcp.is_nspc) {
                size_t inp_w_offset = (size_t)jcp.typesize_in * iw_idx
                        * jcp.ngroups * jcp.ic_without_padding;
                int ic = icb * jcp.ic_block_int_np;
                // TODO: use Xmm or Ymm moves for better small ic efficiency
                auto zmm_tmp_mask
                        = ic + jcp.ic_block_int <= jcp.ic_without_padding
                        ? zmm_tmp
                        : zmm_tmp | ktail_mask | T_z;
                if (is_bf16) {
                    vmovdqu16(
                            zmm_tmp_mask, ptr[reg_aux_inp_ptr + inp_w_offset]);
                    vmovdqu16(ptr[reg_aux_out_ptr + out_base_offset], zmm_tmp);
                } else {
                    vmovdqu8(zmm_tmp_mask, ptr[reg_aux_inp_ptr + inp_w_offset]);
                    vmovdqu8(ptr[reg_aux_out_ptr + out_base_offset], zmm_tmp);
                }
            } else {
                assert(is_bf16);
                size_t inp_w_offset
                        = (size_t)jcp.typesize_in * iw_idx * jcp.ic_block;
                for (int j = 0; j < jcp.ic_block_int_np / jcp.ic_block; j++) {
                    int ic = icb * jcp.ic_block_int_np + j * jcp.ic_block;
                    size_t inp_c_w_offset = (size_t)jcp.typesize_in * j * jcp.ih
                                    * jcp.iw * jcp.ic_block
                            + inp_w_offset;
                    if (ic + jcp.ic_block <= jcp.ic) {
                        vmovdqu16(
                                ymm_tmp, ptr[reg_aux_inp_ptr + inp_c_w_offset]);
                    } else {
                        vpxord(ymm_tmp, ymm_tmp, ymm_tmp);
                    }
                    size_t out_offset = out_base_offset
                            + (size_t)jcp.typesize_in * j * jcp.ic_block;
                    vmovdqu16(ptr[reg_aux_out_ptr + out_offset], ymm_tmp);
                }
            }
        }
    }
}

void jit_avx512_core_amx_copy_to_pbuffer_t::copy_row(int icb) {
    if (jcp.nb_ow == 1) {
        copy_row_body(jcp.l_pad, jcp.iw, icb);
    } else {
        auto get_iw_len_required = [&](int cur_ow_block, int cur_lpad) {
            return (cur_ow_block - 1) * jcp.stride_w
                    + (jcp.kw - 1) * (jcp.dilate_w + 1) + 1 - cur_lpad;
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
            copy_row_body(jcp.l_pad,
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
            copy_row_body(
                    0, get_iw_len_limited(jcp.nb_ow - 1, cur_ow_block, 0), icb);
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
            copy_row_body(
                    0, get_iw_len_limited(jcp.nb_ow - 2, jcp.ow_block, 0), icb);
            jmp(copy_row_done_label, T_NEAR);
            L(skip_penult_block_case_label);
        }

        if (general_owb_cases > 0) // general case
            copy_row_body(0, get_iw_len_required(jcp.ow_block, 0), icb);

        L(copy_row_done_label);
    }
}

void jit_avx512_core_amx_copy_to_pbuffer_t::copy_row_reduced_lowering() {
    assert(jcp.nb_ic_int == 1);
    assert(jcp.ic_block_int * jcp.typesize_in == 64);
    assert(jcp.is_nspc);

    auto load_mask = [=](int tail, Opmask kmask) {
        uint64_t mask = (UINT64_C(1) << tail) - 1;
        mov(reg_tmp, mask);
        kmovq(kmask, reg_tmp);
    };

    const bool is_bf16 = jcp.src_dt == data_type::bf16;
    const int inp_w_step
            = jcp.ngroups * jcp.ic_without_padding * jcp.typesize_in;
    const int inp_h_step = jcp.iw * inp_w_step;
    const int out_h_step = jcp.ic_without_padding * jcp.typesize_in;
    const int out_w_step = jcp.kh * out_h_step;
    const int tail_size = jcp.ic_without_padding % jcp.ic_block_int;
    if (tail_size > 0) load_mask(tail_size, ktail_mask);

    auto zero_it = [=](reg64_t tmp_out_ptr) {
        for (int ic = 0; ic < jcp.ic_without_padding; ic += jcp.ic_block_int) {
            const int offset = ic * jcp.typesize_in;
            const bool masked = ic + jcp.ic_block_int > jcp.ic_without_padding;
            Zmm zmm = masked ? zmm_zero | ktail_mask : zmm_zero;
            if (is_bf16)
                vmovdqu16(ptr[tmp_out_ptr + offset], zmm);
            else
                vmovdqu8(ptr[tmp_out_ptr + offset], zmm);
        }
    };

    // pointer to 1st needed element in src buffer
    mov(reg_inp_ptr, ptr[param1 + GET_OFF(src)]);
    // pointer to 1st needed element in dst buffer
    mov(reg_out_ptr, ptr[param1 + GET_OFF(dst)]);

    // total number of rows to copy
    mov(reg_kht, ptr[param1 + GET_OFF(kh_offset)]);

    // number of rows of src buffer to copy
    mov(reg_khp, ptr[param1 + GET_OFF(kh_padding)]);
    // number of zero-padded rows above src buffer to copy
    mov(reg_tov, ptr[param1 + GET_OFF(t_overflow)]);
    // number of zero-padded rows below src buffer to copy
    mov(reg_bov, ptr[param1 + GET_OFF(b_overflow)]);

    // number of columns of src buffer to copy
    mov(reg_kwp, ptr[param1 + GET_OFF(kw_padding)]);
    // number of zero-padded columns before src buffer to copy
    mov(reg_lov, ptr[param1 + GET_OFF(f_overflow)]);
    // number of zero-padded columns before src buffer to copy
    mov(reg_rov, ptr[param1 + GET_OFF(back_overflow)]);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    { // Handle Left Overflow
        Label label_lov, label_lov_skip;
        test(reg_lov, reg_lov);
        jz(label_lov_skip, T_NEAR);
        L(label_lov); // handle left or right overflow
        {
            Label label_lov_inner;
            mov(reg_aux_out_ptr, reg_out_ptr);
            mov(reg_cnt, reg_kht);
            L(label_lov_inner);
            {
                zero_it(reg_aux_out_ptr);
                add(reg_aux_out_ptr, out_h_step);
                dec(reg_cnt);
                jnz(label_lov_inner, T_NEAR);
            }
            add(reg_out_ptr, out_w_step);
            dec(reg_lov);
            jnz(label_lov, T_NEAR);
        }
        L(label_lov_skip);
    }

    // save output pointer for later use
    mov(reg_save_out_ptr, reg_out_ptr);

    // just in case there is no meat...
    Label label_kwp_end;
    test(reg_kwp, reg_kwp);
    jz(label_kwp_end, T_NEAR);

    // Unroll over W-dimension in powers of 2
    Label label_tov;
    Label label_khp, label_no_khp;
    Label label_bov;
    test(reg_tov, reg_tov);
    jnz(label_tov, T_NEAR);
    test(reg_khp, reg_khp);
    jnz(label_khp, T_NEAR);
    test(reg_bov, reg_bov);
    jnz(label_bov, T_NEAR);
    jmp(label_kwp_end, T_NEAR); // safe exit in case of bad parameters

    L(label_tov); // handle top overflow
    {
        Label label_tov_inner;
        mov(reg_aux_out_ptr, reg_out_ptr);
        mov(reg_cnt, reg_kwp);
        L(label_tov_inner);
        {
            zero_it(reg_aux_out_ptr);
            add(reg_aux_out_ptr, out_w_step);
            dec(reg_cnt);
            jnz(label_tov_inner, T_NEAR);
        }
        add(reg_out_ptr, out_h_step);
        dec(reg_tov);
        jnz(label_tov, T_NEAR);
    }
    test(reg_khp, reg_khp);
    jz(label_no_khp, T_NEAR);
    L(label_khp); // handle kh padding (not fully unrolled)
    {
        Label label_khp_inner;
        mov(reg_aux_inp_ptr, reg_inp_ptr);
        mov(reg_aux_out_ptr, reg_out_ptr);
        mov(reg_cnt, reg_kwp);
        L(label_khp_inner);
        {
            for (int ic = 0; ic < jcp.ic_without_padding;
                    ic += jcp.ic_block_int) {
                const int offset = ic * jcp.typesize_in;
                const bool masked
                        = ic + jcp.ic_block_int > jcp.ic_without_padding;
                // zero masking is needed to avoid dependency on destination
                Zmm zmm_load = masked ? zmm_tmp | ktail_mask | T_z : zmm_tmp;
                Zmm zmm_store = masked ? zmm_tmp | ktail_mask : zmm_tmp;
                if (is_bf16) {
                    vmovdqu16(zmm_load, ptr[reg_aux_inp_ptr + offset]);
                    vmovdqu16(ptr[reg_aux_out_ptr + offset], zmm_store);
                } else {
                    vmovdqu8(zmm_load, ptr[reg_aux_inp_ptr + offset]);
                    vmovdqu8(ptr[reg_aux_out_ptr + offset], zmm_store);
                }
            }
            add(reg_aux_inp_ptr, inp_w_step);
            add(reg_aux_out_ptr, out_w_step);
            dec(reg_cnt);
            jnz(label_khp_inner, T_NEAR);
        }
        add(reg_inp_ptr, inp_h_step);
        add(reg_out_ptr, out_h_step);
        dec(reg_khp);
        jnz(label_khp, T_NEAR);
    }
    L(label_no_khp);
    test(reg_bov, reg_bov);
    jz(label_kwp_end, T_NEAR);
    L(label_bov); // handle bottom overflow
    {
        Label label_bov_inner;
        mov(reg_aux_out_ptr, reg_out_ptr);
        mov(reg_cnt, reg_kwp);
        L(label_bov_inner);
        {
            zero_it(reg_aux_out_ptr);
            add(reg_aux_out_ptr, out_w_step);
            dec(reg_cnt);
            jnz(label_bov_inner, T_NEAR);
        }
        add(reg_out_ptr, out_h_step);
        dec(reg_bov);
        jnz(label_bov, T_NEAR);
    }
    L(label_kwp_end);

    { // Handle Right Overflow
        Label label_rov, label_rov_skip;
        // retrieve output pointer
        mov(reg_out_ptr, reg_save_out_ptr);
        // calculate the shift
        imul(reg_tmp, reg_kwp, out_w_step);
        // shift past the body
        add(reg_out_ptr, reg_tmp);
        // skip if no right overflow
        test(reg_rov, reg_rov);
        jz(label_rov_skip, T_NEAR);

        L(label_rov); // handle left or right overflow
        {
            Label label_rov_inner;
            mov(reg_aux_out_ptr, reg_out_ptr);
            mov(reg_cnt, reg_kht);
            L(label_rov_inner);
            {
                zero_it(reg_aux_out_ptr);
                add(reg_aux_out_ptr, out_h_step);
                dec(reg_cnt);
                jnz(label_rov_inner, T_NEAR);
            }
            add(reg_out_ptr, out_w_step);
            dec(reg_rov);
            jnz(label_rov, T_NEAR);
        }
        L(label_rov_skip);
    }

    // For bf16, zero-pad an extra cacheline to avoid NaNs
    // For int8, it is sufficient to zero-pad the weights only
    if (is_bf16) {
        // shift forward to align h index to end of needed buffer
        imul(reg_tmp, reg_kht, out_h_step);
        add(reg_out_ptr, reg_tmp);
        // shift backward to align w index to end of needed buffer
        sub(reg_out_ptr, out_w_step);
        vmovdqu16(ptr[reg_out_ptr], zmm_zero);
    }
}

void jit_avx512_core_amx_copy_to_pbuffer_t::generate() {

    // Special copy kernel for reduced lowering
    if (jcp.is_relo) {
        assert(jcp.nb_ic_int == 1);
        preamble();
        copy_row_reduced_lowering();
        postamble();
        return;
    }

    preamble();

    const bool is_3d = jcp.ndims == 5;
    mov(reg_inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(reg_out_ptr, ptr[param1 + GET_OFF(dst)]);
    if (is_3d) mov(reg_kdp, ptr[param1 + GET_OFF(kd_padding)]);
    mov(reg_khp, ptr[param1 + GET_OFF(kh_padding)]);
    mov(reg_tover, ptr[param1 + GET_OFF(t_overflow)]);
    mov(reg_bover, ptr[param1 + GET_OFF(b_overflow)]);
    mov(reg_owb, ptr[param1 + GET_OFF(owb)]);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    if (jcp.is_nspc && jcp.ic_without_padding % jcp.ic_block_int) {
        int tail_size = jcp.ic_without_padding % jcp.ic_block_int;
        uint64_t mask = (UINT64_C(1) << tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(ktail_mask, reg_tmp);
    }

    for (int icb = 0; icb < jcp.nb_ic_int; icb++) {
        Xbyak::Label kd_label, no_kd_label;
        Xbyak::Label kh_label, no_kh_label, icb_label;
        Xbyak::Label kh_tover_label, kh_bover_label;
        Xbyak::Label no_kh_tover_label, no_kh_bover_label;

        mov(reg_aux_inp_ptr, reg_inp_ptr);
        mov(reg_aux_out_ptr, reg_out_ptr);
        if (is_3d) {
            cmp(reg_kdp, 0);
            jle(no_kd_label, T_NEAR);
            mov(reg_kdc, reg_kdp);
            L(kd_label);
            push(reg_aux_inp_ptr);
            push(reg_aux_out_ptr);
        }
        cmp(reg_khp, 0);
        jle(no_kh_bover_label, T_NEAR); // nothing to do
        mov(reg_khc, reg_khp);

        cmp(reg_tover, 0);
        jle(no_kh_tover_label, T_NEAR);

        mov(reg_kh_over, reg_tover);
        L(kh_tover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small ic
            for (int iw = 0; iw < jcp.iwp; iw++)
                vmovups(ptr[reg_aux_out_ptr
                                + jcp.typesize_in * iw * jcp.ic_block_int_np],
                        zmm_zero);
            int out_h_offset = jcp.typesize_in * jcp.iwp * jcp.ic_block_int_np;
            add(reg_aux_out_ptr, out_h_offset);

            dec(reg_kh_over);
            jnz(kh_tover_label, T_NEAR);
        }
        sub(reg_khc, reg_tover);
        L(no_kh_tover_label);

        cmp(reg_khc, reg_bover);
        jle(no_kh_label, T_NEAR);

        L(kh_label);
        {
            copy_row(icb);
            size_t inp_h_offset = !jcp.is_nspc
                    ? (size_t)jcp.typesize_in * jcp.iw * jcp.ic_block
                    : (size_t)jcp.typesize_in * jcp.iw * jcp.ngroups
                            * jcp.ic_without_padding;
            size_t out_h_offset
                    = (size_t)jcp.typesize_in * jcp.iwp * jcp.ic_block_int_np;

            add(reg_aux_inp_ptr, inp_h_offset);
            add(reg_aux_out_ptr, out_h_offset);

            dec(reg_khc);
            cmp(reg_khc, reg_bover);
            jg(kh_label, T_NEAR);
        }
        L(no_kh_label);

        cmp(reg_khc, 0);
        jle(no_kh_bover_label, T_NEAR);

        L(kh_bover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small ic
            for (int iw = 0; iw < jcp.iwp; iw++)
                vmovups(ptr[reg_aux_out_ptr
                                + jcp.typesize_in * iw * jcp.ic_block_int_np],
                        zmm_zero);
            int out_h_offset = jcp.typesize_in * jcp.iwp * jcp.ic_block_int_np;
            add(reg_aux_out_ptr, out_h_offset);

            dec(reg_khc);
            jnz(kh_bover_label, T_NEAR);
        }
        size_t out_d_offset = (size_t)jcp.typesize_in
                * (jcp.ihp * jcp.iwp * jcp.ic_block_int_np + jcp.ic_block_int);
        L(no_kh_bover_label);
        if (is_3d) {
            size_t inp_d_offset = !jcp.is_nspc
                    ? (size_t)jcp.typesize_in * jcp.ih * jcp.iw * jcp.ic_block
                            * (jcp.dilate_d + 1)
                    : (size_t)jcp.typesize_in * jcp.ih * jcp.iw * jcp.ngroups
                            * jcp.ic_without_padding * (jcp.dilate_d + 1);
            pop(reg_aux_out_ptr);
            pop(reg_aux_inp_ptr);
            add(reg_aux_inp_ptr, inp_d_offset);
            add(reg_aux_out_ptr, out_d_offset);
            dec(reg_kdc);
            jnz(kd_label, T_NEAR);
            L(no_kd_label);
        }
        // End IC Loop
        size_t inp_cb_offset = !jcp.is_nspc
                ? (size_t)jcp.typesize_in * (jcp.ic_block_int_np / jcp.ic_block)
                        * jcp.id * jcp.ih * jcp.iw * jcp.ic_block
                : (size_t)jcp.typesize_in * jcp.ic_block_int_np;
        size_t out_cb_offset = (size_t)jcp.kd * out_d_offset;

        add(reg_inp_ptr, inp_cb_offset);
        add(reg_out_ptr, out_cb_offset);
    }

    postamble();
}

// Tile register decomposition
// { C_BASE = 0, I_BASE = 4, W_BASE = 6, }
int jit_avx512_core_amx_fwd_kernel_t::get_out_tensor(
        int h, int i, bool is_h_tail) const {
    const int C_BASE = 0;
    const int C_LAST = 4;
    assert(0 <= C_BASE && C_BASE < C_LAST && C_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(C_LAST);
    const int tile = C_BASE
            + (jcp.nb_oh_blocking > 1
                            ? h * jcp.nb_oh_blocking + i
                            : (int)is_h_tail * jcp.nb_oc_blocking + i);
    assert(C_BASE <= tile && tile < C_LAST);
    return tile;
}
int jit_avx512_core_amx_fwd_kernel_t::get_inp_tensor(
        int h, bool is_h_tail) const {
    const int I_BASE = 4;
    const int I_LAST = 6;
    assert(0 <= I_BASE && I_BASE < I_LAST && I_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(I_LAST);
    const int tile = I_BASE + (jcp.nb_oh_blocking > 1 ? h : (int)is_h_tail);
    assert(I_BASE <= tile && tile < I_LAST);
    return tile;
}
int jit_avx512_core_amx_fwd_kernel_t::get_wei_tensor(int i) const {
    const int W_BASE = 6;
    const int W_LAST = 8;
    assert(0 <= W_BASE && W_BASE < W_LAST && W_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(W_LAST);
    const int tile = W_BASE + i;
    assert(W_BASE <= tile && tile < W_LAST);
    return tile;
}

// Shifts and offsets
size_t jit_avx512_core_amx_fwd_kernel_t::get_inp_icb_step() const {
    return (size_t)jcp.kd * get_inp_d_step();
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wei_icb_step() const {
    return (size_t)jcp.typesize_in * jcp.kd * jcp.kh * jcp.kw
            * jcp.ic_block_int_np * jcp.oc_block;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_inp_d_step() const {
    return (size_t)jcp.typesize_in
            * (jcp.ihp * jcp.iwp * jcp.ic_block_int_np + jcp.ic_block_int);
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_inp_h_step() const {
    return (size_t)jcp.typesize_in * jcp.iwp * jcp.ic_block_int_np
            * (jcp.dilate_h + 1);
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wei_d_step() const {
    return (size_t)jcp.typesize_in * jcp.kh * jcp.kw * jcp.ic_block_int_np
            * jcp.oc_block;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wei_h_step() const {
    return (size_t)jcp.typesize_in * jcp.kw * jcp.ic_block_int_np
            * jcp.oc_block;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_out_ocb_offset(
        int ohb, int ocb) const {
    size_t el_offset = jcp.is_nspc
            ? (size_t)ocb * jcp.oc_block
                    + (size_t)ohb * jcp.ow * jcp.ngroups
                            * jcp.oc_without_padding
            : (size_t)ocb * jcp.oh * jcp.ow * jcp.oc_block
                    + (size_t)ohb * jcp.ow * jcp.oc_block;
    return (size_t)jcp.typesize_out * el_offset;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_out_row_offset(
        int ohb, int ocb, int j) const {
    size_t offset_w = jcp.is_nspc ? (size_t)jcp.typesize_out * j * jcp.ngroups
                    * jcp.oc_without_padding
                                  : (size_t)jcp.typesize_out * j * jcp.oc_block;
    return get_out_ocb_offset(ohb, ocb) + offset_w;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_out_shift(int width) const {
    return jcp.is_nspc ? (size_t)jcp.typesize_out * width * jcp.ngroups
                    * jcp.oc_without_padding
                       : (size_t)jcp.typesize_out * width * jcp.oc_block;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wsp_ocb_offset(
        int ohb, int ocb) const {
    size_t el_offset = (size_t)ocb * prv_width_ * jcp.oc_block
            + (size_t)ohb * jcp.nb_oc_blocking * jcp.full_tile_width
                    * jcp.oc_block;
    return jcp.typesize_acc * el_offset;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wsp_row_offset(
        int ohb, int ocb, int j) const {
    return get_wsp_ocb_offset(ohb, ocb)
            + (size_t)jcp.typesize_acc * j * jcp.oc_block;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wsp_shift() const {
    return (size_t)jcp.typesize_acc * jcp.nb_oh_blocking * jcp.full_tile_width
            * jcp.oc_block * jcp.nb_oc_blocking;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_wei_offset(int ocb, int kw) const {
    size_t el_offset = (size_t)kw * jcp.ic_block_int_np * jcp.oc_block;
    size_t raw_oc_subblock_step
            = jcp.kd * jcp.kh * jcp.kw * jcp.ic_block_int_np * jcp.oc_block;
    size_t oc_subblock_step = jcp.is_relo
            ? rnd_up(raw_oc_subblock_step, jcp.ic_block_int * jcp.oc_block)
            : raw_oc_subblock_step;
    el_offset += (size_t)ocb * jcp.nb_ic_int * oc_subblock_step;
    return jcp.typesize_in * el_offset;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_inp_shift() const {
    size_t w_step = (jcp.is_relo ? jcp.stride_w * jcp.kh
                                 : jcp.is_pbuffer_strided ? 1 : jcp.stride_w)
            * jcp.ic_block_int_np;
    return (size_t)jcp.typesize_in * jcp.tile_width * w_step;
}
size_t jit_avx512_core_amx_fwd_kernel_t::get_inp_offset(int ohb, int kw) const {
    if (jcp.is_relo)
        return ohb * jcp.iwp * jcp.kh * jcp.ic_block_int_np * jcp.typesize_in;
    // calculate offset by height dimension
    const int gen_kh = (jcp.kh - 1) * (jcp.dilate_h + 1) + 1;
    const int gen_stride_h = nstl::min(jcp.stride_h, gen_kh);
    size_t el_offset = (size_t)ohb * jcp.oh_per_tile * gen_stride_h * jcp.iwp
            * jcp.ic_block_int_np;

    // add offset by width dimension
    if (IMPLICATION(jcp.is_pbuffer_strided, jcp.stride_w == 1)) {
        el_offset += (size_t)kw * (jcp.dilate_w + 1) * jcp.ic_block_int_np;
    } else if (jcp.dilate_w > 0) {
        el_offset += (size_t)kw * jcp.ow_block * jcp.ic_block_int_np;
    } else {
        // dilate_w == 0 && stride_w > 1
        // there are min(jcp.kw, jcp.stride_w) continuous sets of input data
        // (foreach stride idx), they are placed one by one without additional
        // padding

        // calculate set idx for current kw value
        int set_idx = kw % jcp.stride_w;
        // calculate shift within set for current kw value
        int set_shift = kw / jcp.stride_w;

        // calculate the beginning of the current set along width, each set
        // with index set_i contains number of elements along width equal to
        // jcp.ow - 1 + jcp.kw / jcp.stride_w
        //     + (set_i < jcp.kw % jcp.stride_w)
        size_t set_start = (jcp.ow_block - 1 + jcp.kw / jcp.stride_w) * set_idx
                + nstl::min(set_idx, jcp.kw % jcp.stride_w);
        el_offset += (set_start + set_shift) * jcp.ic_block_int_np;
    }
    return jcp.typesize_in * el_offset;
}

// Code generation
void jit_avx512_core_amx_fwd_kernel_t::prepare_output(int tail) {
    for (int h = 0; h < jcp.nb_oh_blocking; h++)
        for (int i = 0; i < jcp.nb_oc_blocking; i++)
            tilezero(Tmm(get_out_tensor(h, i, tail)));
}

void jit_avx512_core_amx_fwd_kernel_t::init_runtime_counters(
        bool start_with_last_tile_block) {
    prv_width_ = start_with_last_tile_block && jcp.tile_tail > 0
            ? jcp.tile_tail
            : jcp.tile_width;

    row_count_ = 0;
    is_store_done_ = false;
    is_buffer_empty_ = true;
}

bool jit_avx512_core_amx_fwd_kernel_t::maybe_eltwise(int position) {
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

Ymm jit_avx512_core_amx_fwd_kernel_t::ymm_mask(
        const Ymm ymm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                     : ymm_in;
}

Zmm jit_avx512_core_amx_fwd_kernel_t::zmm_mask(
        const Zmm zmm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}

void jit_avx512_core_amx_fwd_kernel_t::cvt2ps(data_type_t type_in,
        const Zmm zmm_in, const Operand &op, bool mask_flag = false) {
    const Zmm zmm = zmm_mask(zmm_in, mask_flag);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: vmovups(zmm, op); break;
        case data_type::s8: vpmovsxbd(zmm, op); break;
        case data_type::u8: vpmovzxbd(zmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32) vcvtdq2ps(zmm_in, zmm_in);
}

void jit_avx512_core_amx_fwd_kernel_t::store_output_vector_bf16(
        Zmm zmm_out, int ocb, int h, int w) {
    const bool mask_flag = jcp.is_nspc && jcp.oc_without_padding != jcp.oc
            && ocb == (jcp.nb_oc_blocking - 1);

    auto addr = EVEX_compress_addr(reg_out_ptr, get_out_row_offset(h, ocb, w));

    const auto &p = attr_.post_ops_;

    const int sum_idx = p.find(primitive_kind::sum);
    if (sum_idx != -1) {
        if (jcp.dst_dt == data_type::bf16) {
            vpmovzxwd(zmm_mask(zmm_prev_dst, mask_flag), addr);
            vpslld(zmm_prev_dst, zmm_prev_dst, 16);
            vaddps(zmm_out, zmm_prev_dst);
        } else {
            vmovups(zmm_mask(zmm_prev_dst, mask_flag), addr);
            vaddps(zmm_out, zmm_prev_dst);
        }
    }
    if (jcp.with_bias) {
        int bias_offset = jcp.typesize_bia * ocb * jcp.oc_block;
        auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        if (jcp.bia_dt == data_type::bf16) {
            vpmovzxwd(zmm_mask(zmm_bias, mask_flag), bias_addr);
            vpslld(zmm_bias, zmm_bias, 16);
            vaddps(zmm_out, zmm_bias);
        } else
            vaddps(zmm_mask(zmm_out, mask_flag), bias_addr);
    }

    const int eltwise_ind = p.find(primitive_kind::eltwise);
    if (eltwise_ind != -1) eltwise_injector_->compute_vector(zmm_out.getIdx());

    if (jcp.dst_dt == data_type::bf16) {
        Ymm ymm_out = Ymm(zmm_out.getIdx());
        vcvtneps2bf16(ymm_out, zmm_out);
        vmovdqu16(addr, ymm_mask(ymm_out, mask_flag, true));
    } else {
        vmovups(addr, zmm_mask(zmm_out, mask_flag, true));
    }
}

void jit_avx512_core_amx_fwd_kernel_t::store_output_vector_int8(
        Zmm zmm_out, int ocb, int h, int w) {
    const int nb_oc_block = jcp.nb_oc_blocking;
    const int oc_block = jcp.oc_block;
    const bool mask_flag = true && jcp.oc_without_padding != jcp.oc
            && ocb == (nb_oc_block - 1);

    auto addr = EVEX_compress_addr(reg_out_ptr, get_out_row_offset(h, ocb, w));

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = nullptr;
    if (sum_idx != -1) {
        const auto &p_entry = p.entry_[sum_idx];
        p_sum_scale = &p_entry.sum.scale;
    }

    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

    int scale_offset = jcp.is_oc_scale * (sizeof(float) * ocb * oc_block);
    if (jcp.with_bias) {
        int bias_offset = jcp.typesize_bia * ocb * oc_block;
        auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        cvt2ps(jcp.bia_dt, zmm_bias, bias_addr, mask_flag);
    }
    if (jcp.src_zero_point) {
        // zero_point: conv(src_x8, wei_s8) - src_shift_s32 * compensation_s32
        int zp_offset = sizeof(int32_t) * ocb * oc_block;
        const Zmm m_zmm_zp = zmm_mask(zmm_zp, mask_flag);
        vpmulld(m_zmm_zp, zmm_src_zp,
                EVEX_compress_addr(reg_zp_compensation, zp_offset));
        vpaddd(zmm_out, zmm_out, zmm_zp);
    }

    /* add bias and zero-point to zmm_accum */
    vcvtdq2ps(zmm_out, zmm_out);
    if (jcp.with_bias) vaddps(zmm_out, zmm_out, zmm_bias);
    const Zmm zmm_out_msk = zmm_mask(zmm_out, mask_flag);
    vmulps(zmm_out_msk, zmm_out,
            EVEX_compress_addr(reg_ptr_scales, scale_offset));

    /* Do post-ops */
    if (maybe_eltwise(0)) eltwise_injector_->compute_vector(zmm_out.getIdx());
    if (p_sum_scale) { // post_op: sum
        cvt2ps(jcp.dst_dt, zmm_prev_dst, addr, mask_flag);
        if (*p_sum_scale == 1.f)
            vaddps(zmm_out, zmm_prev_dst);
        else
            vfmadd231ps(zmm_out, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
    }
    if (maybe_eltwise(1)) eltwise_injector_->compute_vector(zmm_out.getIdx());

    if (jcp.dst_zero_point) { vaddps(zmm_out, zmm_out, zmm_dst_zp); }

    // Properly saturate the accumulators for integer datatypes
    if (one_of(jcp.dst_dt, u8, s8, s32)) {
        init_saturate_f32(
                zmm_zero, zmm_saturation, reg_aux_saturation, f32, jcp.dst_dt);
        saturate_f32(zmm_out, zmm_zero, zmm_saturation, jcp.dst_dt);
        vcvtps2dq(zmm_out, zmm_out);
    }

    const Zmm zmm_out_store = zmm_mask(zmm_out, mask_flag, true);

    switch (jcp.dst_dt) {
        case data_type::f32:
        case data_type::s32: vmovups(addr, zmm_out_store); break;
        case data_type::s8: vpmovsdb(addr, zmm_out_store); break;
        case data_type::u8: vpmovusdb(addr, zmm_out_store); break;
        default: assert(!"unknown dst_dt");
    }
}

void jit_avx512_core_amx_fwd_kernel_t::store_output_vector(
        Zmm zmm_out, int ocb, int h, int w) {
    /*
    Output:
              jcp.is_nspc              !jcp.is_nspc
              ---------------------    ---------------------
        INT8: [N][H][W][NBOC][16OC]
        BF16: [N][H][W][NBOC][16OC] or [N][NBOC][H][W][16OC]
    */
    if (jcp.src_dt == data_type::bf16) {
        store_output_vector_bf16(zmm_out, ocb, h, w);
    } else {
        store_output_vector_int8(zmm_out, ocb, h, w);
    }
}

void jit_avx512_core_amx_fwd_kernel_t::store_output(
        int width, int tail, bool do_store) {
    auto store_output_block = [=](int width, int tail, bool do_store,
                                      bool is_last_h = false) {
        // Calculate the number of oh blocks; it may differ on last call
        const int last_h_blks
                = div_up(jcp.oh, jcp.oh_per_tile) % jcp.nb_oh_blocking;
        const int h_blks = is_last_h && last_h_blks != 0 ? last_h_blks
                                                         : jcp.nb_oh_blocking;
        // Calculate the number of oh rows per tile; it may differ on last call
        const int h_tail = is_last_h && jcp.oh % jcp.oh_per_tile != 0
                ? (h_blks - 1) * jcp.oh_per_tile + jcp.oh % jcp.oh_per_tile
                : h_blks * jcp.oh_per_tile;

        if (jcp.src_zero_point) {
            mov(reg_zp_compensation, ptr[param1 + GET_OFF(zp_compensation)]);
            mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
            vpbroadcastd(zmm_src_zp, EVEX_compress_addr(reg_src_zero_point, 0));
        }
        if (jcp.dst_zero_point) {
            mov(reg_dst_zero_point, ptr[param1 + GET_OFF(dst_zero_point)]);
            vcvtdq2ps(zmm_dst_zp,
                    EVEX_compress_addr(reg_dst_zero_point, 0, true));
        }

        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
            for (int ohb = 0; ohb < h_blks; ohb++) {
                /* Formats: Workspace: [NBOC][W][16OC] */
                tilestored(ptr[reg_wsp_ptr + reg_wei_stride
                                   + get_wsp_ocb_offset(ohb, ocb)],
                        Tmm(get_out_tensor(ohb, ocb, tail)));
                is_buffer_empty_ = false;
                is_store_done_ = false;
                for (int tw = 0; tw < width && do_store; tw++) {
                    const int gen_kw = (jcp.kw - 1) * (jcp.dilate_w + 1) + 1;
                    const int owp = gen_kw + jcp.ow - 1;
                    const int oh_index = ohb * jcp.oh_per_tile + tw / owp;
                    const int ow_index = tw % owp;
                    assert(IMPLICATION(jcp.oh_per_tile == 1,
                            ohb == oh_index && tw == ow_index));
                    if (oh_index < h_tail && ow_index < jcp.ow) {
                        Zmm zmm_r = zmm_out(tw);
                        vmovups(zmm_r,
                                ptr[reg_wsp_ptr
                                        + get_wsp_row_offset(ohb, ocb, tw)]);
                        store_output_vector(zmm_r, ocb, oh_index, ow_index);
                    }
                }
            }
        }
    };

    // adjustment in case interleave store is turned off
    do_store = do_store || jcp.per_one_pstore == 0;
    if (jcp.oh % (jcp.oh_per_tile * jcp.nb_oh_blocking) == 0) {
        store_output_block(width, tail, do_store);
    } else {
        Label label_oh_oc_store, label_done;
        mov(reg_last_h, ptr[param1 + GET_OFF(last_h)]);
        cmp(reg_last_h, 0);
        jne(label_oh_oc_store, T_NEAR);
        store_output_block(width, tail, do_store, true); // last h
        jmp(label_done, T_NEAR);
        L(label_oh_oc_store);
        store_output_block(width, tail, do_store, false);
        L(label_done);
    }
    if (do_store) add(reg_out_ptr, get_out_shift(width));
}

void jit_avx512_core_amx_fwd_kernel_t::interleave_store(int width) {
    for (int c = 0;
            c < jcp.per_one_pstore && !is_store_done_ && !is_buffer_empty_;
            c++) {
        // row_count = ohb * OCB * TW + ocb * TW + tw
        int tw = row_count_ % prv_width_;
        int ocb = (row_count_ / prv_width_) % jcp.nb_oc_blocking;
        int ohb = (row_count_ / prv_width_) / jcp.nb_oc_blocking;

        Zmm zmm_r = zmm_out(tw);
        vmovups(zmm_r, ptr[reg_wsp_ptr + get_wsp_row_offset(ohb, ocb, tw)]);
        store_output_vector(zmm_r, ocb, ohb, tw);
        row_count_++;

        if (row_count_
                == prv_width_ * jcp.nb_oc_blocking * jcp.nb_oh_blocking) {
            add(reg_out_ptr, get_out_shift(prv_width_));
            row_count_ = 0;
            is_store_done_ = true;
            prv_width_ = width;
        }
    }
}

void jit_avx512_core_amx_fwd_kernel_t::compute_icb_loop(
        int width, bool do_store) {
    const bool tail = width == jcp.tile_tail;

    auto tdpbxxd = [=](const Tmm &x1, const Tmm &x2, const Tmm &x3) {
        if (jcp.src_dt == data_type::bf16 && jcp.wei_dt == data_type::bf16) {
            tdpbf16ps(x1, x2, x3);
        } else if (jcp.src_dt == data_type::u8 && jcp.wei_dt == data_type::u8) {
            tdpbuud(x1, x2, x3);
        } else if (jcp.src_dt == data_type::u8 && jcp.wei_dt == data_type::s8) {
            tdpbusd(x1, x2, x3);
        } else if (jcp.src_dt == data_type::s8 && jcp.wei_dt == data_type::u8) {
            tdpbsud(x1, x2, x3);
        } else if (jcp.src_dt == data_type::s8 && jcp.wei_dt == data_type::s8) {
            tdpbssd(x1, x2, x3);
        } else {
            assert(!"unsupported combination");
        }
    };

    prepare_output(tail);

    // prepare registers for when 'interleave_store()' is computed
    if (jcp.src_zero_point) {
        mov(reg_zp_compensation, ptr[param1 + GET_OFF(zp_compensation)]);
        mov(reg_src_zero_point, ptr[param1 + GET_OFF(src_zero_point)]);
        vpbroadcastd(zmm_src_zp, EVEX_compress_addr(reg_src_zero_point, 0));
    }
    if (jcp.dst_zero_point) {
        mov(reg_dst_zero_point, ptr[param1 + GET_OFF(dst_zero_point)]);
        vcvtdq2ps(zmm_dst_zp, EVEX_compress_addr(reg_dst_zero_point, 0, true));
    }

    // reduced lowering path
    if (jcp.is_relo) {
        const int nreduce = jcp.nreduce;
        const int stride = jcp.ic_block_int; // ie 64 (32) for int8 (bf16)

        push(reg_inp_ptr);
        push(reg_wei_ptr);

        for (int ireduce = 0; ireduce < nreduce; ireduce += stride) {
            for (int ohb = 0; ohb < jcp.nb_oh_blocking; ohb++) {
                tileloadd(Tmm(get_inp_tensor(ohb, tail)),
                        ptr[reg_inp_ptr + get_inp_offset(ohb, 0)
                                + reg_inp_stride]);
            }
            for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
                tileloadd(Tmm(get_wei_tensor(ocb)),
                        ptr[reg_wei_ptr + get_wei_offset(ocb, 0)
                                + reg_wei_stride]);
                for (int ohb = 0; ohb < jcp.nb_oh_blocking; ohb++) {
                    tdpbxxd(Tmm(get_out_tensor(ohb, ocb, tail)),
                            Tmm(get_inp_tensor(ohb, tail)),
                            Tmm(get_wei_tensor(ocb)));
                    interleave_store(width);
                }
            }
            if (ireduce + stride < nreduce) {
                add(reg_inp_ptr, stride * jcp.typesize_in);
                add(reg_wei_ptr, stride * jcp.oc_block * jcp.typesize_in);
            }
        }
        pop(reg_wei_ptr);
        pop(reg_inp_ptr);

        store_output(width, tail, do_store);

        add(reg_inp_ptr, get_inp_shift());
        return;
    }

    auto wei_offset = [&](int icb, int ocb, int kd, int kh, int kw) {
        return (size_t)icb * get_wei_icb_step() + kd * get_wei_d_step()
                + kh * get_wei_h_step() + get_wei_offset(ocb, kw);
    };

    auto inp_offset = [&](int icb, int ohb, int kd, int kh, int kw) {
        return (size_t)icb * get_inp_icb_step() + kd * get_inp_d_step()
                + kh * get_inp_h_step() + get_inp_offset(ohb, kw);
    };

    auto safe_tileloadd
            = [=](const Tmm &t1, const Xbyak::Reg64 &reg_ptr, size_t offset,
                      const Xbyak::Reg64 &reg_stride) {
                  if (offset <= INT32_MAX) {
                      tileloadd(t1, ptr[reg_ptr + offset + reg_stride]);
                  } else {
                      safe_add(reg_ptr, offset, reg_tmp);
                      tileloadd(t1, ptr[reg_ptr + reg_stride]);
                      safe_sub(reg_ptr, offset, reg_tmp);
                  }
              };

    // normal and k-remainders path
    const bool check_kd_padding
            = jcp.ndims == 5 && (jcp.f_pad > 0 || jcp.back_pad > 0);
    for (int icb = 0; icb < jcp.nb_ic_int; icb++) {
        Label kd_skip_compute;
        if (check_kd_padding) mov(reg_kd, ptr[param1 + GET_OFF(kd_padding)]);

        for (int kd = 0; kd < jcp.kd; kd++) {
            if (check_kd_padding) {
                dec(reg_kd);
                jl(kd_skip_compute, T_NEAR);
            }
            for (int kh = 0; kh < jcp.kh; kh++) {
                for (int set_idx = 0; set_idx < jcp.n_stride_sets;
                        set_idx++) { // used to optimize input memory reuse in L1$
                    for (int kw = set_idx; kw < jcp.kw; kw += jcp.kw_step) {
                        for (int ohb = 0; ohb < jcp.nb_oh_blocking; ohb++) {
                            const size_t inp_off
                                    = inp_offset(icb, ohb, kd, kh, kw);
                            safe_tileloadd(Tmm(get_inp_tensor(ohb, tail)),
                                    reg_inp_ptr, inp_off, reg_inp_stride);
                        }
                        for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
                            const size_t wei_off
                                    = wei_offset(icb, ocb, kd, kh, kw);
                            safe_tileloadd(Tmm(get_wei_tensor(ocb)),
                                    reg_wei_ptr, wei_off, reg_wei_stride);
                            for (int ohb = 0; ohb < jcp.nb_oh_blocking; ohb++) {
                                tdpbxxd(Tmm(get_out_tensor(ohb, ocb, tail)),
                                        Tmm(get_inp_tensor(ohb, tail)),
                                        Tmm(get_wei_tensor(ocb)));
                                interleave_store(width);
                            }
                        }
                    }
                }
            }
        }
        L(kd_skip_compute);
    }
    store_output(width, tail, do_store);

    add(reg_inp_ptr, get_inp_shift());
}

void jit_avx512_core_amx_fwd_kernel_t::compute_ow_loop() {
    auto compute_ow_loop_body = [=](bool last_owb, int num_tile_blocks) {
        int gen_tile_tail = last_owb && jcp.tile_tail > 0 ? jcp.tile_tail
                                                          : jcp.tile_width;
        init_runtime_counters(last_owb && num_tile_blocks == 1);
        for (int owb = 0; owb < num_tile_blocks - 1; owb++)
            compute_icb_loop(jcp.tile_width, false);
        compute_icb_loop(gen_tile_tail, true);
    };

    if (jcp.nb_ow == 1) {
        compute_ow_loop_body(true, jcp.ow_blocks);
    } else {
        assert(jcp.oh_per_tile == 1);
        Label label_done;
        int ow_blocks_per_call = utils::div_up(jcp.ow_block, jcp.tile_width);
        int last_owb_tile_blocks = jcp.ow_blocks % ow_blocks_per_call;
        if (last_owb_tile_blocks == 0 && jcp.tile_tail > 0)
            last_owb_tile_blocks = ow_blocks_per_call;
        if (last_owb_tile_blocks > 0) {
            Label label_not_last_owb;
            mov(reg_tmp, ptr[param1 + GET_OFF(owb)]);
            cmp(reg_tmp, jcp.nb_ow - 1);
            jne(label_not_last_owb, T_NEAR);

            compute_ow_loop_body(true, last_owb_tile_blocks);

            jmp(label_done, T_NEAR);

            L(label_not_last_owb);
        }
        compute_ow_loop_body(false, ow_blocks_per_call);

        L(label_done);
    }
}

void jit_avx512_core_amx_fwd_kernel_t::generate() {
    preamble();

    mov(reg_inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(reg_wei_ptr, ptr[param1 + GET_OFF(filt)]);
    mov(reg_out_ptr, ptr[param1 + GET_OFF(dst)]);
    mov(reg_wsp_ptr, ptr[param1 + GET_OFF(acc_s32)]);

    mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);

    const int fac = jcp.is_relo ? jcp.stride_w * jcp.kh
                                : jcp.is_pbuffer_strided ? 1 : jcp.stride_w;
    const int inp_stride = fac * jcp.ic_block_int_np * jcp.typesize_in;
    const int wei_stride = jcp.oc_block * jcp.typesize_acc;
    mov(reg_inp_stride, inp_stride);
    mov(reg_wei_stride, wei_stride);

    if (jcp.is_nspc && jcp.oc_without_padding != jcp.oc) {
        // Use mask 0xF by default for all output data and post-ops
        // loads / stores with block index
        // ocb = occ * jcp.nb_oc_blocking + (jcp.nb_oc_blocking - 1)
        // TODO: use masked loads / stores for the last occ only
        int current_block_size = jcp.oc_block;
        int mask = (1 << current_block_size) - 1;
        Xbyak::Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
        Xbyak::Label mask_is_set;
        mov(reg_oc_blocks, ptr[param1 + GET_OFF(oc_blocks)]);
        cmp(reg_oc_blocks, jcp.nb_oc - jcp.nb_oc_blocking);
        jne(mask_is_set, T_NEAR);
        // Reset the mask
        current_block_size = jcp.oc_without_padding % jcp.oc_block;
        mask = (1 << current_block_size) - 1;
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);

        L(mask_is_set);
    }
    compute_ow_loop();

    postamble();

    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
}

bool jit_avx512_core_amx_fwd_kernel_t::post_ops_ok(
        const jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;
    const bool is_bf16 = jcp.src_dt == data_type::bf16;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    auto is_sum = [&](int idx) {
        if (is_bf16)
            return p.entry_[idx].is_sum();
        else
            return p.contain(sum, idx);
    };

    switch (p.len()) {
        case 0: return true;
        case 1: return is_eltwise(0) || is_sum(0);
        case 2:
            return (is_sum(0) && is_eltwise(1))
                    || (!is_bf16 && is_sum(1) && is_eltwise(0));
        default: return false;
    }

    return false;
}

void jit_avx512_core_amx_fwd_kernel_t::tile_configure(char *tcfg_buff) {
    const int vnni_width = jcp.src_dt == data_type::bf16 ? 2 : 4;
    // Input tile dimensions
    const int a_col = jcp.is_relo ? jcp.ic_block_int
                                  : jcp.ic_block_int_np * jcp.kw_per_tile;
    // Weights tile dimensions
    const int b_col = jcp.oc_block * vnni_width;
    const int b_row = a_col / vnni_width;
    // Accumulator tile dimensions
    const int c_col = 16;

    for (size_t i = 0; i < 64; i++)
        tcfg_buff[i] = 0;

    // Weights (W_BASE) Tensor Tiles
    for (int i = 0; i < jcp.nb_oc_blocking; i++)
        tc_configure_tile((palette_config_t *)tcfg_buff, get_wei_tensor(i),
                b_row, b_col * jcp.typesize_in);

    // Input (I_BASE) and Accumulator (C_BASE) Tensor Tiles
    for (int h = 0; h < jcp.nb_oh_blocking; h++) {
        tc_configure_tile((palette_config_t *)tcfg_buff, get_inp_tensor(h),
                jcp.tile_width, a_col * jcp.typesize_in);
        for (int i = 0; i < jcp.nb_oc_blocking; i++)
            tc_configure_tile((palette_config_t *)tcfg_buff,
                    get_out_tensor(h, i), jcp.tile_width,
                    c_col * jcp.typesize_acc);
    }
    if (jcp.tile_tail != 0) {
        assert(jcp.nb_oh_blocking == 1);
        assert(jcp.oh_per_tile == 1);
        assert(jcp.ow > jcp.tile_width);
        tc_configure_tile((palette_config_t *)tcfg_buff,
                get_inp_tensor(0, true), jcp.tile_tail,
                a_col * jcp.typesize_in);
        for (int i = 0; i < jcp.nb_oc_blocking; i++)
            tc_configure_tile((palette_config_t *)tcfg_buff,
                    get_out_tensor(0, i, true), jcp.tile_tail,
                    c_col * jcp.typesize_acc);
    }

    ((palette_config_t *)tcfg_buff)->palette_id = amx::get_max_palette();
}

status_t jit_avx512_core_amx_fwd_kernel_t::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, const primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();
    bool is_1d = ndims == 3;
    bool is_3d = ndims == 5;

    const bool is_bf16_convolution
            = everyone_is(true, src_d.data_type() == data_type::bf16,
                    weights_d.data_type() == data_type::bf16,
                    one_of(dst_d.data_type(), data_type::bf16, data_type::f32));
    const bool is_int8_convolution = everyone_is(true,
            (src_d.data_type() == data_type::u8
                    || src_d.data_type() == data_type::s8),
            weights_d.data_type() == data_type::s8,
            one_of(dst_d.data_type(), data_type::f32, data_type::s32,
                    data_type::s8, data_type::u8));

    bool supported = false
            || (is_bf16_convolution && mayiuse(avx512_core_bf16_amx_bf16))
            || (is_int8_convolution && mayiuse(avx512_core_bf16_amx_int8));
    if (!supported) return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.isa = is_bf16_convolution ? avx512_core_bf16_amx_bf16
                                  : avx512_core_bf16_amx_int8;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;

    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = !is_1d ? src_d.dims()[ndims - 2] : 1;
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = !is_1d ? dst_d.dims()[ndims - 2] : 1;
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = !is_1d ? weights_d.dims()[with_groups + ndims - 2] : 1;
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = !is_1d ? cd.padding[0][ndims - 4] : 0;
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = !is_1d ? cd.strides[ndims - 4] : 1;
    jcp.stride_w = cd.strides[ndims - 3];
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.dilate_d = is_3d ? cd.dilates[ndims - 5] : 0;
    jcp.dilate_h = !is_1d ? cd.dilates[ndims - 4] : 0;
    jcp.dilate_w = cd.dilates[ndims - 3];

    const int gen_kd = (jcp.kd - 1) * (jcp.dilate_d + 1) + 1;
    const int gen_kh = (jcp.kh - 1) * (jcp.dilate_h + 1) + 1;
    const int gen_kw = (jcp.kw - 1) * (jcp.dilate_w + 1) + 1;
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, gen_kd);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, gen_kh);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, gen_kw);
    if (jcp.l_pad >= gen_kw || jcp.r_pad >= gen_kw || jcp.t_pad >= gen_kh
            || jcp.b_pad >= gen_kh || jcp.f_pad >= gen_kd
            || jcp.back_pad >= gen_kd)
        return status::unimplemented;

    const int max_pad = 28; // akin to maximum jcp.ur_w value in other jits
    if (jcp.l_pad > max_pad || jcp.r_pad > max_pad)
        return status::unimplemented; // TODO: relax this restriction

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.src_dt = cd.src_desc.data_type;
    jcp.wei_dt = cd.weights_desc.data_type;

    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);

    if (jcp.is_depthwise)
        return status::unimplemented; // TODO: add support of DW convolution

    const auto zp = attr.zero_points_;
    jcp.dst_zero_point = !zp.has_default_values(DNNL_ARG_DST);
    jcp.src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
    jcp.zp_src_is_common = zp.common(
            DNNL_ARG_SRC); // otherwise, it's per-channel (not supported)
    if (!IMPLICATION(jcp.src_zero_point, jcp.zp_src_is_common)
            || !IMPLICATION(jcp.dst_zero_point || jcp.src_zero_point,
                    is_int8_convolution))
        return status::unimplemented;

    // XXX to be enabled in a separate commit
    if (jcp.src_zero_point
            && !utils::everyone_is(0, jcp.r_pad, jcp.l_pad, jcp.b_pad,
                    jcp.t_pad, jcp.f_pad, jcp.back_pad))
        return status::unimplemented;

    format_tag_t dat_tag_ncsp = utils::pick(ndims - 3, format_tag::nCw16c,
            format_tag::nChw16c, format_tag::nCdhw16c);
    format_tag_t dat_tag_nspc = utils::pick(
            ndims - 3, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    // To toggle the default data layout for BF16 between nChw16c and nhwc,
    // swap the following two variable definitions. Current choice: nhwc.

    // Clang-tidy change - if it was intentional please revert it and
    // put `NOLINTNEXTLINE` to suppress the warning.
    format_tag_t dat_tag_opt = dat_tag_nspc;
    format_tag_t dat_tag_alt
            = is_bf16_convolution ? dat_tag_ncsp : dat_tag_nspc;

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag_opt));
        jcp.src_tag = dat_tag_opt;
    } else
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag_alt, dat_tag_opt);

    if (!one_of(jcp.src_tag, dat_tag_alt, dat_tag_opt))
        return status::unimplemented;

    jcp.is_nspc = jcp.src_tag == dat_tag_nspc;
    assert(IMPLICATION(is_int8_convolution, jcp.is_nspc));

    if (dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(dst_md, jcp.src_tag));
        jcp.dst_tag = jcp.src_tag;
    } else
        jcp.dst_tag = dst_d.matches_one_of_tag(jcp.src_tag);

    if (jcp.dst_tag != jcp.src_tag) return status::unimplemented;

    if (jcp.with_bias && bias_d.format_kind() == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md, format_tag::x));

    jcp.nthr = nthreads;

    jcp.ic_block = 16;
    jcp.oc_block = 16;

    if (jcp.ngroups == 1) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    bool args_ok = jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0;
    if (!args_ok) return status::unimplemented;

    const int vnni_width = is_bf16_convolution ? 2 : 4;
    jcp.ic_block_int = jcp.ic_block * vnni_width; // 32 for bf16, 64 for int8

    // small-ic parameters
    jcp.ic_block_int_np = jcp.is_nspc
            ? nstl::min(jcp.ic_block_int, jcp.ic_without_padding)
            : jcp.ic_block_int;
    bool is_small_ic = jcp.ic_block_int_np < jcp.ic_block_int;

    // reduced lowering
    jcp.is_relo = (!is_3d)
            && is_small_ic
            // no trivial cases
            && 1 < jcp.kh * jcp.kw
            // required for use of VPERMB instruction in weights copy kernel
            && IMPLICATION(is_int8_convolution,
                    cpu().has(Xbyak::util::Cpu::tAVX512_VBMI))
            // no dilation or excessive stride along w-direction
            && everyone_is(0, jcp.dilate_h, jcp.dilate_w)
            // no dilation or excessive stride along h-direction
            && jcp.stride_h <= jcp.kh && jcp.stride_w <= jcp.kw;
    jcp.nreduce = jcp.kh * jcp.kw * jcp.ic_block_int_np;

    if (!jcp.is_relo) {
        jcp.ic_block_int_np = is_bf16_convolution
                ? jcp.ic_block_int
                : rnd_up(jcp.ic_block_int_np, vnni_width);
        is_small_ic = jcp.ic_block_int_np < jcp.ic_block_int;
    }

    // k-remainders
    jcp.kw_per_tile = is_small_ic && !jcp.is_relo && jcp.dilate_w == 0
                    && jcp.stride_w <= jcp.kw // TODO: relax this restriction
                    && jcp.kw * jcp.ic_block_int_np <= jcp.ic_block_int
            ? jcp.kw
            : 1;
    jcp.is_pbuffer_strided = (1 == jcp.kw_per_tile);
    jcp.n_stride_sets
            = jcp.is_pbuffer_strided ? nstl::min(jcp.stride_w, jcp.kw) : 1;
    jcp.kw_step = jcp.is_pbuffer_strided ? jcp.stride_w : jcp.kw_per_tile;

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        using namespace memory_extra_flags;
        format_tag_t wei_tag;
        wei_tag = jcp.is_relo ? pick(with_groups + 2 * (ndims - 3), Owi16o,
                          gOwi16o, Owhi16o, gOwhi16o) // no 3d support
                              : is_bf16_convolution
                        ? pick(with_groups + 2 * (ndims - 3), OIw16i16o2i,
                                gOIw16i16o2i, OIhw16i16o2i, gOIhw16i16o2i,
                                OIdhw16i16o2i, gOIdhw16i16o2i)
                        : is_small_ic ? pick(with_groups + 2 * (ndims - 3),
                                  OwI16o4i, gOwI16o4i, OhwI16o4i, gOhwI16o4i,
                                  OdhwI16o4i, gOdhwI16o4i)
                                      : pick(with_groups + 2 * (ndims - 3),
                                              OIw16i16o4i, gOIw16i16o4i,
                                              OIhw16i16o4i, gOIhw16i16o4i,
                                              OIdhw16i16o4i, gOIdhw16i16o4i);

        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);

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

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;
    jcp.typesize_acc = sizeof(int32_t);

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic_int = div_up(jcp.ic, jcp.ic_block_int);

    jcp.nb_oc_blocking_thr_chunk = 1;

    const int max_palette = amx::get_max_palette();
    jcp.max_tiles = amx::get_max_tiles(max_palette);
    jcp.full_tile_width = amx::get_max_rows(max_palette);
    if (jcp.max_tiles != 8 || jcp.full_tile_width != 16)
        return status::unimplemented;

    // Pack n rows per tile, such that:
    // ow + (ow + gen_kw - 1) * (n - 1) <= jcp.full_tile_width
    auto calculate_tile_width = [&](int n) {
        assert(n > 0);
        return jcp.ow + (gen_kw + jcp.ow - 1) * (n - 1);
    };
    const bool ok_to_pack_tile = !jcp.is_relo
            && (utils::everyone_is(1, jcp.kh, jcp.kw)
                    || utils::everyone_is(1, jcp.stride_h, jcp.stride_w));
    const int max_oh_per_tile
            = 1 + (jcp.full_tile_width - jcp.ow) / (jcp.ow + gen_kw - 1);
    jcp.oh_per_tile = ok_to_pack_tile
            ? nstl::min(jcp.oh, nstl::max(1, max_oh_per_tile))
            : 1;
    jcp.tile_width = nstl::min<int>(
            jcp.full_tile_width, calculate_tile_width(jcp.oh_per_tile));
    jcp.ow_blocks = utils::div_up(jcp.ow, jcp.tile_width);

    // Prefer to use a single tile width when possible
    // (eg ow28 => 2 tiles of 14 vs 1 of 16 and 1 of 12)
    if (jcp.oh_per_tile == 1 && jcp.ow % jcp.ow_blocks == 0)
        jcp.tile_width = jcp.ow / jcp.ow_blocks;
    jcp.tile_tail = jcp.oh_per_tile == 1 ? jcp.ow % jcp.tile_width : 0;

    jcp.nb_oc_blocking = (jcp.nb_oc % 2 == 0) ? 2 : 1;
    jcp.nb_ic_blocking = 1;
    jcp.nb_oh_blocking
            = utils::everyone_is(true, jcp.tile_tail == 0,
                      // requirement for interleave stores
                      IMPLICATION(jcp.ow_blocks > 1, jcp.oh % 2 == 0),
                      // requirement for small spatial
                      utils::div_up(jcp.oh, jcp.oh_per_tile) > 1,
                      // choose maximal pbuffer overlap for reduced lowering
                      !jcp.is_relo)
            ? 2
            : 1;

    // TODO: tune oh blocking
    const int oh_blk_size_param = jcp.is_relo ? 1 : 10;
    const int oh_step_size = jcp.nb_oh_blocking * jcp.oh_per_tile;
    const int oh_blk_size = rnd_up(oh_blk_size_param, oh_step_size);
    jcp.oh_blk_size = rnd_up(nstl::min(jcp.oh, oh_blk_size), oh_step_size);
    // Here ihp means the input buffer height including padding (ie the number
    // of input rows required for computation of jcp.oh_blk_size output rows.
    // If an input row doesn't participate in the computation of any output row,
    // it isn't copied to the buffer at all (eg jcp.stride_h > gen_kh).
    jcp.ihp = jcp.is_relo
            ? jcp.oh_blk_size
            : (jcp.oh_blk_size - 1) * nstl::min(jcp.stride_h, gen_kh) + gen_kh;

    // TODO: tune ow blocking
    const int ow_blocks_per_call = jcp.is_relo ? 10 : 2;
    jcp.ow_block = nstl::min(jcp.ow, jcp.tile_width * ow_blocks_per_call);
    jcp.nb_ow = utils::div_up(jcp.ow, jcp.ow_block);
    // iwp includes all width elements that are really used in calculation
    // including left and right zero padding
    const bool are_sets_interleaved
            = IMPLICATION(jcp.dilate_w != 0, jcp.stride_w == 1);
    jcp.iwp = are_sets_interleaved
            ? (jcp.ow_block - 1) * nstl::min(jcp.stride_w, jcp.kw) + gen_kw
            : jcp.ow_block * jcp.kw;

    // Number of ops per tile store
    int ops_tile_store = jcp.tile_width;
    // Number of ops per accumulation tile
    int avaliable_ops = jcp.is_relo
            ? utils::div_up(jcp.nreduce, jcp.ic_block_int)
            : jcp.nb_ic_int * jcp.kh * (jcp.kw / jcp.kw_per_tile);
    // Number of vectors to store per tile operation
    // NOTE: set to zero to turn off interleave store (mostly for debugging)
    jcp.per_one_pstore = utils::div_up(ops_tile_store, avaliable_ops);

    if (jcp.is_relo) {
        jcp.inp_buffer_size = (size_t)jcp.nb_ic_int * jcp.ihp * jcp.iwp * jcp.kh
                        * jcp.ic_block_int_np
                // pbuffer pointer shifts each oh step for reduced-lowering
                + (jcp.oh - 1) * jcp.stride_h * jcp.ic_block_int_np
                // extra $line due to pbuffer writing full Zmm
                + jcp.ic_block_int;
    } else {
        jcp.inp_buffer_size = (size_t)jcp.nb_ic_int * jcp.kd
                * ((size_t)jcp.ihp * jcp.iwp * jcp.ic_block_int_np
                        // extra $line due to pbuffer writing full Zmm
                        + jcp.ic_block_int);
    }
    jcp.wei_buffer_size = (size_t)jcp.ngroups * jcp.nb_oc
            * rnd_up(jcp.kh * jcp.kw * jcp.ic * jcp.oc_block, 1024);
    jcp.wsp_buffer_size = (size_t)jcp.nb_oh_blocking * jcp.nb_oc_blocking
            * jcp.full_tile_width * jcp.oc_block;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;

    // Note: currently unsupported, results in seg-fault
    const int l_pad_output = nstl::min(jcp.ow, div_up(jcp.l_pad, jcp.stride_w));
    if (!jcp.is_relo && (l_pad_output > jcp.ow_block))
        return status::unimplemented;

    return status::success;
}

status_t jit_avx512_core_amx_fwd_kernel_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {

    size_t inp_buffer_size = jcp.nthr * jcp.inp_buffer_size;
    scratchpad.book(key_conv_amx_inp_buffer, inp_buffer_size, jcp.typesize_in);
    if (jcp.is_relo) {
        scratchpad.book(
                key_conv_amx_wei_buffer, jcp.wei_buffer_size, jcp.typesize_in);
    }

    size_t wsp_size = jcp.nthr * jcp.wsp_buffer_size;
    scratchpad.book(key_conv_amx_wsp_buffer, wsp_size, jcp.typesize_acc);
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding) {
        assert(jcp.ngroups == 1);
        scratchpad.book(key_conv_padded_bias, jcp.oc, jcp.typesize_bia);
    }
    scratchpad.book(key_conv_amx_tilecfg, 1, 64); // 1 whole cacheline

    // Keep scratchpad memory footprint under control
    const size_t L2_size_per_core = platform::get_per_core_cache_size(2);
    const size_t L3_size_per_core = platform::get_per_core_cache_size(3);
    const size_t max_scratchpad_size
            = jcp.nthr * (L2_size_per_core + L3_size_per_core);
    // TODO: tune this relationship as needed
    if (scratchpad.size() > max_scratchpad_size) return status::unimplemented;
    return status::success;
}

void jit_avx512_core_amx_bwd_data_copy_kernel_t::copy_row(
        const bool is_masked) {
    assert(jcp.is_nspc && "no support for nChw16c in this copy kernel");

    const bool is_bf16 = jcp.ddst_dt == data_type::bf16;
    const int inp_w_step
            = jcp.ngroups * jcp.oc_without_padding * jcp.typesize_in;
    const int inp_h_step = jcp.ow * inp_w_step;
    const int out_w_step = jcp.oc_block_int * jcp.typesize_in;
    const int out_h_step = jcp.owp * out_w_step;

    auto zero_it = [=](reg64_t tmp_out_ptr, int offset) {
        // no mask as output is a padded buffer
        if (is_bf16)
            vmovdqu16(ptr[tmp_out_ptr + offset], zmm_zero);
        else
            vmovdqu8(ptr[tmp_out_ptr + offset], zmm_zero);
    };

    auto copy_it = [=](reg64_t tmp_inp_ptr, int inp_off, reg64_t tmp_out_ptr,
                           int out_off) {
        Zmm zmm_load = is_masked ? zmm_tmp | ktail_mask | T_z : zmm_tmp;
        Zmm zmm_stor = zmm_tmp; // no mask as output is padded buffer
        if (is_bf16) {
            vmovdqu16(zmm_load, ptr[tmp_inp_ptr + inp_off]);
            vmovdqu16(ptr[tmp_out_ptr + out_off], zmm_stor);
        } else {
            vmovdqu8(zmm_load, ptr[tmp_inp_ptr + inp_off]);
            vmovdqu8(ptr[tmp_out_ptr + out_off], zmm_stor);
        }
    };

    mov(reg_ptr_aux_out, reg_ptr_out);

    { // Handle Top Overflow
        Label label_tov_loop, label_tov_skip;
        test(reg_tov, reg_tov);
        jz(label_tov_skip, T_NEAR);
        mov(reg_cnt_tmp, reg_tov);
        L(label_tov_loop);
        {
            for (int ow = 0; ow < jcp.owp; ow++) {
                const int offset = ow * out_w_step;
                zero_it(reg_ptr_aux_out, offset);
            }
            add(reg_ptr_aux_out, out_h_step);
            dec(reg_cnt_tmp);
            jnz(label_tov_loop, T_NEAR);
        }
        L(label_tov_skip);
    }

    mov(reg_ptr_aux_inp_h, reg_ptr_inp);

    // Handle Middle Loop
    Label label_khp_loop, label_khp_skip;
    test(reg_khp, reg_khp);
    jz(label_khp_skip, T_NEAR);
    mov(reg_cnt_khp, reg_khp);
    L(label_khp_loop);
    {
        Label label_lov, label_lov_skip;
        Label label_kwp, label_kwp_skip;
        Label label_rov, label_rov_skip;
        test(reg_lov, reg_lov);
        jnz(label_lov, T_NEAR);
        test(reg_kwp, reg_kwp);
        jnz(label_kwp, T_NEAR);
        test(reg_rov, reg_rov);
        jnz(label_rov, T_NEAR);

        test(reg_lov, reg_lov);
        jz(label_lov_skip, T_NEAR); // not really needed, but just to be safe
        L(label_lov); // Handle Left Overflow
        {
            Label label_lov_loop;
            mov(reg_cnt_tmp, reg_lov);
            L(label_lov_loop);
            {
                zero_it(reg_ptr_aux_out, 0);
                add(reg_ptr_aux_out, out_w_step);
                dec(reg_cnt_tmp);
                jnz(label_lov_loop, T_NEAR);
            }
        }
        L(label_lov_skip);

        test(reg_kwp, reg_kwp);
        jz(label_kwp_skip, T_NEAR);
        L(label_kwp); // Handle Center Loop
        {
            Label label_kwp_loop;
            mov(reg_ptr_aux_inp_w, reg_ptr_aux_inp_h);
            mov(reg_cnt_tmp, reg_kwp);
            L(label_kwp_loop);
            {
                copy_it(reg_ptr_aux_inp_w, 0, reg_ptr_aux_out, 0);
                add(reg_ptr_aux_out, out_w_step);
                add(reg_ptr_aux_inp_w, inp_w_step);
                dec(reg_cnt_tmp);

                if (jcp.stride_w > 1) {
                    jz(label_kwp_skip, T_NEAR);
                    // Handle Dilation-by-Stride
                    for (int sw = 0; sw < jcp.stride_w - 1; sw++) {
                        const int offset = sw * out_w_step;
                        zero_it(reg_ptr_aux_out, offset);
                    }
                    add(reg_ptr_aux_out, (jcp.stride_w - 1) * out_w_step);
                    if (jcp.stride_w == 2)
                        dec(reg_cnt_tmp);
                    else
                        sub(reg_cnt_tmp, jcp.stride_w - 1);
                    jmp(label_kwp_loop, T_NEAR);
                } else {
                    jnz(label_kwp_loop, T_NEAR);
                }
            }
        }
        L(label_kwp_skip);

        test(reg_rov, reg_rov);
        jz(label_rov_skip, T_NEAR);
        L(label_rov); // Handle Right Overflow
        {
            Label label_rov_loop;
            mov(reg_cnt_tmp, reg_rov);
            L(label_rov_loop);
            {
                zero_it(reg_ptr_aux_out, 0);
                add(reg_ptr_aux_out, out_w_step);
                dec(reg_cnt_tmp);
                jnz(label_rov_loop, T_NEAR);
            }
        }
        L(label_rov_skip);

        add(reg_ptr_aux_inp_h, inp_h_step);
        dec(reg_cnt_khp);

        if (jcp.stride_h > 1) {
            jz(label_khp_skip, T_NEAR);
            // Handle Dilation-by-Stride
            for (int sh = 0; sh < jcp.stride_h - 1; sh++) {
                for (int ow = 0; ow < jcp.owp; ow++) {
                    const int offset = sh * out_h_step + ow * out_w_step;
                    zero_it(reg_ptr_aux_out, offset);
                }
            }
            add(reg_ptr_aux_out, (jcp.stride_h - 1) * out_h_step);
            if (jcp.stride_h == 2)
                dec(reg_cnt_khp);
            else
                sub(reg_cnt_khp, jcp.stride_h - 1);
            jmp(label_khp_loop, T_NEAR);
        } else {
            jnz(label_khp_loop, T_NEAR);
        }
    }
    L(label_khp_skip);

    { // Handle Bottom Overflow
        Label label_bov_loop, label_bov_skip;
        test(reg_bov, reg_bov);
        jz(label_bov_skip, T_NEAR);
        mov(reg_cnt_tmp, reg_bov);
        L(label_bov_loop);
        {
            for (int ow = 0; ow < jcp.owp; ow++) {
                const int offset = ow * out_w_step;
                zero_it(reg_ptr_aux_out, offset);
            }
            add(reg_ptr_aux_out, out_h_step);
            dec(reg_cnt_tmp);
            jnz(label_bov_loop, T_NEAR);
        }
        L(label_bov_skip);
    }
}

void jit_avx512_core_amx_bwd_data_copy_kernel_t::generate() {

    const int inp_c_step = jcp.oc_block_int * jcp.typesize_in;
    const int out_c_step = jcp.ohp * jcp.owp * inp_c_step;
    const int nb_oc_int_no_tail = jcp.oc_without_padding / jcp.oc_block_int;
    const int oc_block_int_tail = jcp.oc_without_padding % jcp.oc_block_int;

    preamble();

    // pointer to 1st needed element in src buffer
    mov(reg_ptr_inp, ptr[param1 + GET_OFF(src)]);
    // pointer to 1st needed element in dst buffer
    mov(reg_ptr_out, ptr[param1 + GET_OFF(dst)]);

    // number of rows of src buffer to copy
    mov(reg_khp, ptr[param1 + GET_OFF(kh_padding)]);
    // number of zero-padded rows above src buffer to copy
    mov(reg_tov, ptr[param1 + GET_OFF(t_overflow)]);
    // number of zero-padded rows below src buffer to copy
    mov(reg_bov, ptr[param1 + GET_OFF(b_overflow)]);

    // number of columns of src buffer to copy
    mov(reg_kwp, ptr[param1 + GET_OFF(kw_padding)]);
    // number of zero-padded columns before src buffer to copy
    mov(reg_lov, ptr[param1 + GET_OFF(l_overflow)]);
    // number of zero-padded columns before src buffer to copy
    mov(reg_rov, ptr[param1 + GET_OFF(r_overflow)]);

    vpxord(zmm_zero, zmm_zero, zmm_zero);

    if (oc_block_int_tail > 0) {
        uint64_t mask = (UINT64_C(1) << oc_block_int_tail) - 1;
        mov(reg_tmp, mask);
        kmovq(ktail_mask, reg_tmp);
    }

    if (nb_oc_int_no_tail == 0) {
        copy_row(true); // masked
    } else if (nb_oc_int_no_tail == 1) {
        copy_row(false); // unmasked!
        if (oc_block_int_tail > 0) {
            add(reg_ptr_inp, inp_c_step);
            add(reg_ptr_out, out_c_step);
            copy_row(true); // masked
        }
    } else if (nb_oc_int_no_tail > 1) {
        mov(reg_cnt_ocb, nb_oc_int_no_tail);
        Label label_ocb_loop;
        L(label_ocb_loop);
        {
            copy_row(false); // unmasked!
            add(reg_ptr_inp, inp_c_step);
            add(reg_ptr_out, out_c_step);
            dec(reg_cnt_ocb);
            jnz(label_ocb_loop);
        }
        if (oc_block_int_tail > 0) copy_row(true); // masked
    }

    postamble();
}

// Tile register decomposition
// { C_BASE = 0, I_BASE = 4, W_BASE = 6, }
int jit_avx512_core_amx_bwd_data_kernel_t::get_out_tensor(int h, int i) const {
    const int C_BASE = 0;
    const int C_LAST = 4;
    assert(0 <= C_BASE && C_BASE < C_LAST && C_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(C_LAST);
    const int tile = C_BASE + h * jcp.nb_ih_blocking + i;
    assert(C_BASE <= tile && tile < C_LAST);
    return tile;
}
int jit_avx512_core_amx_bwd_data_kernel_t::get_inp_tensor(int h) const {
    const int I_BASE = 4;
    const int I_LAST = 6;
    assert(0 <= I_BASE && I_BASE < I_LAST && I_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(I_LAST);
    const int tile = I_BASE + h;
    assert(I_BASE <= tile && tile < I_LAST);
    return tile;
}
int jit_avx512_core_amx_bwd_data_kernel_t::get_wei_tensor(int i) const {
    const int W_BASE = 6;
    const int W_LAST = 8;
    assert(0 <= W_BASE && W_BASE < W_LAST && W_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(W_LAST);
    const int tile = W_BASE + i;
    assert(W_BASE <= tile && tile < W_LAST);
    return tile;
}

// Strides, shifts and offsets
// - inp is a padded buffer ~ [nb_oc_int][ohp][owp]{32c,64c}
// - weights is user buffer ~ OIhw16o16i{2o,4o}
// - output is tiled buffer ~ [NBIH][NBIC][tile_width][16c]
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_inp_kh_step() const {
    return (size_t)jcp.typesize_in * (jcp.dilate_h + 1) * jcp.owp
            * jcp.oc_block_int;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_inp_ocb_step() const {
    return (size_t)jcp.typesize_in * jcp.ohp * jcp.owp * jcp.oc_block_int;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_inp_shift() const {
    return (size_t)jcp.typesize_in * jcp.tile_width * jcp.oc_block_int;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_inp_offset(
        int ihb, int kh, int kw) const {
    // calculate offset by src height dimension
    size_t sp_offset = (size_t)ihb * jcp.owp;
    // add offset by kernel height dimension
    sp_offset += (size_t)(jcp.kh - 1 - kh) * (jcp.dilate_h + 1) * jcp.owp;
    // add offset by kernel width dimension
    sp_offset += (size_t)(jcp.kw - 1 - kw) * (jcp.dilate_w + 1);
    return jcp.typesize_in * sp_offset * jcp.oc_block_int;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_wei_kh_step() const {
    return (size_t)jcp.typesize_in * jcp.kw * jcp.oc_block_int * jcp.ic_block;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_wei_ocb_step() const {
    return (size_t)jcp.typesize_in * jcp.nb_ic * jcp.kh * jcp.kw
            * jcp.oc_block_int * jcp.ic_block;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_wei_offset(
        int icb, int kh, int kw) const {
    const size_t wei_kw_stride = jcp.oc_block_int * jcp.ic_block;
    const size_t wei_kh_stride = jcp.kw * wei_kw_stride;
    const size_t wei_icb_stride = jcp.kh * wei_kh_stride;
    return jcp.typesize_in
            * (icb * wei_icb_stride + kh * wei_kh_stride + kw * wei_kw_stride);
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_out_icb_offset(
        int ihb, int icb) const {
    size_t el_offset = jcp.is_nspc
            ? (size_t)icb * jcp.ic_block
                    + (size_t)ihb * jcp.iw * jcp.ngroups
                            * jcp.ic_without_padding
            : (size_t)icb * jcp.ih * jcp.iw * jcp.ic_block
                    + (size_t)ihb * jcp.iw * jcp.ic_block;
    return (size_t)jcp.typesize_out * el_offset;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_out_row_offset(
        int ihb, int icb, int j) const {
    size_t offset_w = jcp.is_nspc ? (size_t)jcp.typesize_out * j * jcp.ngroups
                    * jcp.ic_without_padding
                                  : (size_t)jcp.typesize_out * j * jcp.ic_block;
    return get_out_icb_offset(ihb, icb) + offset_w;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_out_shift(int width) const {
    return jcp.is_nspc ? (size_t)jcp.typesize_out * width * jcp.ngroups
                    * jcp.ic_without_padding
                       : (size_t)jcp.typesize_out * width * jcp.ic_block;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_wsp_icb_offset(
        int ihb, int icb) const {
    size_t el_offset = (size_t)icb * prv_width_ * jcp.ic_block
            + (size_t)ihb * jcp.nb_ic_blocking * jcp.full_tile_width
                    * jcp.ic_block;
    return jcp.typesize_acc * el_offset;
}
size_t jit_avx512_core_amx_bwd_data_kernel_t::get_wsp_row_offset(
        int ihb, int icb, int j) const {
    return get_wsp_icb_offset(ihb, icb)
            + (size_t)jcp.typesize_acc * j * jcp.ic_block;
}

// Code generation
void jit_avx512_core_amx_bwd_data_kernel_t::prepare_output() {
    for (int h = 0; h < jcp.nb_ih_blocking; h++)
        for (int i = 0; i < jcp.nb_ic_blocking; i++)
            tilezero(Tmm(get_out_tensor(h, i)));
}

void jit_avx512_core_amx_bwd_data_kernel_t::init_runtime_counters(
        bool start_with_last_tile_block) {
    prv_width_ = start_with_last_tile_block && jcp.tile_tail > 0
            ? jcp.tile_tail
            : jcp.tile_width;

    row_count_ = 0;
    is_store_done_ = false;
    is_buffer_empty_ = true;
}

bool jit_avx512_core_amx_bwd_data_kernel_t::maybe_eltwise(int position) {
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

Ymm jit_avx512_core_amx_bwd_data_kernel_t::ymm_mask(
        const Ymm ymm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? ymm_in | ktail_mask : ymm_in | ktail_mask | T_z)
                     : ymm_in;
}

Zmm jit_avx512_core_amx_bwd_data_kernel_t::zmm_mask(
        const Zmm zmm_in, bool mask_flag, bool store) {
    return mask_flag ? (store ? zmm_in | ktail_mask : zmm_in | ktail_mask | T_z)
                     : zmm_in;
}

void jit_avx512_core_amx_bwd_data_kernel_t::cvt2ps(data_type_t type_in,
        const Zmm zmm_in, const Operand &op, bool mask_flag = false) {
    const Zmm zmm = zmm_mask(zmm_in, mask_flag);
    switch (type_in) {
        case data_type::f32:
        case data_type::s32: vmovups(zmm, op); break;
        case data_type::s8: vpmovsxbd(zmm, op); break;
        case data_type::u8: vpmovzxbd(zmm, op); break;
        default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32) vcvtdq2ps(zmm_in, zmm_in);
}

void jit_avx512_core_amx_bwd_data_kernel_t::store_output_vector_bf16(
        Zmm zmm_out, int icb, int h, int w) {
    const bool mask_flag = jcp.is_nspc && jcp.ic_without_padding != jcp.ic
            && icb == (jcp.nb_ic_blocking - 1);

    auto addr = EVEX_compress_addr(reg_out_ptr, get_out_row_offset(h, icb, w));

    const auto &p = attr_.post_ops_;

    const int sum_idx = p.find(primitive_kind::sum);
    if (sum_idx != -1) {
        if (jcp.dsrc_dt == data_type::bf16) {
            vpmovzxwd(zmm_mask(zmm_prev_dst, mask_flag), addr);
            vpslld(zmm_prev_dst, zmm_prev_dst, 16);
            vaddps(zmm_out, zmm_prev_dst);
        } else {
            vmovups(zmm_mask(zmm_prev_dst, mask_flag), addr);
            vaddps(zmm_out, zmm_prev_dst);
        }
    }
    if (jcp.with_bias) {
        int bias_offset = jcp.typesize_bia * icb * jcp.ic_block;
        auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        if (jcp.bia_dt == data_type::bf16) {
            vpmovzxwd(zmm_mask(zmm_bias, mask_flag), bias_addr);
            vpslld(zmm_bias, zmm_bias, 16);
            vaddps(zmm_out, zmm_bias);
        } else
            vaddps(zmm_mask(zmm_out, mask_flag), bias_addr);
    }

    const int eltwise_ind = p.find(primitive_kind::eltwise);
    if (eltwise_ind != -1) eltwise_injector_->compute_vector(zmm_out.getIdx());

    if (jcp.dsrc_dt == data_type::bf16) {
        Ymm ymm_out = Ymm(zmm_out.getIdx());
        vcvtneps2bf16(ymm_out, zmm_out);
        vmovdqu16(addr, ymm_mask(ymm_out, mask_flag, true));
    } else {
        vmovups(addr, zmm_mask(zmm_out, mask_flag, true));
    }
}

void jit_avx512_core_amx_bwd_data_kernel_t::store_output_vector_int8(
        Zmm zmm_out, int icb, int h, int w) {
    const int nb_ic_block = jcp.nb_ic_blocking;
    const int ic_block = jcp.ic_block;
    const bool mask_flag = true && jcp.ic_without_padding != jcp.ic
            && icb == (nb_ic_block - 1);

    auto addr = EVEX_compress_addr(reg_out_ptr, get_out_row_offset(h, icb, w));

    const auto &p = attr_.post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    const float *p_sum_scale = nullptr;
    if (sum_idx != -1) {
        const auto &p_entry = p.entry_[sum_idx];
        p_sum_scale = &p_entry.sum.scale;
    }

    if (p_sum_scale && *p_sum_scale != 1.f)
        mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

    int scale_offset = jcp.is_ic_scale * (sizeof(float) * icb * ic_block);
    if (jcp.with_bias) {
        int bias_offset = jcp.typesize_bia * icb * ic_block;
        auto bias_addr = EVEX_compress_addr(reg_bias, bias_offset);
        cvt2ps(jcp.bia_dt, zmm_bias, bias_addr, mask_flag);
    }
    /* add bias to zmm_accum */
    vcvtdq2ps(zmm_out, zmm_out);
    if (jcp.with_bias) vaddps(zmm_out, zmm_out, zmm_bias);
    const Zmm zmm_out_msk = zmm_mask(zmm_out, mask_flag);
    vmulps(zmm_out_msk, zmm_out,
            EVEX_compress_addr(reg_ptr_scales, scale_offset));

    /* Do post-ops */
    if (maybe_eltwise(0)) eltwise_injector_->compute_vector(zmm_out.getIdx());
    if (p_sum_scale) { // post_op: sum
        cvt2ps(jcp.dsrc_dt, zmm_prev_dst, addr, mask_flag);
        if (*p_sum_scale == 1.f)
            vaddps(zmm_out, zmm_prev_dst);
        else
            vfmadd231ps(zmm_out, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
    }
    if (maybe_eltwise(1)) eltwise_injector_->compute_vector(zmm_out.getIdx());

    // Properly saturate the accumulators for integer datatypes
    if (one_of(jcp.dsrc_dt, u8, s8, s32)) {
        init_saturate_f32(
                zmm_zero, zmm_saturation, reg_aux_saturation, f32, jcp.dsrc_dt);
        saturate_f32(zmm_out, zmm_zero, zmm_saturation, jcp.dsrc_dt);
        vcvtps2dq(zmm_out, zmm_out);
    }

    const Zmm zmm_out_store = zmm_mask(zmm_out, mask_flag, true);

    switch (jcp.dsrc_dt) {
        case data_type::f32:
        case data_type::s32: vmovups(addr, zmm_out_store); break;
        case data_type::s8: vpmovsdb(addr, zmm_out_store); break;
        case data_type::u8: vpmovusdb(addr, zmm_out_store); break;
        default: assert(!"unknown dst_dt");
    }
}

void jit_avx512_core_amx_bwd_data_kernel_t::store_output_vector(
        Zmm zmm_out, int icb, int h, int w) {
    /*
    Output:
              jcp.is_nspc              !jcp.is_nspc
              ---------------------    ---------------------
        INT8: [N][H][W][NBIC][16IC]
        BF16: [N][H][W][NBIC][16IC] or [N][NBIC][H][W][16IC]
    */
    if (jcp.ddst_dt == data_type::bf16) {
        store_output_vector_bf16(zmm_out, icb, h, w);
    } else {
        store_output_vector_int8(zmm_out, icb, h, w);
    }
}

void jit_avx512_core_amx_bwd_data_kernel_t::store_output(
        int width, bool do_store) {
    auto store_output_block = [=](int width, bool do_store,
                                      bool is_last_ih_blks) {
        // Calculate the number of ih blocks; it may differ on last call
        const int n_ih_blks = is_last_ih_blks ? jcp.ih % jcp.nb_ih_blocking
                                              : jcp.nb_ih_blocking;
        for (int icb = 0; icb < jcp.nb_ic_blocking; icb++) {
            for (int ihb = 0; ihb < n_ih_blks; ihb++) {
                /* Formats: Workspace: [NBIH][NBIC][W][16OC] */
                tilestored(ptr[reg_wsp_ptr + reg_wei_stride
                                   + get_wsp_icb_offset(ihb, icb)],
                        Tmm(get_out_tensor(ihb, icb)));
                is_buffer_empty_ = false;
                is_store_done_ = false;
                for (int tw = 0; tw < width && do_store; tw++) {
                    Zmm zmm_out = Zmm(tw);
                    vmovups(zmm_out,
                            ptr[reg_wsp_ptr
                                    + get_wsp_row_offset(ihb, icb, tw)]);
                    store_output_vector(zmm_out, icb, ihb, tw);
                }
            }
        }
    };

    // adjustment in case interleave store is turned off
    do_store = do_store || jcp.per_one_pstore == 0;
    if (jcp.ih % jcp.nb_ih_blocking == 0) {
        store_output_block(width, do_store, /* is_last_ih_blks = */ false);
    } else {
        Label label_full_store, label_done;
        cmp(reg_last_h, 0);
        jne(label_full_store, T_NEAR);
        store_output_block(width, do_store, /* is_last_ih_blks = */ true);
        jmp(label_done, T_NEAR);
        L(label_full_store);
        store_output_block(width, do_store, /* is_last_ih_blks = */ false);
        L(label_done);
    }
    if (do_store) add(reg_out_ptr, get_out_shift(width));
}

void jit_avx512_core_amx_bwd_data_kernel_t::interleave_store(int width) {
    for (int c = 0;
            c < jcp.per_one_pstore && !is_store_done_ && !is_buffer_empty_;
            c++) {
        // row_count = ihb * ICB * TW + icb * TW + tw
        int tw = row_count_ % prv_width_;
        int icb = (row_count_ / prv_width_) % jcp.nb_ic_blocking;
        int ihb = (row_count_ / prv_width_) / jcp.nb_ic_blocking;

        Zmm zmm_out = Zmm(tw);
        vmovups(zmm_out, ptr[reg_wsp_ptr + get_wsp_row_offset(ihb, icb, tw)]);
        store_output_vector(zmm_out, icb, ihb, tw);
        row_count_++;

        if (row_count_
                == prv_width_ * jcp.nb_ic_blocking * jcp.nb_ih_blocking) {
            add(reg_out_ptr, get_out_shift(prv_width_));
            row_count_ = 0;
            is_store_done_ = true;
            prv_width_ = width;
        }
    }
}

void jit_avx512_core_amx_bwd_data_kernel_t::compute_ocb_loop(
        int width, bool do_store) {

    auto tdpbxxd = [=](const Tmm &x1, const Tmm &x2, const Tmm &x3) {
        switch (jcp.ddst_dt) {
            using namespace data_type;
            case bf16: tdpbf16ps(x1, x2, x3); break;
            case s8: tdpbssd(x1, x2, x3); break;
            case u8: tdpbusd(x1, x2, x3); break;
            default: assert(!"unsupported data type");
        }
    };

    prepare_output();

    for (int ocb = 0; ocb < jcp.nb_oc_int; ocb++) {
        // reverse order through spatial components of weights so that
        // input buffer is accessed in a monotonically increasing fashion
        for (int kh = jcp.kh - 1; kh >= 0; kh--) {
            for (int kw = jcp.kw - 1; kw >= 0; kw--) {
                for (int ihb = 0; ihb < jcp.nb_ih_blocking; ihb++) {
                    tileloadd(Tmm(get_inp_tensor(ihb)),
                            ptr[reg_inp_ptr + get_inp_offset(ihb, kh, kw)
                                    + reg_inp_stride]);
                }
                for (int icb = 0; icb < jcp.nb_ic_blocking; icb++) {
                    tileloadd(Tmm(get_wei_tensor(icb)),
                            ptr[reg_wei_ptr + get_wei_offset(icb, kh, kw)
                                    + reg_wei_stride]);
                    for (int ihb = 0; ihb < jcp.nb_ih_blocking; ihb++) {
                        tdpbxxd(Tmm(get_out_tensor(ihb, icb)),
                                Tmm(get_inp_tensor(ihb)),
                                Tmm(get_wei_tensor(icb)));
                        interleave_store(width);
                    }
                }
            }
        }
        add(reg_inp_ptr, get_inp_ocb_step());
        add(reg_wei_ptr, get_wei_ocb_step());
    }
    sub(reg_inp_ptr, get_inp_ocb_step() * jcp.nb_oc_int);
    sub(reg_wei_ptr, get_wei_ocb_step() * jcp.nb_oc_int);

    store_output(width, do_store);

    add(reg_inp_ptr, get_inp_shift());
}

void jit_avx512_core_amx_bwd_data_kernel_t::compute_iw_loop() {
    auto compute_iw_loop_body = [=](bool last_iwb, int num_tile_blocks) {
        int gen_tile_tail = last_iwb && jcp.tile_tail > 0 ? jcp.tile_tail
                                                          : jcp.tile_width;
        init_runtime_counters(last_iwb && num_tile_blocks == 1);
        for (int iwb = 0; iwb < num_tile_blocks - 1; iwb++)
            compute_ocb_loop(jcp.tile_width, false);
        compute_ocb_loop(gen_tile_tail, true);
    };

    if (jcp.nb_iw == 1) {
        compute_iw_loop_body(true, jcp.iw_blocks);
    } else {
        Label label_done;
        int iw_blocks_per_call = div_up(jcp.iw_block, jcp.tile_width);
        int last_iwb_tile_blocks = jcp.iw_blocks % iw_blocks_per_call;
        if (last_iwb_tile_blocks == 0 && jcp.tile_tail > 0)
            last_iwb_tile_blocks = iw_blocks_per_call;
        if (last_iwb_tile_blocks > 0) {
            Label label_not_last_iwb;
            mov(reg_tmp, ptr[param1 + GET_OFF(iwb)]);
            cmp(reg_tmp, jcp.nb_iw - 1);
            jne(label_not_last_iwb, T_NEAR);

            compute_iw_loop_body(true, last_iwb_tile_blocks);

            jmp(label_done, T_NEAR);

            L(label_not_last_iwb);
        }
        compute_iw_loop_body(false, iw_blocks_per_call);

        L(label_done);
    }
}

void jit_avx512_core_amx_bwd_data_kernel_t::generate() {
    preamble();

    mov(reg_inp_ptr, ptr[param1 + GET_OFF(dst)]); // padded buffer of diff_dst
    mov(reg_wei_ptr, ptr[param1 + GET_OFF(filt)]); // weights
    mov(reg_out_ptr, ptr[param1 + GET_OFF(src)]); // diff_src
    mov(reg_wsp_ptr, ptr[param1 + GET_OFF(acc_s32)]);

    if (jcp.with_bias) mov(reg_bias, ptr[param1 + GET_OFF(bias)]);

    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);

    mov(reg_last_h, ptr[param1 + GET_OFF(last_h)]);

    const int inp_stride = jcp.oc_block_int * jcp.typesize_in;
    const int wei_stride = jcp.ic_block * jcp.typesize_acc;
    mov(reg_inp_stride, inp_stride);
    mov(reg_wei_stride, wei_stride);

    if (jcp.is_nspc && jcp.ic_without_padding != jcp.ic) {
        // Use mask 0xF by default for all output data and post-ops
        // loads / stores with block index
        // icb = icc * jcp.nb_ic_blocking + (jcp.nb_ic_blocking - 1)
        // TODO: use masked loads / stores for the last icc only
        int current_block_size = jcp.ic_block;
        int mask = (1 << current_block_size) - 1;
        Xbyak::Reg32 regw_tmp = reg_tmp.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
        Xbyak::Label mask_is_set;
        mov(reg_ic_blocks, ptr[param1 + GET_OFF(ic_blocks)]);
        cmp(reg_ic_blocks, jcp.nb_ic - jcp.nb_ic_blocking);
        jne(mask_is_set, T_NEAR);
        // Reset the mask
        current_block_size = jcp.ic_without_padding % jcp.ic_block;
        mask = (1 << current_block_size) - 1;
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);

        L(mask_is_set);
    }
    compute_iw_loop();

    postamble();

    if (jcp.with_eltwise) eltwise_injector_->prepare_table();
}

bool jit_avx512_core_amx_bwd_data_kernel_t::post_ops_ok(
        const jit_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;
    const bool is_bf16 = jcp.ddst_dt == data_type::bf16;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    auto is_sum = [&](int idx) {
        if (is_bf16)
            return p.entry_[idx].is_sum();
        else
            return p.contain(sum, idx);
    };

    switch (p.len()) {
        case 0: return true;
        case 1: return is_eltwise(0) || is_sum(0);
        case 2:
            return (is_sum(0) && is_eltwise(1))
                    || (!is_bf16 && is_sum(1) && is_eltwise(0));
        default: return false;
    }

    return false;
}

void jit_avx512_core_amx_bwd_data_kernel_t::tile_configure(char *tcfg_buff) {
    const int vnni_width = jcp.ddst_dt == data_type::bf16 ? 2 : 4;
    // Input tile dimensions
    const int a_col = jcp.oc_block_int;
    const int a_row = jcp.tile_width;
    // Weights tile dimensions
    const int b_col = jcp.ic_block * vnni_width;
    const int b_row = a_col / vnni_width;
    // Accumulator tile dimensions
    const int c_col = jcp.ic_block;
    const int c_row = a_row;

    for (size_t i = 0; i < 64; i++)
        tcfg_buff[i] = 0;

    // Weights (W_BASE) Tensor Tiles
    for (int i = 0; i < jcp.nb_ic_blocking; i++)
        tc_configure_tile((palette_config_t *)tcfg_buff, get_wei_tensor(i),
                b_row, b_col * jcp.typesize_in);

    // Input (I_BASE) and Accumulator (C_BASE) Tensor Tiles
    for (int h = 0; h < jcp.nb_ih_blocking; h++) {
        tc_configure_tile((palette_config_t *)tcfg_buff, get_inp_tensor(h),
                a_row, a_col * jcp.typesize_in);
        for (int i = 0; i < jcp.nb_ic_blocking; i++)
            tc_configure_tile((palette_config_t *)tcfg_buff,
                    get_out_tensor(h, i), c_row, c_col * jcp.typesize_acc);
    }

    ((palette_config_t *)tcfg_buff)->palette_id = amx::get_max_palette();
}

status_t jit_avx512_core_amx_bwd_data_kernel_t::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &diff_src_md,
        memory_desc_t &weights_md, memory_desc_t &diff_dst_md,
        memory_desc_t *bias_md, const primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    const memory_desc_wrapper diff_src_d(&diff_src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    const memory_desc_wrapper bias_d(bias_md);

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();
    bool is_1d = ndims == 3;
    bool is_3d = ndims == 5;

    if (is_3d) return status::unimplemented;

    const bool is_bf16_convolution = everyone_is(true,
            diff_dst_d.data_type() == data_type::bf16,
            weights_d.data_type() == data_type::bf16,
            one_of(diff_src_d.data_type(), data_type::bf16, data_type::f32));

    // TODO: support int8 deconvolution
    bool supported = is_bf16_convolution && mayiuse(avx512_core_bf16_amx_bf16);
    if (!supported) return status::unimplemented;

    jcp = zero<decltype(jcp)>();
    jcp.isa = is_bf16_convolution ? avx512_core_bf16_amx_bf16
                                  : avx512_core_bf16_amx_int8;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;

    jcp.mb = diff_src_d.dims()[0];
    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.ih = !is_1d ? diff_src_d.dims()[ndims - 2] : 1;
    jcp.iw = diff_src_d.dims()[ndims - 1];
    jcp.oh = !is_1d ? diff_dst_d.dims()[ndims - 2] : 1;
    jcp.ow = diff_dst_d.dims()[ndims - 1];
    jcp.kh = !is_1d ? weights_d.dims()[with_groups + ndims - 2] : 1;
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.t_pad = !is_1d ? cd.padding[0][ndims - 4] : 0;
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_h = !is_1d ? cd.strides[ndims - 4] : 1;
    jcp.stride_w = cd.strides[ndims - 3];

    // No bias for bf16 case to simplify integration with ref_deconvolution
    jcp.with_bias = bias_md && !is_bf16_convolution
            && cd.bias_desc.format_kind != format_kind::undef;

    jcp.dilate_h = !is_1d ? cd.dilates[ndims - 4] : 0;
    jcp.dilate_w = cd.dilates[ndims - 3];

    const int gen_kh = (jcp.kh - 1) * (jcp.dilate_h + 1) + 1;
    const int gen_kw = (jcp.kw - 1) * (jcp.dilate_w + 1) + 1;
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, gen_kh);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, gen_kw);
    if (jcp.l_pad >= gen_kw || jcp.r_pad >= gen_kw || jcp.t_pad >= gen_kh
            || jcp.b_pad >= gen_kh)
        return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.ddst_dt = cd.diff_dst_desc.data_type;
    jcp.dsrc_dt = cd.diff_src_desc.data_type;
    jcp.wei_dt = cd.weights_desc.data_type;

    jcp.is_depthwise = true && with_groups && everyone_is(1, jcp.ic, jcp.oc);

    if (jcp.is_depthwise)
        return status::unimplemented; // TODO: add support of DW convolution

    format_tag_t dat_tag_ncsp
            = pick(ndims - 3, format_tag::nCw16c, format_tag::nChw16c);
    format_tag_t dat_tag_nspc
            = pick(ndims - 3, format_tag::nwc, format_tag::nhwc);
    // To toggle the default data layout for BF16 between nChw16c and nhwc,
    // swap the following two variable definitions. Current choice: nhwc.
    format_tag_t dat_tag_opt = dat_tag_nspc;
    format_tag_t dat_tag_alt
            = is_bf16_convolution ? dat_tag_ncsp : dat_tag_nspc;

    if (diff_src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_src_md, dat_tag_opt));
        jcp.src_tag = dat_tag_opt;
    } else
        jcp.src_tag = diff_src_d.matches_one_of_tag(dat_tag_alt, dat_tag_opt);

    if (!one_of(jcp.src_tag, dat_tag_alt, dat_tag_opt))
        return status::unimplemented;

    jcp.is_nspc = jcp.src_tag == dat_tag_nspc;

    // TODO: enable support for nChw16c
    if (!jcp.is_nspc) return status::unimplemented;

    if (diff_dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, jcp.src_tag));
        jcp.dst_tag = jcp.src_tag;
    } else
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(jcp.src_tag);

    if (jcp.dst_tag != jcp.src_tag) return status::unimplemented;

    if (jcp.with_bias && bias_d.format_kind() == format_kind::any)
        CHECK(memory_desc_init_by_tag(*bias_md, format_tag::x));

    jcp.nthr = nthreads;

    jcp.ic_block = 16;
    jcp.oc_block = 16;

    if (jcp.ngroups == 1) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    bool args_ok = jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0;
    if (!args_ok) return status::unimplemented;

    const int vnni_width = is_bf16_convolution ? 2 : 4;
    jcp.oc_block_int = jcp.oc_block * vnni_width; // 32 for bf16, 64 for int8

    if (!post_ops_ok(jcp, attr)) return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    auto set_or_check_wei_format = [&]() {
        using namespace format_tag;
        format_tag_t wei_tag = pick(with_groups + 2 * (ndims - 3), OIw16o16i2o,
                gOIw16o16i2o, OIhw16o16i2o, gOIhw16o16i2o);

        memory_desc_t want_wei_md = weights_md;
        memory_desc_init_by_tag(want_wei_md, wei_tag);

        if (weights_md.format_kind == format_kind::any) {
            weights_md = want_wei_md;
            return true;
        }
        return weights_md == want_wei_md;
    };

    if (!set_or_check_wei_format()) return status::unimplemented;

    jcp.typesize_in = types::data_type_size(diff_dst_d.data_type());
    jcp.typesize_out = types::data_type_size(diff_src_d.data_type());
    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;
    jcp.typesize_acc = sizeof(int32_t);

    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_oc_int = div_up(jcp.oc, jcp.oc_block_int);

    const int max_palette = amx::get_max_palette();
    jcp.max_tiles = amx::get_max_tiles(max_palette);
    jcp.full_tile_width = amx::get_max_rows(max_palette);
    if (jcp.max_tiles != 8 || jcp.full_tile_width != 16)
        return status::unimplemented;

    jcp.tile_width = nstl::min(jcp.full_tile_width, jcp.iw);
    jcp.iw_blocks = div_up(jcp.iw, jcp.tile_width);

    // Prefer to use a single tile width when possible
    // (eg iw28 => 2 tiles of 14 vs 1 of 16 and 1 of 12)
    if (jcp.iw % jcp.iw_blocks == 0) jcp.tile_width = jcp.iw / jcp.iw_blocks;
    jcp.tile_tail = jcp.iw % jcp.tile_width;

    jcp.nb_ic_blocking = (jcp.nb_ic % 2 == 0) ? 2 : 1;
    jcp.nb_ih_blocking
            = everyone_is(true, jcp.ih > 1,
                      // requirement for interleave stores
                      IMPLICATION(jcp.iw_blocks > 1, jcp.ih % 2 == 0))
            ? 2
            : 1;

    // TODO: tune ih blocking
    const int ih_blk_size_tmp = 10;
    const int ih_step = jcp.nb_ih_blocking;
    jcp.ih_blk_size = rnd_up(nstl::min(jcp.ih, ih_blk_size_tmp), ih_step);
    // ohp includes all elements that are really used in calculation,
    // including zero-padded "dilate-by-strides" and top and bottom overflow
    jcp.ohp = jcp.ih_blk_size + gen_kh - 1;

    // TODO: tune iw blocking
    const int iw_blocks_per_call = 2;
    jcp.iw_block = jcp.tile_width * iw_blocks_per_call;
    jcp.nb_iw = div_up(jcp.iw, jcp.iw_block);
    // owp includes all elements that are really used in calculation,
    // including zero-padded "dilate-by-strides" and left and right overflow
    jcp.owp = jcp.iw_block + gen_kw - 1;

    // Number of ops per tile store
    int ops_tile_store = jcp.tile_width;
    // Number of ops per accumulation tile
    int avaliable_ops = jcp.nb_oc_int * jcp.kh * jcp.kw;
    // Number of vectors to store per tile operation
    // NOTE: set to zero to turn off interleave store (mostly for debugging)
    jcp.per_one_pstore = div_up(ops_tile_store, avaliable_ops);

    jcp.inp_buffer_size
            = (size_t)jcp.nb_oc_int * jcp.ohp * jcp.owp * jcp.oc_block_int;
    jcp.wsp_buffer_size = (size_t)jcp.nb_ih_blocking * jcp.nb_ic_blocking
            * jcp.full_tile_width * jcp.ic_block;

    const auto &oscales = attr.output_scales_;
    jcp.is_ic_scale = oscales.mask_ == 1 << 1;

    return status::success;
}

void jit_avx512_core_amx_bwd_data_kernel_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        const primitive_attr_t &attr) {

    size_t inp_buffer_size = jcp.nthr * jcp.inp_buffer_size;
    scratchpad.book(key_conv_amx_inp_buffer, inp_buffer_size, jcp.typesize_in);
    size_t wsp_size = jcp.nthr * jcp.wsp_buffer_size;
    scratchpad.book(key_conv_amx_wsp_buffer, wsp_size, jcp.typesize_acc);
    if (jcp.with_bias && jcp.ic != jcp.ic_without_padding) {
        assert(jcp.ngroups == 1);
        scratchpad.book(key_conv_padded_bias, jcp.ic, jcp.typesize_bia);
    }
    scratchpad.book(key_conv_amx_tilecfg, 1, 64); // 1 whole cacheline
}

const int jit_avx512_core_amx_bwd_weights_kernel_t::max_ur_w = 32;

// Tile register decomposition
// { C_BASE = 0, A_BASE = 4, B_BASE = 6, }
int jit_avx512_core_amx_bwd_weights_kernel_t::get_wei_tensor(
        int ocb, int icb) const {
    const int C_BASE = 0;
    const int C_LAST = 4;
    assert(0 <= C_BASE && C_BASE < C_LAST && C_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(C_LAST);
    const int tile = C_BASE + ocb * jcp.nb_oc_blocking + icb;
    assert(C_BASE <= tile && tile < C_LAST);
    return tile;
}
int jit_avx512_core_amx_bwd_weights_kernel_t::get_src_tensor(int icb) const {
    const int A_BASE = 4;
    const int A_LAST = 6;
    assert(0 <= A_BASE && A_BASE < A_LAST && A_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(A_LAST);
    const int tile = A_BASE + icb;
    assert(A_BASE <= tile && tile < A_LAST);
    return tile;
}
int jit_avx512_core_amx_bwd_weights_kernel_t::get_ddst_tensor(int ocb) const {
    const int B_BASE = 6;
    const int B_LAST = 8;
    assert(0 <= B_BASE && B_BASE < B_LAST && B_LAST <= jcp.max_tiles);
    MAYBE_UNUSED(B_LAST);
    const int tile = B_BASE + ocb;
    assert(B_BASE <= tile && tile < B_LAST);
    return tile;
}

void jit_avx512_core_amx_bwd_weights_kernel_t::tile_configure(char *tcfg_buff) {
    // Input tile dimensions
    const int a_col = jcp.ur_w;
    const int a_row = jcp.ic_block;
    // Weights tile dimensions
    const int b_col = jcp.oc_block * 2;
    const int b_row = a_col / 2;
    // Accumulator tile dimensions
    const int c_col = jcp.oc_block;
    const int c_row = a_row;

    for (size_t i = 0; i < 64; i++)
        tcfg_buff[i] = 0;

    for (int icb = 0; icb < jcp.nb_ic_blocking; icb++)
        tc_configure_tile((palette_config_t *)tcfg_buff, get_src_tensor(icb),
                a_row, a_col * jcp.typesize_in);

    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++)
        tc_configure_tile((palette_config_t *)tcfg_buff, get_ddst_tensor(ocb),
                b_row, b_col * jcp.typesize_in);

    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++)
        for (int icb = 0; icb < jcp.nb_ic_blocking; icb++)
            tc_configure_tile((palette_config_t *)tcfg_buff,
                    get_wei_tensor(ocb, icb), c_row, c_col * jcp.typesize_out);

    ((palette_config_t *)tcfg_buff)->palette_id = amx::get_max_palette();
}

void jit_avx512_core_amx_bwd_weights_kernel_t::od_step_comeback_pointers() {
    Label kd_comeback_label;
    mov(kj, reg_kd_count);
    L(kd_comeback_label);
    {
        sub(reg_src, get_src_offset(0, 0, filter_d_to_src(1)));
        sub(reg_kernel, get_kernel_offset(0, jcp.kh * jcp.kw));
        dec(kj);
        jnz(kd_comeback_label, T_NEAR);
    }
}

void jit_avx512_core_amx_bwd_weights_kernel_t::oh_step_comeback_pointers() {
    Label kh_comeback_label;
    mov(kj, reg_kh);
    L(kh_comeback_label);
    {
        sub(reg_src, get_src_offset(0, 0, filter_h_to_src(1)));
        sub(reg_kernel, get_kernel_offset(0, jcp.kw));
        dec(kj);
        jnz(kh_comeback_label, T_NEAR);
    }
}

void jit_avx512_core_amx_bwd_weights_kernel_t::compute_full_spat_loop(
        int nb_ic_blocking, int nb_oc_blocking) {
    // General code layout:
    //
    // Blocking over OH -- top level
    // (Reduces L2 pressure; not very useful right now)
    //  Loop over all KHxKW kernel -- emit_kh_kw_loop()
    //    Loop over OH block -- emit_h_loop()
    //      Loop over OW blocks -- emit_fma_block()
    //      (Supports both fully unrolled and partially unrolled
    //      versions to reduce code size)
    //          Loop over OW block -- emit_fma_step()

    auto src_row_size = get_src_offset(0, 0, 1);
    auto ddst_row_size = get_ddst_offset(0, 1);
    auto row_size = src_row_size + ddst_row_size;

    int h_block_size = jcp.oh;
    int h_last_block_size = h_block_size;
    int min_h_block_size = nstl::max(1, nstl::max(jcp.b_pad, jcp.t_pad));
    auto working_set_size = row_size * h_block_size;

    if (working_set_size > full_spat_max_working_set_size) {
        assert(full_spat_opt_working_set_size < full_spat_max_working_set_size);

        while (working_set_size > full_spat_opt_working_set_size
                && h_block_size >= min_h_block_size) {
            for (int i = 2; i <= h_block_size; i++)
                if (i == h_block_size)
                    h_block_size = h_block_size / 2;
                else if (h_block_size % i == 0) {
                    h_block_size = h_block_size / i;
                    break;
                }
            working_set_size = row_size * h_block_size;
        }
        h_block_size = nstl::max(min_h_block_size, h_block_size);
        h_last_block_size = jcp.oh % h_block_size;
        if (h_last_block_size < jcp.b_pad) h_last_block_size += h_block_size;
    }

    Opmask reg_h_block = k1;
    Reg64 reg_kh = rax;
    Reg64 reg_kw = rbx;
    Reg64 reg_tmp = abi_not_param1;
    Reg32 reg_tmp_w = reg_tmp.cvt32();
    Reg64 reg_ohs = rdx;
    Reg64 reg_ihs = rsi;
    Reg64 reg_h = r8;
    Reg64 reg_j = r10;

    Reg64 reg_src = r13;
    Reg64 reg_ddst = r14;
    Reg64 reg_ker = r15;

    Reg64 reg_dense_stride = abi_param1;
    Reg64 reg_a_stride = reg_tmp;

    auto emit_block = [&]() {
        mov(reg_a_stride, jcp.tr_iw * jcp.typesize_in);
        for (int ur_w_b = 0; ur_w_b < jcp.ur_w_blocks; ur_w_b++) {
            dim_t ur_w_src_offset = ur_w_b * get_src_offset(0, jcp.ur_w);
            dim_t ur_w_ddst_offset = ur_w_b * get_ddst_offset(jcp.ur_w);

            for (int icb = 0; icb < nb_ic_blocking; icb++) {
                dim_t icb_offset = jcp.typesize_in * icb * jcp.tr_src_buf_size;
                tileloadd(Tmm(get_src_tensor(icb)),
                        ptr[reg_src + reg_a_stride + icb_offset
                                + ur_w_src_offset]);
            }
            for (int ocb = 0; ocb < nb_oc_blocking; ocb++) {
                tileloadd(Tmm(get_ddst_tensor(ocb)),
                        ptr[reg_ddst + reg_dense_stride
                                + jcp.typesize_in * ocb
                                        * jcp.tr_diff_dst_buf_size
                                + ur_w_ddst_offset]);
                for (int icb = 0; icb < nb_ic_blocking; icb++)
                    tdpbf16ps(Tmm(get_wei_tensor(ocb, icb)),
                            Tmm(get_src_tensor(icb)),
                            Tmm(get_ddst_tensor(ocb)));
            }
        }
    };

    auto emit_h_loop = [&]() {
        Label h_loop, skip_h_loop;
        mov(reg_j, 1);
        cmp(reg_j, reg_h);
        je(skip_h_loop, T_NEAR);
        L(h_loop);
        {
            emit_block();

            add(reg_src, get_src_offset(0, 0, 1));
            add(reg_ddst, get_ddst_offset(0, 1));
            add(reg_j, 1);
            cmp(reg_j, reg_h);
            jb(h_loop);
        }
        L(skip_h_loop);

        emit_block();
    };

    auto emit_kh_kw_loop = [&](bool is_first_block, bool is_last_block) {
        xor_(reg_kh, reg_kh);
        Label kh_loop, kh_loop_end;

        int oh_block_size = (is_last_block) ? h_last_block_size : h_block_size;
        // NB: this is correct because we only support t_pad = kh / 2 and thus
        // ih == oh
        int ih_block_size = oh_block_size
                + (!is_first_block + !is_last_block) * jcp.t_pad;

        L(kh_loop);
        {
            if (is_first_block) {
                xor_(reg_tmp, reg_tmp);
                mov(reg_ohs, jcp.t_pad);
                sub(reg_ohs, reg_kh);
                cmovb(reg_ohs, reg_tmp);

                mov(reg_ihs, reg_ohs);
                sub(reg_ihs, jcp.t_pad);
                add(reg_ihs, reg_kh);
            } else {
                xor_(reg_ohs, reg_ohs);
                mov(reg_ihs, reg_kh);
            }

            mov(reg_tmp, oh_block_size);
            sub(reg_tmp, reg_ohs);
            mov(reg_h, ih_block_size);
            sub(reg_h, reg_ihs);
            cmp(reg_tmp, reg_h);
            cmovb(reg_h, reg_tmp);

            Label kh_loop_work;
            cmp(reg_h, 0);
            jg(kh_loop_work, T_NEAR);

            // empty h loop for this jcp.kh:
            // - set the ddst to 0 if necessary
            // - move ker pt
            // - jump to the end
            sub(reg_h, 1);
            Label skip_ker_zeroing;

            // The reg_ker ptr has highest bit set if the ddst needs to be
            // zeroed. Those who have byte-aligned their data will suffer the
            // consequences :(
            // TODO: move the flag to a mask register? (Roma)
            test(reg_ker, 1);
            jz(skip_ker_zeroing, T_NEAR);

            Label zeroing_loop;
            vpxord(zmm0, zmm0, zmm0);
            and_(reg_ker, ~1); // temporarily clear the zeroing flag

            mov(reg_dense_stride, 64);
            tilezero(Tmm(get_wei_tensor(0, 0)));
            for (int kw = 0; kw < jcp.kw; kw++) {
                // dim_t kw_offset = kw * get_kernel_offset(jcp.ic_block, 0);
                for_(int ocb = 0; ocb < nb_oc_blocking; ocb++)
                for (int icb = 0; icb < nb_ic_blocking; icb++)
                    tilestored(
                            ptr[reg_ker + reg_dense_stride
                                    + get_full_kernel_offset(ocb, icb, 0, kw)],
                            Tmm(get_wei_tensor(0, 0)));
            }
            // restore the zeroing flag (it will be cleared after the end of
            // emit_kh_kw_loop, but we may need it until then)
            or_(reg_ker, 1);
            jmp(kh_loop_end, T_NEAR);

            L(skip_ker_zeroing);
            add(reg_ker, get_kernel_offset(0, jcp.kw));
            jmp(kh_loop_end, T_NEAR);

            L(kh_loop_work);

            mul_by_const(reg_ihs, reg_tmp, get_src_offset(0, 0, 1));
            mul_by_const(reg_ohs, reg_tmp, get_ddst_offset(0, 1));

            add(reg_src, reg_ihs);
            add(reg_ddst, reg_ohs);

            Label kw_loop;
            xor_(reg_kw, reg_kw);

            mov(reg_dense_stride, 64);
            L(kw_loop);
            {
                Label do_zero, ker_init_done;
                test(reg_ker, 1);
                jnz(do_zero, T_NEAR);

                for_(int ocb = 0; ocb < nb_oc_blocking; ocb++)
                for (int icb = 0; icb < nb_ic_blocking; icb++)
                    tileloadd(Tmm(get_wei_tensor(ocb, icb)),
                            ptr[reg_ker + reg_dense_stride
                                    + get_full_kernel_offset(ocb, icb, 0, 0)]);
                jmp(ker_init_done);
                L(do_zero);
                for_(int ocb = 0; ocb < nb_oc_blocking; ocb++)
                for (int icb = 0; icb < nb_ic_blocking; icb++)
                    tilezero(Tmm(get_wei_tensor(ocb, icb)));

                L(ker_init_done);

                mov(ptr[rsp + ddst_save_offset], reg_ddst);
                mov(ptr[rsp + src_save_offset], reg_src);

                lea(reg_src, ptr[reg_src + reg_kw * jcp.typesize_in]);
                emit_h_loop();

                mov(reg_ddst, ptr[rsp + ddst_save_offset]);
                mov(reg_src, ptr[rsp + src_save_offset]);

                // The reg_ker ptr has highest bit set if the ddst needs to
                // be zeroed. Those who have byte-aligned their data will
                // suffer the consiquences :(
                mov(reg_tmp, reg_ker);
                and_(reg_ker, ~1);

                for_(int ocb = 0; ocb < nb_oc_blocking; ocb++)
                for (int icb = 0; icb < nb_ic_blocking; icb++)
                    tilestored(
                            ptr[reg_ker + reg_dense_stride
                                    + get_full_kernel_offset(ocb, icb, 0, 0)],
                            Tmm(get_wei_tensor(ocb, icb)));

                mov(reg_ker, reg_tmp);
                add(reg_ker, get_kernel_offset(jcp.ic_block, 0));
                add(reg_kw, 1);
                cmp(reg_kw, jcp.kw);
                jl(kw_loop);
            }

            sub(reg_src, reg_ihs);
            sub(reg_ddst, reg_ohs);

            L(kh_loop_end);
            add(reg_kh, 1);
            cmp(reg_kh, jcp.kh);
            jl(kh_loop);
        }
    };

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_ddst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);
    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    or_(reg_ker, reg_tmp);

    bool single_kh_kw_loop = (h_last_block_size == jcp.oh);

    auto src_row_step = get_src_offset(0, 0, 1);
    auto first_src_block_step = src_row_step * (h_block_size - jcp.t_pad);
    auto ddst_block_step = get_ddst_offset(0, h_block_size);

    emit_kh_kw_loop(true, single_kh_kw_loop);

    if (!single_kh_kw_loop) {
        auto ker_reset_offset = get_kernel_offset(0, jcp.kw * jcp.kh);
        sub(reg_ker, ker_reset_offset);
        and_(reg_ker, ~1); // Clear the zeroing flag for subsequent updates

        add(reg_src, first_src_block_step);
        add(reg_ddst, ddst_block_step);

        int num_innermost_iters
                = (jcp.oh - h_last_block_size) / h_block_size - 1;
        if (num_innermost_iters > 0) {
            Label h_block_loop;

            mov(reg_tmp_w, num_innermost_iters);
            kmovw(reg_h_block, reg_tmp_w);
            L(h_block_loop);
            {
                emit_kh_kw_loop(false, false);
                sub(reg_ker, ker_reset_offset);
                add(reg_src, src_row_step * h_block_size);
                add(reg_ddst, ddst_block_step);

                kmovw(reg_tmp_w, reg_h_block);
                sub(reg_tmp_w, 1);
                kmovw(reg_h_block, reg_tmp_w);
                jnz(h_block_loop);
            }
        }

        emit_kh_kw_loop(false, true);
    }
}

void jit_avx512_core_amx_bwd_weights_kernel_t::compute_ic_loop(
        int ic_block, int nb_ic_blocking, int nb_oc_blocking) {
    assert(jcp.ur_w % 2 == 0);
    const int str_w = jcp.stride_w;
    assert(jcp.tr_iw % str_w == 0);
    const int src_stride_w_shift = jcp.tr_iw / str_w;

    mov(reg_b_stride, 64);
    mov(reg_a_stride, jcp.tr_iw * jcp.typesize_in);

    for (int s = 0; s < str_w; s++) {
        for (int i_kw = s; i_kw < jcp.kw; i_kw += str_w) {

            for (int ocb = 0; ocb < nb_oc_blocking; ocb++)
                for (int icb = 0; icb < nb_ic_blocking; icb++)
                    tileloadd(Tmm(get_wei_tensor(ocb, icb)),
                            ptr[reg_kernel + reg_b_stride
                                    + get_full_kernel_offset(
                                            ocb, icb, 0, i_kw)]);

            int src_offset_l = (i_kw * (jcp.dilate_w + 1)) / str_w
                    + s * src_stride_w_shift;

            for (int ur_w_b = 0; ur_w_b < jcp.ur_w_blocks; ur_w_b++) {
                dim_t ur_w_src_offset = ur_w_b
                        * get_src_offset(0, filter_w_to_src(0, jcp.ur_w, 0));
                dim_t ur_w_ddst_offset = ur_w_b * get_ddst_offset(jcp.ur_w);
                for (int icb = 0; icb < nb_ic_blocking; icb++) {
                    dim_t icb_offset = icb * jcp.tr_src_buf_size;
                    tileloadd(Tmm(get_src_tensor(icb)),
                            ptr[reg_src
                                    + jcp.typesize_in
                                            * (src_offset_l + icb_offset)
                                    + ur_w_src_offset + reg_a_stride]);
                }
                for (int ocb = 0; ocb < nb_oc_blocking; ocb++) {
                    tileloadd(Tmm(get_ddst_tensor(ocb)),
                            ptr[reg_ddst
                                    + jcp.typesize_in * ocb
                                            * jcp.tr_diff_dst_buf_size
                                    + ur_w_ddst_offset + reg_b_stride]);
                    for (int icb = 0; icb < nb_ic_blocking; icb++)
                        tdpbf16ps(Tmm(get_wei_tensor(ocb, icb)),
                                Tmm(get_src_tensor(icb)),
                                Tmm(get_ddst_tensor(ocb)));
                }
            }

            for (int ocb = 0; ocb < nb_oc_blocking; ocb++)
                for (int icb = 0; icb < nb_ic_blocking; icb++)
                    tilestored(ptr[reg_kernel + reg_b_stride
                                       + get_full_kernel_offset(
                                               ocb, icb, 0, i_kw)],
                            Tmm(get_wei_tensor(ocb, icb)));
        }
    }
    safe_add(reg_src, get_src_offset(ic_block, 0), reg_long_offt);
    add(reg_kernel, get_kernel_offset(ic_block, 0));
}

void jit_avx512_core_amx_bwd_weights_kernel_t::compute_diff_bias_init(int ocb) {
    auto reg_unit_val = reg_tmp.cvt16();
    mov(reg_unit_val, 0x3f80); // bf16 value of 1.
    vpbroadcastw(vreg_bias_unit, reg_unit_val);

    mov(reg_tmp, ptr[param + GET_OFF(bias)]);
    vmovups(vreg_bias_acc, ptr[reg_tmp + sizeof(float) * ocb * jcp.oc_block]);
}

void jit_avx512_core_amx_bwd_weights_kernel_t::compute_diff_bias_row(
        bool is_partial, int ocb) {
    if (!jcp.with_bias) return;
    mov(reg_tmp, ptr[param + GET_OFF(flags)]);
    Label skip_label;
    test(reg_tmp, FLAG_IC_FIRST);
    jz(skip_label, T_NEAR);

    if (is_partial) { compute_diff_bias_init(ocb); }
    auto compute_step = [&]() {
        vmovups(vreg_bias_ddst, ptr[reg_ddst]);
        vdpbf16ps(vreg_bias_acc, vreg_bias_ddst, vreg_bias_unit);
    };

    Label ow_loop, ow_tail;
    int niters = jcp.tr_ow / 2;
    if (niters > 0) {
        mov(reg_tmp, jcp.tr_ow / 2);
        L(ow_loop);
        compute_step();
        add(reg_ddst, get_ddst_offset(2));
        sub(reg_tmp, 1);
        jnz(ow_loop, T_NEAR);
    }
    if (jcp.tr_ow % 2) compute_step();

    if (niters > 0) sub(reg_ddst, get_ddst_offset(2 * niters));

    if (is_partial) {
        mov(reg_tmp, ptr[param + GET_OFF(bias)]);
        vmovups(ptr[reg_tmp + sizeof(float) * ocb * jcp.oc_block],
                vreg_bias_acc);
    }

    L(skip_label);
}

void jit_avx512_core_amx_bwd_weights_kernel_t::maybe_compute_diff_bias(
        int nb_oc_blocking) {
    // In harness_3d_reduction case calculation of diff_bias is called
    // for every ow row separately to be aligned with od loop in
    // compute_od_loop_common()
    if (!jcp.with_bias || jcp.harness == harness_3d_reduction) return;
    mov(reg_tmp, ptr[param + GET_OFF(flags)]);

    Label skip_label;
    test(reg_tmp, FLAG_IC_FIRST);
    jz(skip_label, T_NEAR);

    for (int ocb = 0; ocb < nb_oc_blocking; ocb++) {
        Label bias_loop, skip_label_local;

        mov(reg_ddst, ptr[param + GET_OFF(dst)]);
        add(reg_ddst, jcp.typesize_in * ocb * jcp.tr_diff_dst_buf_size);

        switch (jcp.harness) {
            case harness_2d_reduction:
                mov(reg_oj, ptr[param + GET_OFF(os_index_end)]);
                sub(reg_oj, ptr[param + GET_OFF(os_index_begin)]);
                break;
            case harness_mb_reduction:
            case harness_compute_full_spatial: mov(reg_oj, jcp.oh); break;
            case harness_3d_reduction:
            default: assert(!"Invalid harness type");
        }

        cmp(reg_oj, 0);
        jle(skip_label_local, T_NEAR); // nothing to do

        compute_diff_bias_init(ocb);
        L(bias_loop);
        {
            compute_diff_bias_row(false, ocb);
            add(reg_ddst, get_ddst_offset(0, 1));

            sub(reg_oj, 1);
            jnz(bias_loop, T_NEAR);
        }

        mov(reg_tmp, ptr[param + GET_OFF(bias)]);
        vmovups(ptr[reg_tmp + sizeof(float) * ocb * jcp.oc_block],
                vreg_bias_acc);

        L(skip_label_local);
    }
    // restore reg_ddst value
    mov(reg_ddst, ptr[param + GET_OFF(dst)]);

    L(skip_label);
}

void jit_avx512_core_amx_bwd_weights_kernel_t::compute_oh_step_common(
        int nb_ic_blocking, int nb_oc_blocking) {
    Label kh_label, ic_block_label, ow_block_label, kd_label;

    int ic_block = jcp.ic_block;
    int ic_tail = jcp.ic_tail;

    auto ic_loop = [&](int nb_ic_blocking, int nb_oc_blocking) {
        Label ic_tail_label, ic_loop_done_label;

        if (ic_tail) {
            mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
            cmp(reg_icb, jcp.ic_tail);
            jne(ic_tail_label, T_NEAR);

            compute_ic_loop(ic_block, nb_ic_blocking, nb_oc_blocking);
            jmp(ic_loop_done_label, T_NEAR);

            L(ic_tail_label);
            compute_ic_loop(ic_tail, nb_ic_blocking, nb_oc_blocking);
            add(reg_kernel, get_kernel_offset(jcp.ic_block - ic_tail, 0));
            safe_add(reg_src,
                    get_src_offset(0, 0, filter_h_to_src(1))
                            - get_src_offset(ic_tail, 0),
                    reg_long_offt);
            L(ic_loop_done_label);
        } else {
            compute_ic_loop(ic_block, nb_ic_blocking, nb_oc_blocking);
        }
    };

    if (jcp.ndims == 5) {
        /* NOTE: reg_kd_count = aux_reg_src = r12. The following order of
         * 'movs' must be guaranteed. */
        mov(ki, reg_kd_count);
        mov(EVEX_compress_addr(rsp, kd_count_offset), reg_kd_count);
        mov(aux_reg_src, reg_src);
        mov(aux_reg_kernel, reg_kernel);

        L(kd_label);
        mov(reg_src, aux_reg_src);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        ic_loop(nb_ic_blocking, nb_oc_blocking);

        if (jcp.dilate_h > 0) {
            add(reg_src, get_src_offset(0, 0, jcp.dilate_h));
        }
        // substract pointer shift made within ic block loop
        // and move to next kh index
        add(reg_kernel, get_kernel_offset(-ic_block, jcp.kw));
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_src, get_src_offset(0, 0, filter_d_to_src(1)));
        add(aux_reg_kernel, get_kernel_offset(0, jcp.kh * jcp.kw));
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
    // In harness_3d_reduction case calculation of diff_bias is called
    // for every ow row separately to be aligned with od loop in
    // compute_od_loop_common()
    if (jcp.harness == harness_3d_reduction) {
        auto reg_save_ddst = reg_a_stride;
        mov(reg_save_ddst, reg_ddst);
        for (int ocb = 0; ocb < nb_oc_blocking; ocb++) {
            safe_add(reg_ddst, jcp.typesize_in * ocb * jcp.tr_diff_dst_buf_size,
                    reg_long_offt);
            compute_diff_bias_row(true, ocb);
        }
        mov(reg_ddst, reg_save_ddst);
    }

    if (jcp.ndims == 5) {
        mov(reg_src, aux_reg_src);
        mov(reg_kernel, aux_reg_kernel);
        mov(reg_kd_count, EVEX_compress_addr(rsp, kd_count_offset));
        od_step_comeback_pointers();
    } else {
        oh_step_comeback_pointers();
    }
}

void jit_avx512_core_amx_bwd_weights_kernel_t::maybe_zero_kernel(
        int nb_ic_blocking, int nb_oc_blocking) {
    if (jcp.harness == harness_compute_full_spatial && !jcp.with_bias) return;
    Label skip_zeroing, zeroing_loop;

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jz(skip_zeroing, T_NEAR);

    Zmm zero = Zmm(0);
    vpxord(zero, zero, zero);
    if (jcp.with_bias) {
        Label skip_bias_zeroing;
        mov(reg_tmp, ptr[param + GET_OFF(flags)]);
        test(reg_tmp, FLAG_IC_FIRST);
        jz(skip_bias_zeroing, T_NEAR);
        for (int ocb = 0; ocb < nb_oc_blocking; ocb++) {
            mov(reg_tmp, ptr[param + GET_OFF(bias)]);
            vmovups(ptr[reg_tmp + sizeof(float) * ocb * jcp.oc_block], zero);
        }
        L(skip_bias_zeroing);
        if (jcp.harness == harness_compute_full_spatial)
            jmp(skip_zeroing, T_NEAR);
    }

    mov(reg_b_stride, 64);
    tilezero(Tmm(get_wei_tensor(0, 0)));
    for (dim_t shift = 0;
            shift < get_kernel_offset(0, jcp.kw * jcp.kh * jcp.kd);
            shift += get_kernel_offset(jcp.ic_block, 0)) {
        for_(int icb = 0; icb < nb_ic_blocking; icb++)
        for (int ocb = 0; ocb < nb_oc_blocking; ocb++) {
            tilestored(
                    ptr[reg_kernel + reg_b_stride
                            + get_full_kernel_offset(ocb, icb, 0, 0) + shift],
                    Tmm(get_wei_tensor(0, 0)));
        }
    }
    L(skip_zeroing);
}

void jit_avx512_core_amx_bwd_weights_kernel_t::compute_oh_loop_common(
        int nb_ic_blocking, int nb_oc_blocking, bool is_partial) {
    int b_pad = jcp.b_pad;
    int t_pad = jcp.t_pad;

    bool is_dilated = jcp.dilate_h != 0;
    int dilate_h = jcp.dilate_h + 1;
    int stride_h = jcp.stride_h;
    auto filter_step_size = get_kernel_offset(0, jcp.kw);
    auto src_step_size = get_src_offset(0, 0, 1);
    auto ddst_step_size = get_ddst_offset(0, 1);
    Label oh_label, oh_label_end, oh_tpad_label, oh_tpad_label_end,
            oh_tpad_tail_label, oh_tpad_tail_label_end, oh_bpad_label,
            oh_bpad_label_end, oh_dilate_label_shift, oh_dilate_label_noshift,
            oh_dilate_label_end, oh_dilate_setup_label_shift,
            oh_dilate_setup_label_noshift, oh_dilate_setup_label_end;

    int ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    int oh_body_end = div_up(t_pad + jcp.ih - ext_kh + 1, stride_h);
    int oh_head_end = nstl::min(div_up(t_pad, stride_h), oh_body_end);
    int oh_head_overflow_end = div_up(t_pad, stride_h);
    int oh_tail_end = jcp.oh;

    int body_src_start_offset = (stride_h - (t_pad % stride_h)) % stride_h;
    int ih_body_end
            = nstl::max(-t_pad + oh_body_end * stride_h, body_src_start_offset);

    if (is_partial)
        mov(reg_oj, ptr[param + GET_OFF(os_index_begin)]);
    else
        xor_(reg_oj, reg_oj);

    /* Compute 'top' edge */
    if (t_pad > 0) {
        if (is_partial) {
            cmp(reg_oj, oh_head_overflow_end);
            jge(oh_tpad_tail_label_end, T_NEAR);
        }
        const int overflow
                = nstl::max(0, jcp.kh - div_up(t_pad + jcp.ih, dilate_h));
        const int underflow = div_up(t_pad, dilate_h);
        const int initial_kh = jcp.kh - overflow - underflow;

        // Setup reg_kh, reg_kernel, and reg_src
        mov(reg_kh, initial_kh);
        add(reg_kernel, filter_step_size * underflow);
        if (is_dilated) {
            const int tail = t_pad % dilate_h;
            const int shift = tail == 0 ? 0 : dilate_h - tail;
            mov(reg_ih_shift, shift);
            if (!is_partial) mov(ptr[rsp + ih_dilate_offset], reg_ih_shift);
            add(reg_src, src_step_size * shift);
        }

        if (is_partial) {
            Label head_setup, head_setup_finish;
            cmp(reg_oj, 0);
            je(head_setup_finish, T_NEAR);
            mov(reg_oj_setup, reg_oj);

            L(head_setup);
            if (is_dilated) {
                inc(reg_ih_shift);
                cmp(reg_ih_shift, dilate_h);
                jl(oh_dilate_setup_label_shift, T_NEAR);
                // unshift src as new kernel element enters
                sub(reg_src, src_step_size * (dilate_h - 1));
                xor_(reg_ih_shift, reg_ih_shift);
            }
            // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
            add(reg_kh, stride_h);
            sub(reg_kernel, filter_step_size * stride_h);
            if (is_dilated) {
                jmp(oh_dilate_setup_label_noshift, T_NEAR);
                L(oh_dilate_setup_label_shift);
                // shift src as old kernel element progresses
                add(reg_src, src_step_size * stride_h);
                L(oh_dilate_setup_label_noshift);
            }
            sub(reg_oj_setup, 1);
            jg(head_setup, T_NEAR);
            L(head_setup_finish);

            if (is_dilated) mov(ptr[rsp + ih_dilate_offset], reg_ih_shift);
            if (oh_head_end < oh_head_overflow_end) {
                cmp(reg_oj, oh_head_end);
                jge(oh_tpad_label_end, T_NEAR);
            }
        }

        //Setup reg_kernel
        // If dilated, shift src ptr
        // Loop
        L(oh_tpad_label);
        compute_oh_step_common(nb_ic_blocking, nb_oc_blocking);
        add(reg_ddst, ddst_step_size);
        if (is_dilated) {
            mov(reg_ih_shift, ptr[rsp + ih_dilate_offset]);
            inc(reg_ih_shift);
            mov(ptr[rsp + ih_dilate_offset], reg_ih_shift);
            cmp(reg_ih_shift, dilate_h);
            jl(oh_dilate_label_shift, T_NEAR);
            // unshift src as new kernel element enters
            sub(reg_src, src_step_size * (dilate_h - 1));
            xor_(reg_ih_shift, reg_ih_shift);
            mov(ptr[rsp + ih_dilate_offset], reg_ih_shift);
        }
        // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
        add(reg_kh, stride_h);
        sub(reg_kernel, filter_step_size * stride_h);
        if (is_dilated) {
            jmp(oh_dilate_label_noshift, T_NEAR);
            L(oh_dilate_label_shift);
            // shift src as old kernel element progresses
            add(reg_src, src_step_size * stride_h);
            L(oh_dilate_label_noshift);
        }
        inc(reg_oj);

        if (is_partial) {
            cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
            jge(oh_bpad_label_end, T_NEAR);
        }
        cmp(reg_oj, oh_head_end);
        jl(oh_tpad_label, T_NEAR);

        L(oh_tpad_label_end);
        // need second loop to process kernel if it is larger than the src
        // (does not apply to dilations as they must have unit stride)
        if (oh_head_end < oh_head_overflow_end) {
            assert(!is_dilated);

            cmp(reg_oj, oh_head_overflow_end);
            jge(oh_tpad_tail_label_end, T_NEAR);

            mov(reg_kh, jcp.ih);
            L(oh_tpad_tail_label);
            {
                compute_oh_step_common(nb_ic_blocking, nb_oc_blocking);
                add(reg_ddst, ddst_step_size);
                sub(reg_kernel, filter_step_size * stride_h);

                inc(reg_oj);

                if (is_partial) {
                    cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
                    jge(oh_bpad_label_end, T_NEAR);
                }
                cmp(reg_oj, oh_head_overflow_end);
                jl(oh_tpad_tail_label, T_NEAR);
            }
        }
        if (body_src_start_offset != 0) {
            add(reg_kernel, filter_step_size * body_src_start_offset);
            add(reg_src, src_step_size * body_src_start_offset);
        }
        L(oh_tpad_tail_label_end);
    }

    if (is_partial) {
        cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
        jge(oh_bpad_label_end, T_NEAR);
    }
    cmp(reg_oj, oh_body_end);
    jge(oh_label_end, T_NEAR);

    /* Compute middle block(s) */
    mov(reg_kh, jcp.kh);
    L(oh_label);
    {
        compute_oh_step_common(nb_ic_blocking, nb_oc_blocking);
        add(reg_src, src_step_size * stride_h);
        add(reg_ddst, ddst_step_size);

        inc(reg_oj);

        if (is_partial) {
            cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
            jge(oh_bpad_label_end, T_NEAR);
        }

        cmp(reg_oj, oh_body_end);
        jl(oh_label, T_NEAR);
    }
    L(oh_label_end);

    /* Compute bottom edge */
    if (b_pad > 0) {
        if (is_partial) {
            cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
            jge(oh_bpad_label_end, T_NEAR);
        }
        cmp(reg_oj, jcp.oh);
        jge(oh_bpad_label_end, T_NEAR);

        if (is_dilated) {
            // Assumes unit stride for dilations
            mov(reg_kh, jcp.kh - 1);
            xor_(reg_ih_shift, reg_ih_shift);
        } else {
            assert(jcp.dilate_h == 0);
            mov(reg_kh, jcp.ih - ih_body_end);
        }
        if (is_partial) {
            lea(reg_oj_setup,
                    ptr[reg_oj - nstl::max(oh_body_end, oh_head_overflow_end)]);
            if (stride_h == 1 && !is_dilated) {
                sub(reg_kh, reg_oj_setup);
            } else {
                Label body_setup, body_setup_finish, dilate_skip;
                cmp(reg_oj_setup, 0);
                je(body_setup_finish, T_NEAR);

                L(body_setup);
                if (is_dilated) {
                    inc(reg_ih_shift);
                    cmp(reg_ih_shift, dilate_h);
                    jl(dilate_skip, T_NEAR);
                    xor_(reg_ih_shift, reg_ih_shift);
                }
                sub(reg_kh, stride_h);
                L(dilate_skip);
                sub(reg_oj_setup, 1);
                jg(body_setup, T_NEAR);
                L(body_setup_finish);
            }
        }

        if (is_dilated) mov(ptr[rsp + ih_dilate_offset], reg_ih_shift);
        L(oh_bpad_label);
        {
            compute_oh_step_common(nb_ic_blocking, nb_oc_blocking);
            add(reg_src, src_step_size * stride_h);
            add(reg_ddst, ddst_step_size);

            if (is_dilated) {
                mov(reg_ih_shift, ptr[rsp + ih_dilate_offset]);
                inc(reg_ih_shift);
                mov(ptr[rsp + ih_dilate_offset], reg_ih_shift);
                cmp(reg_ih_shift, dilate_h);
                jl(oh_dilate_label_end, T_NEAR);
                xor_(reg_ih_shift, reg_ih_shift);
                mov(ptr[rsp + ih_dilate_offset], reg_ih_shift);
            }
            sub(reg_kh, stride_h);
            L(oh_dilate_label_end);
            inc(reg_oj);
            if (is_partial) {
                cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
                jge(oh_bpad_label_end, T_NEAR);
            }
            cmp(reg_oj, oh_tail_end);
            jl(oh_bpad_label, T_NEAR);
        }
    }
    L(oh_bpad_label_end);
}

void jit_avx512_core_amx_bwd_weights_kernel_t::compute_od_loop_common(
        int nb_ic_blocking, int nb_oc_blocking, bool is_partial) {
    assert(jcp.harness == harness_3d_reduction);

    const int src_backpad_overlap
            = div_up(jcp.id + jcp.f_pad - (jcp.kd - 1), jcp.stride_d);

    const auto filter_shift = get_kernel_offset(0, jcp.kh * jcp.kw);
    const auto src_shift = get_src_offset(0, 0, jcp.ih);
    const auto ddst_shift = get_ddst_offset(0, jcp.oh);

    const int kd_front_pad = nstl::max(0, jcp.f_pad);
    const int kd_back_pad = nstl::max(0, jcp.kd - jcp.f_pad - jcp.id);

    Label d_loop_label, loop_end_label, common_block_label, fpad_end_label,
            backpad_end_label, backpad_label;

    /* initially offset 'kd' by f_pad */
    mov(reg_src_d, ptr[param + GET_OFF(src)]);
    mov(reg_ddst_d, ptr[param + GET_OFF(dst)]);

    if (is_partial) {
        add(reg_kernel, ptr[param + GET_OFF(kd_offset)]);
        mov(reg_d_index, ptr[param + GET_OFF(os_index_begin)]);
        mov(reg_kd_count, ptr[param + GET_OFF(kd_padding)]);
    } else {
        const int kd_padding = jcp.kd - kd_front_pad - kd_back_pad;
        const int kd_offset = get_kernel_offset(
                0, nstl::min(jcp.kd - 1, kd_front_pad) * jcp.kh * jcp.kw);
        add(reg_kernel, kd_offset);
        xor_(reg_d_index, reg_d_index);
        mov(reg_kd_count, kd_padding);
    }

    cmp(reg_kd_count, 0);
    jle(loop_end_label, T_NEAR); // no iterations along kd
    if (is_partial)
        cmp(reg_d_index, ptr[param + GET_OFF(os_index_end)]);
    else
        cmp(reg_d_index, jcp.od);
    jge(loop_end_label, T_NEAR); // no iterations along depth dimension

    L(d_loop_label);

    mov(reg_src, reg_src_d);
    mov(reg_ddst, reg_ddst_d);

    mov(EVEX_compress_addr(rsp, src_d_offset), reg_src_d);
    mov(EVEX_compress_addr(rsp, ddst_d_offset), reg_ddst_d);
    mov(EVEX_compress_addr(rsp, d_index_offset), reg_d_index);

    compute_oh_loop_common(nb_ic_blocking, nb_oc_blocking);

    mov(reg_src_d, EVEX_compress_addr(rsp, src_d_offset));
    mov(reg_ddst_d, EVEX_compress_addr(rsp, ddst_d_offset));
    mov(reg_d_index, EVEX_compress_addr(rsp, d_index_offset));

    /* Compute 'front' edge */
    if (jcp.f_pad > 0) {
        /* Check if within fpad region */
        cmp(reg_d_index, div_up(jcp.f_pad, jcp.stride_d));
        jge(fpad_end_label, T_NEAR);

        /* Fpad steps */
        sub(reg_kernel, filter_shift * jcp.stride_d);
        add(reg_kd_count, jcp.stride_d);

        /* Final number of kernel elements that overlap with src */
        const int src_ker_overlap = nstl::min(jcp.kd, jcp.id);
        cmp(reg_kd_count, src_ker_overlap);
        jle(common_block_label, T_NEAR);

        /* Correct any excess shifts to kernel and src */
        if (jcp.f_pad <= jcp.od * jcp.stride_d) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.f_pad % jcp.stride_d != 0) {
                int src_corr = jcp.stride_d - jcp.f_pad % jcp.stride_d;
                add(reg_kernel, filter_shift * src_corr);
                add(reg_src_d, src_shift * src_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub(reg_kernel, (jcp.f_pad - jcp.od * jcp.stride_d) * filter_shift);
        }

        /* Apply correction */
        mov(reg_kd_count, src_ker_overlap);
        jmp(common_block_label);

        L(fpad_end_label);
    }

    /* Compute bottom edge */
    if (jcp.back_pad > 0) {

        /* Check if within back_pad region */
        cmp(reg_d_index, src_backpad_overlap - 1);
        jl(backpad_end_label, T_NEAR);
        jg(backpad_label, T_NEAR);

        /* Execute overlap correction between the filter and the initial
         * back_pad region. */
        mov(reg_kd_count,
                jcp.id + jcp.f_pad - src_backpad_overlap * jcp.stride_d);
        jmp(backpad_end_label, T_NEAR);

        L(backpad_label);
        sub(reg_kd_count, jcp.stride_d);
        cmp(reg_kd_count, 0);
        jle(loop_end_label, T_NEAR);

        L(backpad_end_label);
    }

    /* Compute middle block */
    add(reg_src_d, src_shift * jcp.stride_d);

    /* Execute common block and loop */
    L(common_block_label);
    add(reg_ddst_d, ddst_shift);
    inc(reg_d_index);
    if (is_partial)
        cmp(reg_d_index, ptr[param + GET_OFF(os_index_end)]);
    else
        cmp(reg_d_index, jcp.od);
    jl(d_loop_label, T_NEAR);

    L(loop_end_label);
}

void jit_avx512_core_amx_bwd_weights_kernel_t::compute_loop(
        int nb_ic_blocking, int nb_oc_blocking) {
    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_ddst, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    maybe_zero_kernel(nb_ic_blocking, nb_oc_blocking);
    maybe_compute_diff_bias(nb_oc_blocking);

    switch (jcp.harness) {
        case harness_3d_reduction:
            compute_od_loop_common(nb_ic_blocking, nb_oc_blocking, true);
            break;
        case harness_2d_reduction:
            compute_oh_loop_common(nb_ic_blocking, nb_oc_blocking, true);
            break;
        case harness_mb_reduction:
            compute_oh_loop_common(nb_ic_blocking, nb_oc_blocking);
            break;
        case harness_compute_full_spatial:
            compute_full_spat_loop(nb_ic_blocking, nb_oc_blocking);
            break;
        default: assert(!"Invalid harness type");
    }
}

void jit_avx512_core_amx_bwd_weights_kernel_t::setup_stack_space() {
    kd_count_offset = ic_block_step_stack_size;
    src_d_offset = ic_block_step_stack_size + 8;
    ddst_d_offset = ic_block_step_stack_size + 16;
    d_index_offset = ic_block_step_stack_size + 24;
    ih_dilate_offset = ic_block_step_stack_size + 32;
    src_save_offset = ic_block_step_stack_size + 40;
    ddst_save_offset = ic_block_step_stack_size + 48;
    stack_space_needed = ic_block_step_stack_size + 56;
}

void jit_avx512_core_amx_bwd_weights_kernel_t::generate() {
    preamble();

    setup_stack_space();

    sub(rsp, stack_space_needed);

    Label last_ic_block_label, last_blocks_done_label;

    mov(reg_tmp, ptr[param + GET_OFF(last_ic_block)]);
    cmp(reg_tmp, 0);
    jne(last_ic_block_label, T_NEAR);
    { // full nb_ic_blocking
        Label last_oc_block_label;
        mov(reg_tmp, ptr[param + GET_OFF(last_oc_block)]);
        cmp(reg_tmp, 0);
        jne(last_oc_block_label, T_NEAR);
        { // full nb_oc_blocking
            compute_loop(jcp.nb_ic_blocking, jcp.nb_oc_blocking);
            jmp(last_blocks_done_label, T_NEAR);
        }
        L(last_oc_block_label);
        { // tail of nb_oc_blocking
            compute_loop(jcp.nb_ic_blocking, 1);
            jmp(last_blocks_done_label, T_NEAR);
        }
    }
    L(last_ic_block_label);
    { // tail nb_ic_blocking
        Label last_oc_block_label;
        mov(reg_tmp, ptr[param + GET_OFF(last_oc_block)]);
        cmp(reg_tmp, 0);
        jne(last_oc_block_label, T_NEAR);
        { // full nb_oc_blocking
            compute_loop(1, jcp.nb_oc_blocking);
            jmp(last_blocks_done_label, T_NEAR);
        }
        L(last_oc_block_label);
        { // tail of nb_oc_blocking
            compute_loop(1, 1);
            jmp(last_blocks_done_label, T_NEAR);
        }
    }

    L(last_blocks_done_label);
    add(rsp, stack_space_needed);

    postamble();
}

status_t jit_avx512_core_amx_bwd_weights_kernel_t::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &diff_weights_md,
        memory_desc_t &diff_bias_md, memory_desc_t &diff_dst_md, int nthreads) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    const memory_desc_wrapper diff_bias_d(&diff_bias_md);

    jcp = zero<decltype(jcp)>();

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    if (!mayiuse(avx512_core_bf16_amx_bf16)) return status::unimplemented;
    jcp.isa = avx512_core_bf16_amx_bf16;

    jcp.ver = ver_vnni; // Needed for transpose routines
    jcp.nthr = nthreads;

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

    ok = true && one_of(ndims, 3, 4, 5)
            && everyone_is(
                    data_type::bf16, src_d.data_type(), diff_dst_d.data_type())
            && one_of(diff_weights_d.data_type(), data_type::f32,
                    data_type::bf16);
    if (!ok) return status::unimplemented;

    jcp.transform_to_vnni = diff_weights_d.data_type() == data_type::bf16;

    jcp.r_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw));
    jcp.b_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh));
    jcp.back_pad = nstl::max(0,
            calculate_end_padding(
                    jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd));

    /* XXX: no support for padding when dilation_d > 0 */
    if (!IMPLICATION(jcp.dilate_d > 0, everyone_is(0, jcp.back_pad, jcp.f_pad)))
        return status::unimplemented;

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;

    const int dat_format_tag = ndims - 3;
    format_tag_t dat_tag_nspc = utils::pick(dat_format_tag, format_tag::nwc,
            format_tag::nhwc, format_tag::ndhwc);
    format_tag_t dat_tag_opt = dat_tag_nspc;

    if (src_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(src_md, dat_tag_opt));
        jcp.src_tag = dat_tag_opt;
    } else
        jcp.src_tag = src_d.matches_one_of_tag(dat_tag_opt);
    if (!one_of(jcp.src_tag, dat_tag_opt)) return status::unimplemented;
    jcp.is_nspc = jcp.src_tag == dat_tag_nspc;

    if (diff_dst_d.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, jcp.src_tag));
        jcp.dst_tag = jcp.src_tag;
    } else
        jcp.dst_tag = diff_dst_d.matches_one_of_tag(jcp.src_tag);
    if (jcp.dst_tag != jcp.src_tag) return status::unimplemented;

    if (!jcp.is_nspc) return status::unimplemented;

    const int wei_format_tag = 2 * ndims - 6 + with_groups;
    format_tag_t wei_tag;
    if (jcp.transform_to_vnni)
        wei_tag = pick(wei_format_tag, format_tag::OIw16i16o2i,
                format_tag::gOIw16i16o2i, format_tag::OIhw16i16o2i,
                format_tag::gOIhw16i16o2i, format_tag::OIdhw16i16o2i,
                format_tag::gOIdhw16i16o2i);
    else
        wei_tag = pick(wei_format_tag, format_tag::OIw16i16o,
                format_tag::gOIw16i16o, format_tag::OIhw16i16o,
                format_tag::gOIhw16i16o, format_tag::OIdhw16i16o,
                format_tag::gOIdhw16i16o);
    if (diff_weights_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_md, wei_tag));
        jcp.wei_tag = wei_tag;
    } else {
        jcp.wei_tag = diff_weights_d.matches_one_of_tag(wei_tag);
        if (jcp.wei_tag != wei_tag) return status::unimplemented;
    }
    jcp.wei_dt = diff_weights_d.data_type();

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_bias_md, format_tag::x));
    }
    jcp.bia_dt = jcp.with_bias ? diff_bias_d.data_type() : data_type::undef;
    jcp.typesize_bia = jcp.with_bias ? types::data_type_size(jcp.bia_dt) : 0;

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const int max_pad_h = ext_kh / 2;
    const bool boundaries_ok = true && jcp.l_pad < ext_kw && jcp.r_pad < ext_kw
            && jcp.t_pad <= max_pad_h && jcp.b_pad <= max_pad_h
            && jcp.f_pad < ext_kd && jcp.back_pad < ext_kd;
    if (!boundaries_ok) return status::unimplemented;

    jcp.ic_block = 16;
    jcp.oc_block = 16;

    jcp.nb_ic = utils::div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = utils::div_up(jcp.oc, jcp.oc_block);

    jcp.ic_tail = jcp.ic % jcp.ic_block;
    jcp.oc_tail = jcp.oc % jcp.oc_block;

    jcp.nb_oc_blocking = (jcp.nb_oc > 1) ? 2 : 1;
    jcp.nb_ic_blocking = (jcp.nb_ic > 1) ? 2 : 1;

    int max_palette = amx::get_max_palette();
    jcp.max_tiles = amx::get_max_tiles(max_palette);
    jcp.full_tile_width = amx::get_max_rows(max_palette);

    if (jcp.max_tiles != 8 || jcp.full_tile_width != 16)
        return status::unimplemented;

    const bool is_2d = (ndims == 4);
    const bool is_3d = (ndims == 5);
    jcp.typesize_in = sizeof(bfloat16_t);
    jcp.typesize_out = sizeof(float);

    // TODO: Find more shapes (especially 3D with large spatials) for which
    // local transposition will be beneficial. Furthermore, for TBB threads
    // more shapes can potentially benefit from spatial blocking
    int optimal_blk_size = is_3d ? jcp.od : is_2d ? jcp.oh : jcp.ow;

    jcp.global_transpose = dnnl_thr_syncable();
    jcp.spatial_blk_size = optimal_blk_size;

    const int tr_round = 32; // To load full tile register
    int tr_pad = rnd_up(nstl::max(jcp.l_pad, jcp.r_pad + 1), tr_round);
    jcp.tr_iw = rnd_up(div_up(jcp.iw, jcp.stride_w) + tr_pad, tr_round)
            * jcp.stride_w;

    jcp.tr_src_num_guard_elems = tr_pad; // upper bound
    jcp.tr_ow = rnd_up(jcp.ow, 2);

    if (jcp.tr_ow <= max_ur_w) {
        jcp.ur_w = jcp.tr_ow;
        jcp.ur_w_blocks = 1;
    } else {
        jcp.ur_w = 1;
        for (int i = max_ur_w; i >= 1; i -= 2) {
            if (jcp.tr_ow % i == 0) {
                jcp.ur_w = i;
                break;
            }
        }
        jcp.ur_w_blocks = jcp.tr_ow / jcp.ur_w;
    }

    bool args_ok = true && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= diff_weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= diff_weights_d.padded_dims()[with_groups + 0];
    if (!args_ok) return status::unimplemented;

    bool use_full_spat_loop = jcp.ndims < 5 && jcp.ih == jcp.oh
            && jcp.iw == jcp.ow && everyone_is(1, jcp.stride_h, jcp.stride_w)
            && everyone_is(0, jcp.dilate_h, jcp.dilate_w)
            && jcp.l_pad == jcp.kw / 2 && jcp.t_pad == jcp.kh / 2;

    jcp.harness = ndims == 5
            ? harness_3d_reduction
            : (use_full_spat_loop ? harness_compute_full_spatial
                                  : (ndims == 4) ? harness_2d_reduction
                                                 : harness_mb_reduction);
    switch (jcp.harness) {
        case harness_2d_reduction: jcp.nthr_mb_work = jcp.mb * jcp.oh; break;
        case harness_3d_reduction: jcp.nthr_mb_work = jcp.mb * jcp.od; break;
        case harness_compute_full_spatial:
        case harness_mb_reduction: jcp.nthr_mb_work = jcp.mb; break;
        default: assert(!"Invalid harness"); jcp.nthr_mb_work = jcp.mb;
    }
    { // balancing
        int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
        balance(jcp, nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b);
        jcp.nthr = nthr;
        jcp.nthr_mb = nthr_mb;
        jcp.nthr_g = nthr_g;
        jcp.nthr_oc_b = nthr_oc_b;
        jcp.nthr_ic_b = nthr_ic_b;

        // TODO: Optimize memory allocation when threaded on height and depth
        jcp.tr_src_buf_size = jcp.tr_iw * jcp.ic_block * jcp.ih * jcp.id;
        jcp.tr_src_buf_count = jcp.global_transpose
                ? jcp.nthr_mb * jcp.nb_ic * jcp.ngroups
                : jcp.nthr;

        jcp.tr_diff_dst_buf_size = jcp.tr_ow * jcp.oc_block * jcp.oh * jcp.od;
        jcp.tr_diff_dst_buf_count = jcp.global_transpose
                ? jcp.nthr_mb * jcp.nb_oc * jcp.ngroups
                : jcp.nthr;
    }

    return status::success;
}

status_t jit_avx512_core_amx_bwd_weights_kernel_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp,
        memory_desc_t &src_md, memory_desc_t &diff_weights_md,
        memory_desc_t &diff_dst_md) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);

    // XXX: See the comment about tr_iw and guarding elements in
    // jit_avx512_core_amx_bwd_weights_kernel_t::init_conf()
    const size_t tr_src_size
            = (jcp.tr_src_buf_count * jcp.tr_src_buf_size * jcp.nb_ic_blocking)
            + jcp.tr_src_num_guard_elems;
    scratchpad.book(key_conv_tr_src, tr_src_size, jcp.typesize_in);

    /* prepare synchronization contexts */
    if (jcp.global_transpose && jcp.nthr_oc_b > 1) {
        const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_tr_src_bctx, tr_src_bctx_size);
    }

    const size_t tr_diff_dst_size = jcp.tr_diff_dst_buf_count
            * jcp.tr_diff_dst_buf_size * jcp.nb_oc_blocking;

    const size_t min_align = 64;
    scratchpad.book(
            key_conv_tr_diff_dst, tr_diff_dst_size, jcp.typesize_in, min_align);

    /* prepare synchronization contexts */
    if (jcp.global_transpose && jcp.nthr_ic_b > 1) {
        const size_t tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_tr_diff_dst_bctx, tr_diff_dst_bctx_size);
    }

    if (IMPLICATION(jcp.nthr_mb == 1,
                (jcp.with_bias && jcp.bia_dt == data_type::bf16)
                        || jcp.wei_dt == data_type::bf16)) {
        const size_t wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block
                * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
        const size_t bia_size
                = jcp.with_bias * jcp.ngroups * jcp.nb_oc * jcp.oc_block;

        const int num_wei_buffers
                = jcp.wei_dt == data_type::bf16 ? jcp.nthr_mb : jcp.nthr_mb - 1;
        const int num_bia_buffers = jcp.with_bias
                ? (jcp.bia_dt == data_type::bf16 ? jcp.nthr_mb
                                                 : jcp.nthr_mb - 1)
                : 0;

        const size_t wei_bia_reduction_size
                = wei_size * num_wei_buffers + bia_size * num_bia_buffers;

        scratchpad.book<float>(
                key_conv_wei_bia_reduction, wei_bia_reduction_size);

        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx, 1);
    }

    if (jcp.with_bias
            && ((jcp.oc_without_padding % jcp.oc_block != 0)
                    && jcp.bia_dt == data_type::f32)) {
        scratchpad.book(key_conv_padded_bias,
                jcp.ngroups * jcp.nb_oc * jcp.oc_block, jcp.typesize_bia);
    }
    scratchpad.book(key_conv_amx_tilecfg, 1, 64); // 1 whole cacheline

    constexpr size_t scratchpad_limit_by_absolute_value = (size_t)32
            << 30; // 32Gb - TODO: may it's too large?
    const size_t scratchpad_limit_by_tensor_sizes = (size_t)32 * jcp.nthr
            * (src_d.size() + diff_weights_d.size() + diff_dst_d.size());
    const size_t scratchpad_limit
            = nstl::min(scratchpad_limit_by_absolute_value,
                    scratchpad_limit_by_tensor_sizes);
    if (scratchpad.size() > scratchpad_limit)
        return status::unimplemented;
    else
        return status::success;
}

void jit_avx512_core_amx_bwd_weights_kernel_t::balance(const jit_conv_conf_t &j,
        int &nthr_, int &nthr_mb_, int &nthr_g_, int &nthr_oc_b_,
        int &nthr_ic_b_) {
    nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

    const int max_threads = dnnl_get_max_threads();

    if (max_threads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        nthr_ = nthr_g_ = max_threads;
        return;
    }

    nthr_g_ = j.ngroups;
    const int nthr = max_threads / nthr_g_;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level optimizer
         * tries to minimize memory consumption. few notes:
         *  (n1) if weights tensor size is less than source and destination
         *       tensors we apply the ratio of the source and destination
         *       tensor sizes to weights one as compensation coefficient to
         *       avoid parallelization across batch size only, othervise we
         *       apply additional coefficient to source component based on
         *       performance measurements
         *  (n2) use scales based on output vs input channels ratio for source
         *       and destination componets to imporve threading balance across
         *       input and output channels */

        const dim_t src_type_size = 2;
        const dim_t wei_type_size = 4;

        dim_t src_size
                = (dim_t)j.mb * j.ic * j.id * j.ih * j.tr_iw * src_type_size;
        dim_t dst_size
                = (dim_t)j.mb * j.oc * j.od * j.oh * j.tr_ow * src_type_size;
        dim_t wei_size
                = (dim_t)j.oc * j.ic * j.kd * j.kh * j.kw * wei_type_size;

        float wei_compensation_scale = 0.5f * (dst_size + src_size) / wei_size;
        float oi_channels_ratio = (float)(j.nb_oc / j.nb_oc_blocking)
                / (j.nb_ic / j.nb_ic_blocking);
        auto get_src_coef = [=]() {
            float src_coef = nstl::max(1.0f / oi_channels_ratio, 1.0f);
            if (wei_compensation_scale < 1.0f) src_coef *= 4.0f;

            return src_coef;
        };

        auto get_dst_coef
                = [=]() { return nstl::max(oi_channels_ratio, 1.0f); };

        auto get_wei_coef
                = [=]() { return nstl::max(wei_compensation_scale, 1.0f); };

        const float src_coef = get_src_coef();
        const float dst_coef = get_dst_coef();
        const float wei_coef = get_wei_coef();

        float src_v = src_coef * div_up(j.nthr_mb_work, nthr_mb)
                * div_up(j.ngroups, nthr_g_)
                * div_up((j.nb_ic / j.nb_ic_blocking), nthr_ic_b) * j.mb
                * (j.ic_block * j.nb_ic_blocking) * j.id * j.ih * j.tr_iw
                / j.nthr_mb_work / j.stride_d / j.stride_h / j.stride_w;
        float wei_v = wei_coef * div_up(j.ngroups, nthr_g_)
                * div_up((j.nb_oc / j.nb_oc_blocking),
                        (j.oc_block * j.nb_oc_blocking) * nthr_oc_b)
                * div_up((j.nb_ic / j.nb_ic_blocking), nthr_ic_b) * j.kh * j.kw
                * j.kd * (j.ic_block * j.nb_ic_blocking)
                * (j.oc_block * j.nb_oc_blocking);
        float dst_v = dst_coef * div_up(j.nthr_mb_work, nthr_mb)
                * div_up(j.ngroups, nthr_g_)
                * div_up((j.nb_oc / j.nb_oc_blocking),
                        (j.oc_block * j.nb_oc_blocking) * nthr_oc_b)
                * j.mb * (j.oc_block * j.nb_oc_blocking) * j.od * j.oh * j.tr_ow
                / j.nthr_mb_work;

        return src_v + dst_v + wei_v;
    };

    float best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* find the best thread distribution with lowest memory cost */

    const int nthr_mb_max = nstl::min(nthr, j.nthr_mb_work);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par,
                (j.nb_oc / j.nb_oc_blocking)); // Amount of nb_oc_blocks
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = nstl::min(
                    nthr_par / nthr_oc_b, (j.nb_ic / j.nb_ic_blocking));

            float mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

    if (nthr_mb_ > nthr / 2 && nthr_mb_ < nthr)
        nthr_mb_ = nstl::min(j.nthr_mb_work, nthr);
    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;

    assert(nthr_ <= max_threads);
}

void jit_diff_wei_trans_to_vnni_t::generate() {
    /* Reorder part of F32 weights tensor
       from [2I][kd][kh][kw][16i][16o] to VNNI format [kd][kh][kw][16i][16o][2i]
       and downconvert it to Bfloat16. */
    const int typesize_out = 2;
    const int typesize_acc = 4;

    using reg64_t = const Xbyak::Reg64;
    const reg64_t &reg_output = r15;
    const reg64_t &org_reg_output = r14;
    const reg64_t &reg_input = r13;
    const reg64_t &reg_input_1 = r12;
    const reg64_t &org_reg_input_1 = r11;
    const reg64_t &reg_input_2 = r10;
    const reg64_t &reg_prm_table = r9;
    const reg64_t &reg_last_ic_block = rax;
    const reg64_t &reg_kd = rsi;
    const reg64_t &reg_kh = abi_not_param1;
    const reg64_t &reg_tmp = rdx;

    const Xbyak::Zmm &zmm_idx = Xbyak::Zmm(31);
    auto get_zmm_src_0 = [&](int ic) { return Xbyak::Zmm(ic); };
    auto get_zmm_src_1 = [&](int ic) { return Xbyak::Zmm(4 + ic); };
    auto get_zmm_bf16 = [&](int ic) { return Xbyak::Zmm(8 + ic); };

    Xbyak::Label prm_table, zero_buffer;
    Xbyak::Label kd_loop_label, kh_loop_label;

    preamble();

    mov(reg_last_ic_block, ptr[abi_param1 + GET_OFF(last_ic_block)]);
    mov(org_reg_input_1, ptr[abi_param1 + GET_OFF(src)]);
    mov(org_reg_output, ptr[abi_param1 + GET_OFF(dst)]);

    mov(reg_prm_table, prm_table);
    vmovups(zmm_idx, ptr[reg_prm_table]);

    xor_(reg_kd, reg_kd);
    L(kd_loop_label);
    {
        mov(reg_output, org_reg_output);
        mov(reg_input_1, org_reg_input_1);
        xor_(reg_kh, reg_kh);
        L(kh_loop_label);
        {
            for (int kw = 0; kw < jcp.kw; kw++) {
                Xbyak::Label last_ic_label, done_ic_label;

                dim_t out_offset = (dim_t)typesize_out * kw * jcp.ic_block
                        * jcp.oc_block * 2;
                dim_t inp_1_offset = (dim_t)typesize_acc * kw * jcp.ic_block
                        * jcp.oc_block;
                dim_t inp_2_offset = (dim_t)typesize_acc
                        * (jcp.kd * jcp.kh * jcp.kw * jcp.ic_block
                                        * jcp.oc_block
                                + kw * jcp.ic_block * jcp.oc_block);

                cmp(reg_last_ic_block, 0);
                jne(last_ic_label, T_NEAR);

                mov(reg_input_2, reg_input_1);
                safe_add(reg_input_2, inp_2_offset, reg_tmp);
                jmp(done_ic_label, T_NEAR);

                L(last_ic_label);
                mov(reg_input_2, zero_buffer);

                L(done_ic_label);

                int ic_count = 0;
                for (int bc = 0; bc < 2; bc++) {
                    if (!bc) {
                        mov(reg_input, reg_input_1);
                        safe_add(reg_input, inp_1_offset, reg_tmp);
                    } else
                        mov(reg_input, reg_input_2);

                    for (int ic = 0; ic < jcp.ic_block / 2; ic++) {
                        auto zmm_src_0 = get_zmm_src_0(ic);
                        auto zmm_src_1 = get_zmm_src_1(ic);
                        auto zmm_bf16 = get_zmm_bf16(ic);

                        vmovups(zmm_src_0,
                                ptr[reg_input
                                        + typesize_acc
                                                * (2 * ic * jcp.oc_block)]);
                        vmovups(zmm_src_1,
                                ptr[reg_input
                                        + typesize_acc
                                                * ((2 * ic + 1)
                                                        * jcp.oc_block)]);

                        vcvtne2ps2bf16(zmm_bf16, zmm_src_1, zmm_src_0);
                        vpermw(zmm_bf16, zmm_idx, zmm_bf16);

                        vmovups(ptr[reg_output + out_offset
                                        + typesize_out * ic_count * 2
                                                * jcp.oc_block],
                                zmm_bf16);
                        ic_count++;
                    }
                }
            }
            safe_add(reg_output,
                    (dim_t)typesize_out * jcp.kw * 2 * jcp.ic_block
                            * jcp.oc_block,
                    reg_tmp);
            safe_add(reg_input_1,
                    (dim_t)typesize_acc * jcp.kw * jcp.ic_block * jcp.oc_block,
                    reg_tmp);

            add(reg_kh, 1);
            cmp(reg_kh, jcp.kh);
            jl(kh_loop_label, T_NEAR);
        }
        safe_add(org_reg_output,
                (dim_t)typesize_out * jcp.kh * jcp.kw * 2 * jcp.ic_block
                        * jcp.oc_block,
                reg_tmp);
        safe_add(org_reg_input_1,
                (dim_t)typesize_acc * jcp.kh * jcp.kw * jcp.ic_block
                        * jcp.oc_block,
                reg_tmp);

        add(reg_kd, 1);
        cmp(reg_kd, jcp.kd);
        jl(kd_loop_label, T_NEAR);
    }

    postamble();

    align(64);
    L(prm_table);
    const uint16_t prm_array[32]
            = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9,
                    25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31};
    for (size_t i = 0; i < 32; ++i)
        dw(prm_array[i]);

    align(64);
    L(zero_buffer);
    const uint16_t zero = 0;
    for (size_t i = 0; i < typesize_acc * jcp.oc_block * jcp.ic_block; ++i)
        db(zero);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
