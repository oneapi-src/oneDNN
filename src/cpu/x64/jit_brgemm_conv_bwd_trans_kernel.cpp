/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#include "cpu/x64/jit_brgemm_conv_bwd_trans_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace nstl;
using namespace data_type;

namespace jit_avx512_core_brgemm_conv_bwd_trans_kernel {

#define GET_OFF(field) offsetof(jit_brgemm_conv_bwd_trans_kernel_call_s, field)

template <typename Vmm>
jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::
        jit_avx512_core_brgemm_conv_bwd_trans_kernel_t(
                const jit_brgemm_conv_conf_t &ajcp, const char *name)
    : jit_generator(name)
    , jcp(ajcp)
    , inp_dsz(jcp.src_dsz)
    , oc_block_sz(inp_dsz * jcp.oc_block)
    , ow_size(inp_dsz * jcp.ngroups * jcp.oc_without_padding)
    , dst_w_block(jcp.ow_block)
    , dst_stride(jcp.owp)
    , dst_w_offset(oc_block_sz)
    , dst_h_offset(dst_stride * dst_w_offset)
    , VL(vreg_traits<Vmm>::vlen)
    , n_vec(jcp.oc_block / jcp.simd_w)
    , n_tail_vec((jcp.oc_without_padding % jcp.oc_block) / jcp.simd_w) {}

template <typename Vmm>
int jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::inp_w(
        int out_w) const {
    const auto res = div_up(out_w + jcp.l_pad % jcp.stride_w, jcp.stride_w)
            + (jcp.ext_kw - 1 - jcp.l_pad % jcp.stride_w) / jcp.stride_w;
    return res;
}

template <typename Vmm>
int jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::inp_w_start(
        int iwb) const {
    const auto sw = jcp.l_pad % jcp.stride_w;
    const auto kw = (jcp.kw - 1) % jcp.stride_w;
    const auto kw_x = (jcp.kw - 1) - nstl::modulo(kw - sw, jcp.stride_w);
    const auto ow = (iwb * jcp.iw_block + jcp.l_pad - kw_x * (jcp.dilate_w + 1))
            / jcp.stride_w;
    return ow;
}

// use different vmovdqu32/16/8 due to case when tail mask used
template <typename Vmm>
void jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::load(
        const Vmm &x, const Xbyak::Address &addr, const int load_size) {
    assert(is_zmm_ && "only Zmm registers allowed");
    switch (jcp.src_dt) {
        case f32:
        case s32: vmovdqu32(x, addr); break;
        case bf16:
        case f16: vmovdqu16(x, addr); break;
        case s8:
        case u8:
        case f8_e5m2:
        case f8_e4m3: vmovdqu8(x, addr); break;
        default: assert(!"Unknown type!");
    }
}

template <typename Vmm>
void jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::store(
        const Xbyak::Address &addr, const Vmm &x, const int store_size) {
    assert(is_zmm_ && "only Zmm registers allowed");
    switch (jcp.src_dt) {
        case f32:
        case s32: vmovdqu32(addr, x); break;
        case bf16:
        case f16: vmovdqu16(addr, x); break;
        case s8:
        case u8:
        case f8_e5m2:
        case f8_e4m3: vmovdqu8(addr, x); break;
        default: assert(!"Unknown type!");
    }
}

template <>
void jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Xbyak::Ymm>::load(
        const Xbyak::Ymm &x, const Xbyak::Address &addr, const int load_size) {
    load_bytes(x, addr, jcp.src_dsz * load_size, true);
}

template <>
void jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Xbyak::Ymm>::store(
        const Xbyak::Address &addr, const Xbyak::Ymm &x, const int store_size) {
    store_bytes(x, addr, jcp.src_dsz * store_size);
}

template <typename Vmm>
void jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::zero_oc_block(
        bool is_oc_tail, dim_t dst_off) {
    bool has_block_tail = (jcp.oc_block % jcp.simd_w);

    // TODO: use Xmm or Ymm moves for better small oc efficiency
    auto nvec = is_oc_tail ? n_tail_vec : n_vec;
    for (int iv = 0; iv < nvec; iv++)
        store(ptr[aux_dst_ptr + dst_off + iv * VL], vmm_zero, jcp.simd_w);
    const auto last_dst_off = aux_dst_ptr + dst_off + nvec * VL;
    if (has_block_tail) {
        if (is_zmm_)
            store(ptr[last_dst_off] | kblock_tail_mask | T_z, vmm_zero);
        else {
            const int block_tail = jcp.oc_block % jcp.simd_w;
            store(ptr[last_dst_off], vmm_zero, block_tail);
        }
    } else if (is_oc_tail)
        store(ptr[last_dst_off], vmm_zero, jcp.simd_w);
}

template <typename Vmm>
void jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::copy_oc_block(
        bool is_oc_tail, dim_t inp_off, dim_t dst_off, bool do_load) {
    bool has_block_tail = (jcp.oc_block % jcp.simd_w);
    // TODO: use Xmm or Ymm moves for better small oc efficiency
    auto nvec = is_oc_tail ? n_tail_vec : n_vec;
    for (int iv = 0; iv < nvec; iv++) {
        if (do_load)
            load(vmm_tmp, ptr[aux_inp_ptr + inp_off + iv * VL], jcp.simd_w);
        store(ptr[aux_dst_ptr + dst_off + iv * VL], vmm_tmp, jcp.simd_w);
    }
    const auto last_inp_off = aux_inp_ptr + inp_off + nvec * VL;
    const auto last_dst_off = aux_dst_ptr + dst_off + nvec * VL;

    if (is_oc_tail) {
        if (is_zmm_) {
            auto vmm_tmp_mask = vmm_tmp | ktail_mask | T_z;
            if (do_load) load(vmm_tmp_mask, ptr[last_inp_off]);
            if (has_block_tail)
                store(ptr[last_dst_off] | kblock_tail_mask | T_z, vmm_tmp);
            else
                store(ptr[last_dst_off], vmm_tmp);
        } else {
            const int oc_tail
                    = (jcp.oc_without_padding % jcp.oc_block) % jcp.simd_w;
            if (do_load) load(vmm_tmp, ptr[last_inp_off], oc_tail);
            if (has_block_tail) {
                const int block_tail = jcp.oc_block % jcp.simd_w;
                store(ptr[last_dst_off], vmm_tmp, block_tail);
            } else
                store(ptr[last_dst_off], vmm_tmp, jcp.simd_w);
        }
    } else if (has_block_tail) {
        if (is_zmm_) {
            auto vmm_tmp_mask = vmm_tmp | kblock_tail_mask | T_z;
            if (do_load) load(vmm_tmp_mask, ptr[last_inp_off]);
            store(ptr[last_dst_off] | kblock_tail_mask | T_z, vmm_tmp);
        } else {
            const int block_tail = jcp.oc_block % jcp.simd_w;
            if (do_load) load(vmm_tmp, ptr[last_inp_off], block_tail);
            store(ptr[last_dst_off], vmm_tmp, block_tail);
        }
    }
}

template <typename Vmm>
void jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::generate() {
    preamble();

    mov(inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(dst_ptr, ptr[param1 + GET_OFF(dst)]);
    mov(reg_hc, ptr[param1 + GET_OFF(h_count)]);
    mov(reg_t_pad, ptr[param1 + GET_OFF(t_pad)]);
    mov(reg_b_pad, ptr[param1 + GET_OFF(b_pad)]);
    mov(reg_iwb, ptr[param1 + GET_OFF(iwb)]);
    mov(reg_oc, ptr[param1 + GET_OFF(oc)]);

    uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

    if (is_zmm_ && jcp.oc_without_padding % jcp.oc_block) {
        int tail_size = (jcp.oc_without_padding % jcp.oc_block) % jcp.simd_w;
        uint64_t mask = (UINT64_C(1) << tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(ktail_mask, reg_tmp);
    }

    if (is_zmm_ && jcp.oc_block % jcp.simd_w) {
        int block_tail_size = jcp.oc_block % jcp.simd_w;
        uint64_t mask = (UINT64_C(1) << block_tail_size) - 1;
        mov(reg_tmp, mask);
        kmovq(kblock_tail_mask, reg_tmp);
    }

    auto ocb_loop_body = [&](bool is_oc_tail) {
        Xbyak::Label kh_label, no_kh_label;
        Xbyak::Label kh_tover_label, kh_bover_label;
        Xbyak::Label no_kh_tover_label, no_kh_bover_label;

        mov(aux_inp_ptr, inp_ptr);
        mov(aux_dst_ptr, dst_ptr);

        cmp(reg_hc, 0);
        jle(no_kh_bover_label, T_NEAR); // nothing to do

        cmp(reg_t_pad, 0);
        jle(no_kh_tover_label, T_NEAR);

        mov(kh_over, reg_t_pad);
        L(kh_tover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small oc
            for (dim_t ow = 0; ow < dst_w_block; ow++)
                zero_oc_block(is_oc_tail, ow * dst_w_offset);
            add(aux_dst_ptr, dst_h_offset);

            dec(kh_over);
            jnz(kh_tover_label, T_NEAR);
        }
        sub(reg_hc, reg_t_pad);
        L(no_kh_tover_label);

        cmp(reg_hc, reg_b_pad);
        jle(no_kh_label, T_NEAR);

        L(kh_label);
        {
            copy_iw_block(is_oc_tail);
            auto inp_h_offset = jcp.ow * ow_size;

            add(aux_inp_ptr, inp_h_offset);
            add(aux_dst_ptr, dst_h_offset);

            dec(reg_hc);
            cmp(reg_hc, reg_b_pad);
            jg(kh_label, T_NEAR);
        }
        L(no_kh_label);

        cmp(reg_hc, 0);
        jle(no_kh_bover_label, T_NEAR);

        L(kh_bover_label);
        {
            // TODO: adjust step to improve zeroing efficiency for small oc
            for (dim_t ow = 0; ow < dst_w_block; ow++)
                zero_oc_block(is_oc_tail, ow * dst_w_offset);
            add(aux_dst_ptr, dst_h_offset);

            dec(reg_hc);
            jnz(kh_bover_label, T_NEAR);
        }
        L(no_kh_bover_label);

        // End IC Loop
        auto inp_cb_offset = oc_block_sz;
        auto dst_cb_offset = jcp.ohp * dst_h_offset;

        add(inp_ptr, inp_cb_offset);
        add(dst_ptr, dst_cb_offset);
    };

    for (int ocb = 0; ocb < jcp.nb_oc_blocking; ocb++) {
        Xbyak::Label oc_tail_label, ocb_continue_label;
        add(reg_oc, jcp.oc_block);
        cmp(reg_oc, jcp.oc);
        jg(oc_tail_label, T_NEAR);

        ocb_loop_body(false);
        jmp(ocb_continue_label, T_NEAR);

        L(oc_tail_label);
        ocb_loop_body(true);

        L(ocb_continue_label);
    }

    postamble();
}

template <typename Vmm>
void jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::copy_iw_block(
        bool is_oc_tail) {
    if (jcp.l_ovf > 0) {
        for (dim_t ind_w = 0; ind_w < jcp.l_ovf; ind_w++)
            zero_oc_block(is_oc_tail, (ind_w + jcp.l_ovf) * dst_w_offset);
    }

    Xbyak::Label copy_block_done_label;

    int start_first_zero_block = -1;
    int end_first_zero_block = -1;
    int start_first_partial_block = -1;
    int end_first_partial_block = -1;
    int start_full_block = -1;
    int end_full_block = -1;
    int start_last_partial_block = -1;
    int end_last_partial_block = -1;

    int iw_block_tail = jcp.iw % jcp.iw_block;

    for (int iwb = 0; iwb < jcp.nb_iw; iwb++) {
        const auto inp_block = inp_w(jcp.iw_block);
        const auto inp_start = inp_w_start(iwb);
        const auto inp_end = inp_start + inp_block;
        if (inp_start + inp_block < 0) {
            if (start_first_zero_block == -1) start_first_zero_block = iwb;
            end_first_zero_block = iwb;
        } else if (inp_start < 0) {
            if (start_first_partial_block == -1)
                start_first_partial_block = iwb;
            end_first_partial_block = iwb;
        } else if (inp_start < jcp.ow) {
            if (inp_end <= jcp.ow) {
                if (start_full_block == -1) start_full_block = iwb;
                end_full_block = iwb;
            } else {
                if (start_last_partial_block == -1)
                    start_last_partial_block = iwb;
                end_last_partial_block = iwb;
            }
        }
    }

    if (start_first_zero_block != -1) {
        Xbyak::Label skip_first_zero_blocks;
        cmp(reg_iwb, end_first_zero_block);
        jg(skip_first_zero_blocks, T_NEAR);
        // zero block
        copy_iw_block_body(0, jcp.iw_block, 0, is_oc_tail);
        jmp(copy_block_done_label, T_NEAR);

        L(skip_first_zero_blocks);
    }
    if (start_first_partial_block != -1) {
        for (int b = start_first_partial_block; b <= end_first_partial_block;
                b++) {
            int cur_iw_block = (b == jcp.nb_iw - 1 && iw_block_tail > 0)
                    ? iw_block_tail
                    : jcp.iw_block;
            const auto inp_block = inp_w(cur_iw_block);
            const auto inp_start = inp_w_start(b);
            const auto inp_end = inp_start + inp_block;
            const auto block_lpad = -inp_start;
            const auto block_len = nstl::min(jcp.ow, inp_end);
            Xbyak::Label skip_first_partial_block;
            cmp(reg_iwb, b);
            jne(skip_first_partial_block, T_NEAR);
            copy_iw_block_body(block_lpad, jcp.iw_block, block_len, is_oc_tail);
            jmp(copy_block_done_label, T_NEAR);
            L(skip_first_partial_block);
        }
    }
    if (start_full_block != -1) {
        Xbyak::Label skip_full_blocks;
        cmp(reg_iwb, end_full_block);
        jg(skip_full_blocks, T_NEAR);
        copy_iw_block_body(0, jcp.iw_block, inp_w(jcp.iw_block), is_oc_tail);
        jmp(copy_block_done_label, T_NEAR);

        L(skip_full_blocks);
    }
    if (start_last_partial_block != -1) {
        for (int b = start_last_partial_block; b <= end_last_partial_block;
                b++) {
            int cur_iw_block = (b == jcp.nb_iw - 1 && iw_block_tail > 0)
                    ? iw_block_tail
                    : jcp.iw_block;
            const auto inp_block = inp_w(cur_iw_block);
            const auto inp_start = inp_w_start(b);
            const auto inp_end = inp_start + inp_block;
            const auto block_lpad = 0;
            const auto block_len = nstl::min(jcp.ow, inp_end) - inp_start;
            Xbyak::Label skip_last_partial_block;
            cmp(reg_iwb, b);
            jne(skip_last_partial_block, T_NEAR);
            copy_iw_block_body(block_lpad, cur_iw_block, block_len, is_oc_tail);
            jmp(copy_block_done_label, T_NEAR);

            L(skip_last_partial_block);
        }
    }

    // if there are no executed cases above then owb is among last zero blocks
    // check if this is needed and if it is partial
    copy_iw_block_body(0, jcp.iw_block, 0, is_oc_tail);

    L(copy_block_done_label);
}

template <typename Vmm>
void jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>::copy_iw_block_body(
        int lpad, int iw_len, int ow_len, bool is_oc_tail) {
    const auto dst_width = inp_w(iw_len) + lpad;
    for (dim_t ind_w = 0; ind_w < dst_width; ind_w++) {
        auto ow_idx = ind_w - lpad;
        auto dst_off = (ind_w + jcp.l_ovf) * dst_w_offset;
        if (ow_idx < 0 || ow_idx >= ow_len) {
            zero_oc_block(is_oc_tail, dst_off);
        } else {
            auto inp_off = ow_idx * ow_size;
            copy_oc_block(is_oc_tail, inp_off, dst_off, true);
        }
    }
}

template struct jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Xbyak::Ymm>;
template struct jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Xbyak::Zmm>;

} // namespace jit_avx512_core_brgemm_conv_bwd_trans_kernel
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
