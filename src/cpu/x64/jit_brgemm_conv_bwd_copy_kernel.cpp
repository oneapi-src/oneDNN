/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "cpu/x64/jit_brgemm_conv_bwd_copy_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace nstl;
using namespace data_type;

namespace jit_avx512_core_brgemm_conv_bwd_copy_kernel {

#define GET_OFF(field) offsetof(jit_brgemm_conv_bwd_copy_kernel_call_s, field)

template <typename Vmm>
jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Vmm>::
        jit_avx512_core_brgemm_conv_bwd_copy_kernel_t(
                const jit_brgemm_conv_conf_t &ajcp)
    : jit_generator(jit_name()), jcp(ajcp) {}

// use different vmovdqu32/16/8 due to case when tail mask used
template <typename Vmm>
void jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Vmm>::load(
        const Vmm &x, const Xbyak::Address &addr, const int load_size) {
    assert(is_zmm_ && "only Zmm registers allowed");
    switch (jcp.dst_dt) {
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
void jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Vmm>::store(
        const Xbyak::Address &addr, const Vmm &x, const int store_size) {
    assert(is_zmm_ && "only Zmm registers allowed");
    switch (jcp.dst_dt) {
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
void jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Xbyak::Ymm>::load(
        const Xbyak::Ymm &x, const Xbyak::Address &addr, const int load_size) {
    load_bytes(x, addr, jcp.dst_dsz * load_size, true);
}

template <>
void jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Xbyak::Ymm>::store(
        const Xbyak::Address &addr, const Xbyak::Ymm &x, const int store_size) {
    store_bytes(x, addr, jcp.dst_dsz * store_size);
}

template <typename Vmm>
void jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Vmm>::generate() {
    preamble();

    const auto VL = vreg_traits<Vmm>::vlen;
    const auto simd_w = VL / jcp.dst_dsz;
    const auto n_vec = jcp.ic_block / simd_w;
    const auto n_tail_vec = (jcp.ic % jcp.ic_block) / simd_w;

    mov(inp_ptr, ptr[param1 + GET_OFF(src)]);
    mov(dst_ptr, ptr[param1 + GET_OFF(dst)]);
    mov(reg_num_ic, ptr[param1 + GET_OFF(num_ic)]);

    const auto iw_tail = jcp.iw % jcp.iw_block;
    const auto ic_tail = jcp.ic % jcp.ic_block;
    if (is_zmm_ && ic_tail) {
        int simd_tail = ic_tail % simd_w;
        uint64_t mask = (UINT64_C(1) << simd_tail) - 1;
        mov(reg_tmp, mask);
        kmovq(ktail_mask, reg_tmp);
    }

    const auto block_tail = jcp.ic_block % simd_w;
    if (is_zmm_ && block_tail) {
        uint64_t mask = (UINT64_C(1) << block_tail) - 1;
        mov(reg_tmp, mask);
        kmovq(kblock_tail_mask, reg_tmp);
    }

    auto iw_tail_loop_body = [&](bool is_ic_tail) {
        for (int iw = 0; iw < iw_tail; ++iw) {
            const auto iw_off = iw * jcp.ic_without_padding * jcp.dst_dsz;
            auto nvec = is_ic_tail ? n_tail_vec : n_vec;
            for (size_t iv = 0; iv < nvec; iv++) {
                load(vmm_tmp, ptr[inp_ptr + iw_off + iv * VL], simd_w);
                store(ptr[dst_ptr + iw_off + iv * VL], vmm_tmp, simd_w);
            }
            const auto last_inp_off = inp_ptr + iw_off + nvec * VL;
            const auto last_dst_off = dst_ptr + iw_off + nvec * VL;

            if (is_ic_tail) {
                if (is_zmm_) {
                    auto zmm_tmp_mask = vmm_tmp | ktail_mask | T_z;
                    load(zmm_tmp_mask, ptr[last_inp_off]);
                    store(ptr[last_dst_off] | ktail_mask, vmm_tmp);
                } else {
                    int simd_tail = ic_tail % simd_w;
                    load(vmm_tmp, ptr[last_inp_off], simd_tail);
                    store(ptr[last_dst_off], vmm_tmp, simd_tail);
                }
            } else if (block_tail) {
                if (is_zmm_) {
                    auto zmm_tmp_mask = vmm_tmp | kblock_tail_mask | T_z;
                    load(zmm_tmp_mask, ptr[last_inp_off]);
                    store(ptr[last_dst_off] | kblock_tail_mask, vmm_tmp);
                } else {
                    load(vmm_tmp, ptr[last_inp_off], block_tail);
                    store(ptr[last_dst_off], vmm_tmp, block_tail);
                }
            }
        }
    };

    Xbyak::Label ic_tail_label, end_label;

    cmp(reg_num_ic, ic_tail);
    jle(ic_tail_label, T_NEAR);
    iw_tail_loop_body(false);
    jmp(end_label, T_NEAR);
    L(ic_tail_label);
    iw_tail_loop_body(true);
    L(end_label);

    postamble();
}

template struct jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Xbyak::Ymm>;
template struct jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Xbyak::Zmm>;

} // namespace jit_avx512_core_brgemm_conv_bwd_copy_kernel
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
