/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

void jit_generator::transpose(const Xbyak::Reg64 &reg_src,
        const Xbyak::Reg64 &reg_dst, dim_t src_stride, dim_t dst_stride,
        int nrows, int ncolumns, data_type_t dt, Xbyak::Ymm &ymm_tmp,
        Xbyak::Ymm &ymm_mask, Xbyak::Xmm &xmm_upper_mask) {
    // no row padding for dst, so no work needed to be done
    if (ncolumns == 0) return;

    // Note: For stores we assume, the memory is padded, hence avoiding use of
    // mask stores.
    const auto xmm_lower_mask = Xbyak::Xmm(ymm_mask.getIdx());
    const auto xmm_tmp = Xbyak::Xmm(ymm_tmp.getIdx());

    // only avx2 version is supported for now. TODO for others
    const int transpose_size
            = vreg_traits<Xbyak::Ymm>::vlen / types::data_type_size(dt);
    assert(is_valid_isa(avx2));
    assert(nrows <= transpose_size && ncolumns <= transpose_size);

    assert(dt == data_type::f32
            && "transpose utils not supported for current data type");

    if (transpose_size > nrows) uni_vxorps(ymm_tmp, ymm_tmp, ymm_tmp);

    auto load_src = [=](Xbyak::Xmm vmm, int r, int c) {
        const int simd_w = vmm.getBit() / (types::data_type_size(dt) * 8);
        const auto addr
                = ptr[reg_src + r * src_stride + c * types::data_type_size(dt)];
        if (r >= nrows) {
            uni_vxorps(vmm, vmm, vmm);
        } else if (c + simd_w <= ncolumns) {
            vmovups(vmm, addr);
        } else if (simd_w == 8) {
            vmaskmovps(vmm, ymm_mask, addr);
        } else if (c == 0) {
            vmaskmovps(vmm, xmm_lower_mask, addr);
        } else {
            vmaskmovps(vmm, xmm_upper_mask, addr);
        }
    };

    auto vinsert = [=](Xbyak::Ymm ymm, int r, int c) {
        const int xmm_simd_w = 4;
        const auto addr = ptr[reg_src + r * src_stride + c * sizeof(float)];
        if (r >= nrows) {
            // upper xmm of ymm_tmp is initialized to zero
            vperm2i128(ymm, ymm, ymm_tmp, 0x30);
            // vinsertf128(ymm, ymm, xmm_zero, 1);
        } else if (c + xmm_simd_w <= ncolumns) {
            vinsertf128(ymm, ymm, addr, 1);
        } else {
            vmaskmovps(xmm_tmp, c == 0 ? xmm_lower_mask : xmm_upper_mask, addr);
            vinsertf128(ymm, ymm, xmm_tmp, 1);
        }
    };

    // Intel(R) Software Optimization manual
    // Example 15-20. 8x8 Matrix Transpose Using VINSERTPS
    auto transpose_8x4 = [=](int col) {
        load_src(xmm0, 0, col);
        vinsert(ymm0, 4, col);
        load_src(xmm1, 1, col);
        vinsert(ymm1, 5, col);
        vunpcklpd(ymm8, ymm0, ymm1);
        vunpckhpd(ymm9, ymm0, ymm1);

        load_src(xmm2, 2, col);
        vinsert(ymm2, 6, col);
        load_src(xmm3, 3, col);
        vinsert(ymm3, 7, col);
        vunpcklpd(ymm10, ymm2, ymm3);
        vunpckhpd(ymm11, ymm2, ymm3);

        vshufps(ymm4, ymm8, ymm10, 0x88);
        vmovups(ptr[reg_dst + col * dst_stride], ymm4);

        if (col + 1 < ncolumns) {
            vshufps(ymm5, ymm8, ymm10, 0xDD);
            vmovups(ptr[reg_dst + (col + 1) * dst_stride], ymm5);
        }

        if (col + 2 < ncolumns) {
            vshufps(ymm6, ymm9, ymm11, 0x88);
            vmovups(ptr[reg_dst + (col + 2) * dst_stride], ymm6);
        }

        if (col + 3 < ncolumns) {
            vshufps(ymm7, ymm9, ymm11, 0xDD);
            vmovups(ptr[reg_dst + (col + 3) * dst_stride], ymm7);
        }
    };

    transpose_8x4(0);
    if (ncolumns > 4) transpose_8x4(4);
}
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
