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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "jit_transpose_src_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

void jit_transpose_src::transpose(int nrows, int l_pad, int r_pad, bool nontemporal_stores) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows)
        return;

    auto pf_src_t0 = [=](int i) {
        if(enable_prefetch) prefetcht0(EVEX_compress_addr(reg_src, (transpose_size + i) * src_stride));
    };

    auto pf_tr_src_t0 = [=](int i) {
        int offset = (transpose_size) * typesize + i * tr_src_stride;
        if(enable_prefetch) prefetcht0(EVEX_compress_addr(reg_tr_src, offset));
        if(enable_prefetch) prefetcht0(EVEX_compress_addr(reg_tr_src, offset + 64));
    };

    auto pf_src_t1 = [=](int i) {
        if(enable_prefetch) prefetcht1(EVEX_compress_addr(reg_src_prf, i * src_stride));
    };

    auto pf_tr_src_t1 = [=](int i) {
        if(enable_prefetch) prefetchwt1(EVEX_compress_addr(reg_tr_src_prf, i * tr_src_stride));
    };

    auto src_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(i);
    };

    auto tmp_zmm = [=](int i) {
        assert(i >= 0 && i < 16);
        return Zmm(16 + i);
    };

    auto load = [=](int i) {
        vmovups(src_zmm(i), EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [=](Zmm r, int i) {
        auto kmovw = [=](Opmask k, unsigned w) {
            mov(regw_tmp, w);
            jit_generator::kmovw(k, regw_tmp);
        };

        auto padding = [=] (Reg64 reg, int pad) {
            kmovw(kTail, (1 << pad) - 1);
            auto k = kTail;
            auto base = reg;
            base.setOpmaskIdx(k.getIdx(), true);

            auto zmm_zero = r;
            vpxord(zmm_zero, zmm_zero, zmm_zero);
            auto addr = EVEX_compress_addr(base, i * tr_src_stride);
            vmovups(addr, zmm_zero);
        };

        mov(reg_tr_src_tmp, reg_tr_src);
        if (l_pad > 0)
            add(reg_tr_src_tmp, l_pad * typesize);

        if (tail != transpose_size)
            kmovw(kTail, (1 << tail) - 1);

        // Xbyak does not allow k0 to be specified explicitly via the '|'
        // operator, so we have to do this via a method call (implicitly
        // EVEX encoding uses k0 to mean 'no mask')
        bool partial_store = nrows < 16;
        auto k = partial_store ? kTail : k0;
        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(k.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        if (nontemporal_stores && !partial_store)
            vmovntps(addr, r);
        else
            vmovups(addr, r);

        if (r_pad > 0) {
            add(reg_tr_src_tmp, tail * typesize);
            padding(reg_tr_src_tmp, r_pad);
        }

        if (l_pad > 0) {
            padding(reg_tr_src, l_pad);
        }

    };

    auto transpose16x8 = [=](int base_idx) {
        assert(base_idx == 0 || base_idx == 8);

        // swap 1
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i * 2;
            int src_idx1 = src_idx0 + 1;

            int next_src_idx0 = src_idx0 + 2;
            int next_src_idx1 = src_idx1 + 2;
            bool load_next = base_idx == 0 || i < 3;

            if (base_idx == 0 && i == 0) {
                load(src_idx0);
                load(src_idx1);
            }

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx1);
            auto src0 = src_zmm(src_idx0);
            auto src1 = src_zmm(src_idx1);

            if (next_src_idx0 < nrows && load_next)
                load(next_src_idx0);
            valignd(tmp0, src0, src0, 0x1);
            pf_src_t1(base_idx + i);

            if (next_src_idx1 < nrows && load_next)
                load(next_src_idx1);
            valignd(tmp1, src1, src1, 0xf);
            pf_src_t0(base_idx + i);

            vmovaps(src0 | kAAAA, tmp1);
            vmovaps(src1 | k5555, tmp0);
        }
        // swap 2
        for (int i = 0; i < 4; i++) {
            int select_half = (i < 2) ? 0 : 2;
            int src_idx0 = base_idx + i + select_half + 0;
            int src_idx2 = src_idx0 + 2;

            auto tmp0 = tmp_zmm(src_idx0);
            auto tmp1 = tmp_zmm(src_idx2);
            auto src0 = src_zmm(src_idx0);
            auto src2 = src_zmm(src_idx2);

            valignd(tmp0, src0, src0, 0x2);
            pf_src_t1(base_idx + 4 + i);
            valignd(tmp1, src2, src2, 0xe);
            pf_src_t0(base_idx + 4 + i);
            vmovaps(src2 | k3333, tmp0);
            vmovaps(src0 | kCCCC, tmp1);
        }

        // swap 4
        for (int i = 0; i < 4; i++) {
            int src_idx0 = base_idx + i;
            int src_idx4 = src_idx0 + 4;

            auto tmp0 = tmp_zmm(src_idx0);
            auto src0 = src_zmm(src_idx0);
            auto src4 = src_zmm(src_idx4);

            vmovaps(tmp0, src0);
            vshuff32x4(src0 | kF0F0, src4, src4, 0xb1);
            pf_tr_src_t1(base_idx / 2 + i);
            vshuff32x4(src4 | k0F0F, tmp0, tmp0, 0xb1);
            pf_tr_src_t0(base_idx / 2 + i);
        }
    };

    auto fixup16x16 = [=]() {

        // swap 8
        for (int i = 0; i < 8; i++) {
            auto tmp = tmp_zmm(i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0x44);
            store(tmp, i);
            if (i % 2 == 0) {
                pf_tr_src_t1(8 + i / 2);
                pf_tr_src_t0(8 + i / 2);
            }
        }

        for (int i = 0; i < 8; i++) {
            auto tmp = tmp_zmm(8 + i);
            auto src0 = src_zmm(i);
            auto src8 = src_zmm(8 + i);
            vshuff64x2(tmp, src0, src8, 0xee);
            store(tmp, 8 + i);
            if (i % 2 == 0) {
                pf_tr_src_t1(12 + i / 2);
                pf_tr_src_t0(12 + i / 2);
            }
        }
    };

    transpose16x8(0);
    transpose16x8(8);
    fixup16x16();

}

void jit_transpose_src::generate()
{
    preamble();

    const int ic_block = params->ic_block;
    const int iw = params->iw;
    const int tr_iw = params->tr_iw;
    const int transposes = utils::div_up(iw, transpose_size);
    int loop_iters = nstl::max(0, transposes - 1);
    tail = iw - loop_iters * transpose_size;

    src_stride = ic_block * typesize;
    assert(src_stride == 64);
    tr_src_stride = tr_iw * typesize;

    bool nontemporal_stores = false;
    enable_prefetch = iw > small_spatial ? 1 : 0;

    assert(transpose_size == ic_block);
    const int src_step = ic_block * transpose_size * typesize;
    const int tr_src_step = ic_block * typesize;

    const int left_pad = params->l_pad;
    const int right_pad = tr_iw - iw - left_pad;

#define GET_TR_OFF(x) offsetof(jit_src_transpose_s, x)
    mov(reg_src, ptr [param1 + GET_TR_OFF(src)]);
    mov(reg_tr_src, ptr [param1 + GET_TR_OFF(tr_src)]);
    mov(reg_src_prf, ptr [param1 + GET_TR_OFF(src_prf)]);
    mov(reg_tr_src_prf, ptr [param1 + GET_TR_OFF(tr_src_prf)]);
#undef GET_TR_OFF

    auto kmovw = [=](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator::kmovw(k, regw_tmp);
    };

    kmovw(k3333, 0x3333); // 0011001100110011
    kmovw(k5555, 0x5555); // 0101010101010101
    kmovw(kAAAA, 0xaaaa); // 1010101010101010
    kmovw(kCCCC, 0xcccc); // 1100110011001100
    kmovw(k0F0F, 0x0f0f); // 0000111100001111
    kmovw(kF0F0, 0xf0f0); // 1111000011110000

    if (left_pad > 0 && loop_iters > 0) {
        loop_iters--;
        transpose(transpose_size, left_pad, 0, nontemporal_stores);
        add(reg_src, src_step);
        add(reg_tr_src, tr_src_step + left_pad * typesize);
        add(reg_src_prf, src_step);
        add(reg_tr_src_prf, tr_src_step + left_pad * typesize);
    }

    if (loop_iters) {
        mov(reg_loop, loop_iters);
        L("loop"); {
            transpose(transpose_size, 0, 0, nontemporal_stores);
            add(reg_src, src_step);
            add(reg_tr_src, tr_src_step);
            add(reg_src_prf, src_step);
            add(reg_tr_src_prf, tr_src_step);
            sub(reg_loop, 1);
            jnz("loop");
        }
    }
    if (transposes > 1)
        transpose(tail, 0, right_pad, nontemporal_stores);
    else
        transpose(tail, left_pad, right_pad, nontemporal_stores);

    postamble();
}

}
}
}
