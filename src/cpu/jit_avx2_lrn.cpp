/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "jit_avx2_lrn.hpp"
#include "jit_generator.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_format;

enum { VECTOR_LENGTH = 8, MAX_LOCAL_SIZE = 32 };
typedef struct {
    const float *src;
    float *dst, *scratch;
} jit_args_t;

struct nchw8c_across {
    int HW, version; // -1 channels 0..7, 1 channels C-8 .. C-1, 0 -- other channels
    nchw8c_across(int hw, int v) : HW(hw), version(v) {}
};

struct nchw8c_within {
    int H, W, size;
    nchw8c_within(int h, int w, int s) : H(h), W(w), size(s) {}
};

struct nchw_across {
    int C, HW, tail;
    nchw_across(int c, int hw, int t) : C(c), HW(hw), tail(t) {}
};

struct nhwc_across {
    int C;
    nhwc_across(int c) : C(c) {}
};

template <impl::precision_t prec>
struct jit_avx2_lrn<prec>::xbyak_lrn : public jit_generator {
    Xbyak::Reg64 src = rax;
    Xbyak::Reg64 dst = r8;
    Xbyak::Reg64 scratch = rdx;
    Xbyak::Reg64 imm_addr64 = rbx;

    Xbyak::Ymm yalpha = ymm0;
    Xbyak::Ymm yone = ymm1;

    static const float one;
    float alpha;

    void within_body(
        int hoff, int Hoff, int woff, int Woff, int stride,
        Xbyak::Ymm ysum, Xbyak::Ymm ydst, Xbyak::Ymm ytmp, Xbyak::Ymm ysum2,
        prop_kind_t pk)
    {
        vxorps(ysum, ysum, ysum);
        for (int i = hoff; i <= Hoff; ++i)
        {
            for (int j = woff; j <= Woff; ++j)
            {
                if (i == 0 && j == 0)
                {
                    vmovups(ydst, ptr[src]);
                    vfmadd231ps(ysum, ydst, ydst);
                }
                else
                {
                    vmovups(ytmp, ptr[src + (i * stride + j)* VECTOR_LENGTH * 4]);
                    vfmadd231ps(ysum, ytmp, ytmp);
                }
            }
        }
        vfmadd132ps(ysum, yone, yalpha); // ysum <- ysum*yalpha+yone
        vmovaps(ytmp, ysum);
        vmulps(ysum2, ysum, ysum);
        vmulps(ysum, ysum, ysum2); // ysum = (ysum*yalpha+yone)^3;
        vsqrtps(ysum, ysum);
        vsqrtps(ysum, ysum); // ysum = (ysum*yalpha+yone)^0.75
        vdivps(ydst, ydst, ysum); // ydst <- ydst / ysum
        if (pk != prop_kind::forward_scoring)
            vmulps(ysum, ysum, ytmp); // ysum = (ysum*yalpha+yone) ^ 1.75 -- for back prop
        vmovups(ptr[dst], ydst);
        if (pk != prop_kind::forward_scoring)
            vmovups(ptr[scratch], ysum);
        add(src, 32);
        add(dst, 32);
        if (pk != prop_kind::forward_scoring)
            add(scratch, 32);
    }

    xbyak_lrn(
        const struct nchw8c_within &J,
        float A,
        prop_kind_t pk,
        void *code_ptr = nullptr,
        size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , alpha(A)
    {
        Xbyak::Reg64 h = r9;
        Xbyak::Reg64 w = r10;
        Xbyak::Ymm ysum = ymm9;
        Xbyak::Ymm ysum2 = ymm10;
        Xbyak::Ymm ydst = ymm11;
        Xbyak::Ymm ytmp = ymm12;

        static const char *label[MAX_LOCAL_SIZE] = {
            ".l00", ".l01", ".l02", ".l03", ".l04", ".l05", ".l06", ".l07", ".l08", ".l09",
            ".l10", ".l11", ".l12", ".l13", ".l14", ".l15", ".l16", ".l17", ".l18", ".l19",
            ".l20", ".l21", ".l22", ".l23", ".l24", ".l25", ".l26", ".l27", ".l28", ".l29",
            ".l30", ".l31"
        };

        this->preamble();

        mov(src, ptr[this->param1 + 0]);
        mov(dst, ptr[this->param1 + 8]);
        if (pk != prop_kind::forward_scoring)
            mov(scratch, ptr[this->param1 + 16]);
        mov(imm_addr64, reinterpret_cast<size_t>(&this->alpha));
        vbroadcastss(yalpha, ptr[imm_addr64]);
        mov(imm_addr64, reinterpret_cast<size_t>(&this->one));
        vbroadcastss(yone, ptr[imm_addr64]);

        int s2 = (J.size - 1) / 2, S2 = J.size - s2 - 1;
        const char **label_t = &label[0];
        const char **label_b = &label[s2];

        for (int i = 0; i < s2; ++i)
        {
            for (int j = 0; j < s2; ++j)
                within_body(-i, S2, -j, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
            mov(w, J.W - J.size + 1);
            L(label_t[i]);
            within_body(-i, S2, -s2, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
            dec(w);
            cmp(w, 0);
            jne(label_t[i], T_NEAR);
            for (int j = J.W - S2; j < J.W; ++j)
                within_body(-i, S2, -s2, J.W - 1 - j, J.W, ysum, ydst, ytmp, ysum2, pk);
        }

        mov(h, J.H - J.size + 1);
        L(".lrn_loop_h");
        for (int j = 0; j < s2; ++j)
            within_body(-s2, S2, -j, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
        mov(w, J.W - J.size + 1);
        L(".lrn_loop_w");
        within_body(-s2, S2, -s2, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
        dec(w);
        cmp(w, 0);
        jne(".lrn_loop_w", T_NEAR);
        for (int j = J.W - S2; j < J.W; ++j)
            within_body(-s2, S2, -s2, J.W - 1 - j, J.W, ysum, ydst, ytmp, ysum2, pk);
        dec(h);
        cmp(h, 0);
        jne(".lrn_loop_h", T_NEAR);

        for (int i = J.H - S2; i < J.H; ++i)
        {
            for (int j = 0; j < s2; ++j)
                within_body(-s2, J.H - 1 - i, -j, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
            mov(w, J.W - J.size + 1);
            L(label_b[i - (J.H - S2)]);
            within_body(-s2, J.H - 1 - i, -s2, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
            dec(w);
            cmp(w, 0);
            jne(label_b[i - (J.H - S2)], T_NEAR);
            for (int j = J.W - S2; j < J.W; ++j)
                within_body(-s2, J.H - 1 - i, -s2, J.W - 1 - j, J.W, ysum, ydst, ytmp, ysum2, pk);
        }

        this->postamble();
    }

    xbyak_lrn(
        const struct nchw8c_across &J,
        float A,
        prop_kind_t pk,
        void *code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , alpha(A)
    {
        Xbyak::Reg64 t = rsp;
        Xbyak::Reg64 hw = r9;
        Xbyak::Xmm xsrc_prev = xmm2;
        Xbyak::Ymm ysrc = ymm3;
        Xbyak::Ymm yc = ymm3;
        Xbyak::Xmm xsrc_next = xmm4;
        Xbyak::Ymm ya = ymm5;
        Xbyak::Ymm yb = ymm6;
        Xbyak::Ymm yd = ymm7;
        Xbyak::Ymm ye = ymm8;
        Xbyak::Ymm ysum = ymm9;
        Xbyak::Ymm ysum2 = ymm10;
        Xbyak::Ymm ydst = ymm11;
        Xbyak::Ymm ybase = ymm12;

        this->preamble();

        mov(src, ptr[this->param1 + 0]);
        mov(dst, ptr[this->param1 + 8]);
        if (pk != prop_kind::forward_scoring)
            mov(scratch, ptr[this->param1 + 16]);
        sub(t, 64);
        mov(imm_addr64, reinterpret_cast<size_t>(&this->alpha));
        vbroadcastss(yalpha, ptr[imm_addr64]);
        mov(imm_addr64, reinterpret_cast<size_t>(&this->one));
        vbroadcastss(yone, ptr[imm_addr64]);
        if (J.version == -1)
        {
            vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
            vmovups(ptr[t + 0], xsrc_prev);
        }
        if (J.version == +1)
        {
            vxorps(xsrc_next, xsrc_next, xsrc_next);
            vmovups(ptr[t + 48], xsrc_next);
        }

        mov(hw, J.HW);
        L(".lrn_loop");

        if (J.version != -1) vmovups(xsrc_prev, ptr[src - J.HW * 32 + 16]);
        vmovups(ysrc, ptr[src]);
        if (J.version != +1) vmovups(xsrc_next, ptr[src + J.HW * 32]);

        if (J.version != -1) vmovups(ptr[t + 0], xsrc_prev);
        vmovups(ptr[t + 16], ysrc);
        if (J.version != +1) vmovups(ptr[t + 48], xsrc_next);

        vmovups(ya, ptr[t + 16 - 8]);
        vmovups(yb, ptr[t + 16 - 4]);
        vmovups(yd, ptr[t + 16 + 4]);
        vmovups(ye, ptr[t + 16 + 8]);
        vmulps(ysum, yc, yc);
        vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya*ya
        vfmadd231ps(ysum, yb, yb);
        vfmadd231ps(ysum, yd, yd);
        vfmadd231ps(ysum, ye, ye);

        vfmadd132ps(ysum, yone, yalpha); // ysum <- ysum*yalpha+yone
        vmovaps(ybase, ysum);
        vmulps(ysum2, ysum, ysum);
        vmulps(ysum, ysum, ysum2); // ysum = ybase^3;
        vsqrtps(ysum, ysum);
        vsqrtps(ysum, ysum); // ysum = ybase^0.75
        vdivps(ydst, ysrc, ysum); // ydst = ysrc / ysum
        if (pk != prop_kind::forward_scoring)
            vmulps(ysum, ysum, ybase); // ysum = ybase ^ 1.75 -- for back prop
        vmovups(ptr[dst], ydst);
        if (pk != prop_kind::forward_scoring)
            vmovups(ptr[scratch], ysum);

        add(src, 32);
        add(dst, 32);
        if (pk != prop_kind::forward_scoring)
            add(scratch, 32);
        dec(hw);
        cmp(hw, 0);
        jne(".lrn_loop", T_NEAR);

        add(t, 64);
        this->postamble();
    }

    xbyak_lrn(
        const struct nhwc_across &J,
        float A,
        prop_kind_t pk,
        void *code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , alpha(A)
    {
        static const uint32_t mask[] = {
            0, 0, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0, 0
        };

        Xbyak::Reg64 c = r9;
        Xbyak::Ymm ya = ymm2;
        Xbyak::Ymm yb = ymm3;
        Xbyak::Ymm yc = ymm4;
        Xbyak::Ymm yd = ymm5;
        Xbyak::Ymm ye = ymm6;
        Xbyak::Ymm ysum = ymm7;
        Xbyak::Ymm ydst = ymm8;
        Xbyak::Ymm ybase = ymm9;
        Xbyak::Ymm ymask = ymm10;

        this->preamble();

        mov(src, ptr[this->param1 + 0]);
        mov(dst, ptr[this->param1 + 8]);
        if (pk != prop_kind::forward_scoring)
            mov(scratch, ptr[this->param1 + 16]);
        mov(imm_addr64, reinterpret_cast<size_t>(&this->alpha));
        vbroadcastss(yalpha, ptr[imm_addr64]);
        mov(imm_addr64, reinterpret_cast<size_t>(&this->one));
        vbroadcastss(yone, ptr[imm_addr64]);

        vxorps(ysum, ysum, ysum);

        mov(imm_addr64, reinterpret_cast<size_t>(&mask[0]));
        vmovups(ymask, ptr[imm_addr64]);
        vmaskmovps(ya, ymask, ptr[src - 8]);
        vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

        mov(imm_addr64, reinterpret_cast<size_t>(&mask[1]));
        vmovups(ymask, ptr[imm_addr64]);
        vmaskmovps(yb, ymask, ptr[src - 4]);
        vfmadd231ps(ysum, yb, yb);

        mov(c, J.C / 8 - 1);
        L(".lrn_loop");

        vmovups(yc, ptr[src]);
        vmovups(yd, ptr[src + 4]);
        vmovups(ye, ptr[src + 8]);
        vfmadd231ps(ysum, yc, yc);
        vfmadd231ps(ysum, yd, yd);
        vfmadd231ps(ysum, ye, ye);

        vmovups(ydst, ysum);
        vfmadd132ps(ydst, yone, yalpha); // ydst <- ysum*yalpha+yone

        vmovaps(ybase, ydst);
        vmulps(ydst, ydst, ydst);
        vmulps(ydst, ydst, ybase); // ydst = (ysum*yalpha+yone)^3;
        vsqrtps(ydst, ydst);
        vsqrtps(ydst, ydst); // ydst = (ysum*yalpha+yone)^0.75
        vmulps(ybase, ydst, ybase); // ybase = (ysum*yalpha+yone) ^ 1.75 -- for back prop
        if (pk != prop_kind::forward_scoring)
            vmovups(ptr[scratch], ybase);

        vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*yalpha+yone)^0.75
        vmovups(ptr[dst], ydst);

        vxorps(ysum, ysum, ysum);

        add(src, 32);
        add(dst, 32);
        if (pk != prop_kind::forward_scoring)
            add(scratch, 32);

        vmovups(ya, ptr[src - 8]);
        vfmadd231ps(ysum, ya, ya);
        vmovups(yb, ptr[src - 4]);
        vfmadd231ps(ysum, yb, yb);

        dec(c);
        cmp(c, 0);
        jne(".lrn_loop", T_NEAR);

        vmovups(yc, ptr[src]);
        vfmadd231ps(ysum, yc, yc);

        mov(imm_addr64, reinterpret_cast<size_t>(&mask[2]));
        vmovups(ymask, ptr[imm_addr64]);
        vmaskmovps(yd, ymask, ptr[src + 4]);
        vfmadd231ps(ysum, yd, yd); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

        mov(imm_addr64, reinterpret_cast<size_t>(&mask[3]));
        vmovups(ymask, ptr[imm_addr64]);
        vmaskmovps(ye, ymask, ptr[src + 8]);
        vfmadd231ps(ysum, ye, ye);

        vmovups(ydst, ysum);
        vfmadd132ps(ydst, yone, yalpha); // ydst <- ysum*yalpha+yone

        vmovaps(ybase, ydst);
        vmulps(ydst, ydst, ydst);
        vmulps(ydst, ydst, ybase); // ydst = (ysum*yalpha+yone)^3;
        vsqrtps(ydst, ydst);
        vsqrtps(ydst, ydst); // ydst = (ysum*yalpha+yone)^0.75
        vmulps(ybase, ydst, ybase); // ybase = (ysum*yalpha+yone) ^ 1.75 -- for back prop
        vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*yalpha+yone)^0.75

        vmovups(ptr[dst], ydst);
        if (pk != prop_kind::forward_scoring)
            vmovups(ptr[scratch], ybase);

        this->postamble();
    }

    void nchw_body(int tail, int HW, prop_kind_t pk,
        Xbyak::Ymm ymask,
        Xbyak::Ymm ya,
        Xbyak::Ymm yb,
        Xbyak::Ymm yc,
        Xbyak::Ymm yd,
        Xbyak::Ymm ye,
        Xbyak::Ymm ysum)
    {
        Xbyak::Ymm ydst = ymm14;
        Xbyak::Ymm ybase = ymm15;

        vfmadd231ps(ysum, ye, ye);

        vmovups(ydst, ysum);
        vfmadd132ps(ydst, yone, yalpha); // ydst <- ysum*yalpha+yone

        vmovaps(ybase, ydst);
        vmulps(ydst, ydst, ydst);
        vmulps(ydst, ydst, ybase); // ydst = (ysum*yalpha+yone)^3;
        vsqrtps(ydst, ydst);
        vsqrtps(ydst, ydst); // ydst = (ysum*yalpha+yone)^0.75
        vmulps(ybase, ydst, ybase); // ybase = (ysum*yalpha+yone) ^ 1.75 -- for back prop
        vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*yalpha+yone)^0.75

        if (tail != 0)
            vmaskmovps(ptr[dst], ymask, ydst);
        else
            vmovups(ptr[dst], ydst);

        if (pk != prop_kind::forward_scoring)
        {
            if (tail != 0)
                vmaskmovps(ptr[scratch], ymask, ybase);
            else
                vmovups(ptr[scratch], ybase);
        }

        vfnmadd231ps(ysum, ya, ya);
        vmovups(ya, yb);
        vmovups(yb, yc);
        vmovups(yc, yd);
        vmovups(yd, ye);
    }

    xbyak_lrn(
        struct nchw_across J,
        float A,
        prop_kind_t pk,
        void* code_ptr = nullptr,
        size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , alpha(A)
    {
        static const uint32_t mask[] = {
            0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
            0, 0, 0, 0, 0, 0, 0
        };
        Xbyak::Reg64 c = r10;
        Xbyak::Ymm ymask = ymm2;
        Xbyak::Ymm ye = ymm3;
        Xbyak::Ymm ya = ymm4;
        Xbyak::Ymm yb = ymm5;
        Xbyak::Ymm yc = ymm6;
        Xbyak::Ymm yd = ymm7;
        Xbyak::Ymm ysum = ymm8;

        this->preamble();

        if (J.tail != 0)
        {
            mov(imm_addr64, reinterpret_cast<size_t>(&mask[7 - J.tail]));
            vmovups(ymask, ptr[imm_addr64]);
        }
        mov(imm_addr64, reinterpret_cast<size_t>(&this->alpha));
        vbroadcastss(yalpha, ptr[imm_addr64]);
        mov(imm_addr64, reinterpret_cast<size_t>(&this->one));
        vbroadcastss(yone, ptr[imm_addr64]);

        mov(src, ptr[this->param1 + 0]);
        mov(dst, ptr[this->param1 + 8]);
        if (pk != prop_kind::forward_scoring)
            mov(scratch, ptr[this->param1 + 16]);

        vxorps(ya, ya, ya);
        vxorps(yb, yb, yb);
        if (J.tail != 0)
            vmaskmovps(yc, ymask, ptr[src + J.HW * 0]);
        else
            vmovups(yc, ptr[src + J.HW * 0]);
        if (J.tail != 0)
            vmaskmovps(yd, ymask, ptr[src + J.HW * 4]);
        else
            vmovups(yd, ptr[src + J.HW * 4]);

        vxorps(ysum, ysum, ysum);
        vfmadd231ps(ysum, yc, yc); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2
        vfmadd231ps(ysum, yd, yd);

        mov(c, J.C - 2);
        L(".lrn_loop");

        if (J.tail != 0)
            vmaskmovps(ye, ymask, ptr[src + J.HW * 8]);
        else
            vmovups(ye, ptr[src + J.HW * 8]);

        nchw_body(J.tail, J.HW, pk, ymask, ya, yb, yc, yd, ye, ysum);

        add(src, J.HW * 4);
        add(dst, J.HW * 4);
        if (pk != prop_kind::forward_scoring)
            add(scratch, J.HW * 4);
        dec(c);
        cmp(c, 0);
        jne(".lrn_loop", T_NEAR);

        vxorps(ye, ye, ye);

        nchw_body(J.tail, J.HW, pk, ymask, ya, yb, yc, yd, ye, ysum);
        add(src, J.HW * 4);
        add(dst, J.HW * 4);
        if (pk != prop_kind::forward_scoring)
            add(scratch, J.HW * 4);

        nchw_body(J.tail, J.HW, pk, ymask, ya, yb, yc, yd, ye, ysum);

        this->postamble();
    }
};

template <impl::precision_t prec>
const float jit_avx2_lrn<prec>::xbyak_lrn::one = 1.0f;

template <impl::precision_t prec>
jit_avx2_lrn<prec>::~jit_avx2_lrn() {
    delete this->jit;
    delete this->jit_first;
    delete this->jit_last;
}

template <impl::precision_t prec>
status_t jit_avx2_lrn<prec>::execute_forward() {
    auto src = reinterpret_cast<const data_t *>(
            this->input()[0].primitive->output()[
            this->input()[0].output_index]->memory_const());
    data_t *scratch = this->_is_training
        ? reinterpret_cast<data_t *>(this->input()[1].primitive->output()[
                this->input()[1].output_index]->memory())
        : nullptr;
    auto dst = reinterpret_cast<data_t *>(this->output()[0]->memory());

    const memory_desc_wrapper src_d(this->_lpd.src_primitive_desc.memory_desc);

    const int N = src_d.dims()[0];
    const int C = src_d.dims()[1];
    const int HW = src_d.dims()[2]*src_d.dims()[3];

    if (this->_lpd.dst_primitive_desc.memory_desc.format == nChw8c
        && this->_lpd.lrn_desc.alg_kind == mkldnn_lrn_across_channels
        && this->_lpd.lrn_desc.local_size == 5)
    {
#       pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < N; ++n) {
            for (int c8 = 0; c8 < C / VECTOR_LENGTH; ++c8) {
                jit_args_t args;
                args.src = &src[n*HW*C + c8 * HW * VECTOR_LENGTH];
                args.dst = &dst[n*HW*C + c8 * HW * VECTOR_LENGTH];
                args.scratch = &scratch[n*HW*C + c8 * HW * VECTOR_LENGTH];
                if (c8 == 0)
                    ker_first(&args);
                else if (c8 == C / VECTOR_LENGTH - 1)
                    ker_last(&args);
                else
                    ker(&args);
            }
        }
    }
    else if (this->_lpd.dst_primitive_desc.memory_desc.format == nChw8c
        && this->_lpd.lrn_desc.alg_kind == mkldnn_lrn_within_channel)
    {
#       pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < N; ++n) {
            for (int c8 = 0; c8 < C / VECTOR_LENGTH; ++c8) {
                jit_args_t args;
                args.src = &src[n*HW*C + c8 * HW * VECTOR_LENGTH];
                args.dst = &dst[n*HW*C + c8 * HW * VECTOR_LENGTH];
                args.scratch = &scratch[n*HW*C + c8 * HW * VECTOR_LENGTH];
                ker(&args);
            }
        }
    }
    else if (this->_lpd.dst_primitive_desc.memory_desc.format == nchw
        && this->_lpd.lrn_desc.alg_kind == mkldnn_lrn_across_channels
        && this->_lpd.lrn_desc.local_size == 5)
    {
#       pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < N; ++n) {
            for (int hw8 = 0; hw8 < (HW + VECTOR_LENGTH - 1) / VECTOR_LENGTH; ++hw8) {
                jit_args_t args;
                args.src = &src[n*HW*C + hw8 * VECTOR_LENGTH];
                args.dst = &dst[n*HW*C + hw8 * VECTOR_LENGTH];
                args.scratch = &scratch[n*HW*C + hw8 * VECTOR_LENGTH];
                if ((hw8+1)*VECTOR_LENGTH > HW)
                    ker_last(&args);
                else
                    ker(&args);
            }
        }
    }
    else // nhwc
    {
#       pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < N; ++n) {
            for (int hw = 0; hw < HW; ++hw) {
                jit_args_t args;
                args.src = &src[n*HW*C + hw * C];
                args.dst = &dst[n*HW*C + hw * C];
                args.scratch = &scratch[n*HW*C + hw * C];
                ker(&args);
            }
        }
    }

    return success;
}

template <impl::precision_t prec>
status_t jit_avx2_lrn<prec>::set_default_parameters(lrn_desc_t &lrn_d) {
    if (lrn_d.src_desc.format == any)
        CHECK(types::set_default_format<prec>(lrn_d.src_desc, nChw8c));
    if (lrn_d.dst_desc.format == any)
        CHECK(types::set_default_format<prec>(lrn_d.dst_desc, nChw8c));
    return status::success;
}

template <impl::precision_t prec>
status_t jit_avx2_lrn<prec>::constraint(const lrn_desc_t &lrn_d) {
    const memory_desc_wrapper src_d(lrn_d.src_desc);

    bool args_ok_common = true
        && one_of(lrn_d.prop_kind, prop_kind::forward_training,
        prop_kind::forward_scoring)
        && src_d.ndims() == 4
        && src_d.dims()[1] % VECTOR_LENGTH == 0
        && src_d.dims()[1] >= 2 * VECTOR_LENGTH
        && lrn_d.beta == 0.75
        && lrn_d.dst_desc.format == src_d.format();
    if (!args_ok_common) return unimplemented;

    bool args_ok_across = true
        && lrn_d.alg_kind == alg_kind::lrn_across_channels
        && lrn_d.local_size == 5
        && one_of(src_d.format(), nChw8c, nchw, nhwc);

    bool args_ok_within = true
        && lrn_d.alg_kind == alg_kind::lrn_within_channel
        && lrn_d.local_size <= MAX_LOCAL_SIZE
        && src_d.dims()[2] >= lrn_d.local_size
        && src_d.dims()[3] >= lrn_d.local_size
        && one_of(src_d.format(), nChw8c);

    return args_ok_across || args_ok_within ? success : unimplemented;
}

template <impl::precision_t prec>
const primitive_impl jit_avx2_lrn<prec>::implementation = {
    jit_avx2_lrn<prec>::create
};

template class jit_avx2_lrn<precision::f32>;

template <impl::precision_t prec>
jit_avx2_lrn<prec>::jit_avx2_lrn(const lrn_primitive_desc_t &ppd,
    const primitive_at_t *inputs, const primitive *outputs[])
    : lrn<jit_avx2_lrn<prec>>(ppd, inputs, outputs)
{
    int C = ppd.src_primitive_desc.memory_desc.tensor_desc.dims[1];
    int H = ppd.src_primitive_desc.memory_desc.tensor_desc.dims[2];
    int W = ppd.src_primitive_desc.memory_desc.tensor_desc.dims[3];
    double A = ppd.lrn_desc.alpha / ppd.lrn_desc.local_size;
    mkldnn_prop_kind_t pk = ppd.lrn_desc.prop_kind;
    mkldnn_alg_kind_t ak = ppd.lrn_desc.alg_kind;

    if (ppd.src_primitive_desc.memory_desc.format == nChw8c && ppd.lrn_desc.local_size == 5 && ak == mkldnn_lrn_across_channels)
    {
        this->jit = new xbyak_lrn(nchw8c_across(H*W, 0), A, pk);
        this->jit_first = new xbyak_lrn(nchw8c_across(H*W, -1), A, pk);
        this->jit_last = new xbyak_lrn(nchw8c_across(H*W, +1), A, pk);
    }
    else if (ppd.src_primitive_desc.memory_desc.format == nChw8c && ak == mkldnn_lrn_within_channel) // RCNN case
    {
        A /= ppd.lrn_desc.local_size; // within channel, local_size (x) local_size
        this->jit = new xbyak_lrn(nchw8c_within(H, W, ppd.lrn_desc.local_size), A, pk);
        this->jit_first = nullptr;
        this->jit_last = nullptr;
    }
    else if (ppd.src_primitive_desc.memory_desc.format == nchw && ak == mkldnn_lrn_across_channels)
    {
        this->jit = new xbyak_lrn(nchw_across(C, H*W, 0), A, pk);
        this->jit_first = nullptr;
        this->jit_last = (H*W) % VECTOR_LENGTH == 0 ? nullptr : new xbyak_lrn(nchw_across(C, H*W, (H*W) % VECTOR_LENGTH), A, pk);
    }
    else // nhwc
    {
        this->jit = new xbyak_lrn(nhwc_across(C), A, pk);
        this->jit_first = nullptr;
        this->jit_last = nullptr;
    }
    typedef void(*kernel_t)(const void*);
    this->ker = this->jit == nullptr ? nullptr : reinterpret_cast<kernel_t>(const_cast<uint8_t*>(this->jit->getCode()));
    this->ker_first = this->jit_first == nullptr ? nullptr : reinterpret_cast<kernel_t>(const_cast<uint8_t*>(this->jit_first->getCode()));
    this->ker_last = this->jit_last == nullptr ? nullptr : reinterpret_cast<kernel_t>(const_cast<uint8_t*>(this->jit_last->getCode()));
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
