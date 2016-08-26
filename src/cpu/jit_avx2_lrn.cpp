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

enum { VECTOR_LENGTH = 8 };
typedef struct {
    const float *src;
    float *dst, *scratch;
} jit_args_t;

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

    xbyak_lrn(
        float A,
        uint32_t HW,
        int version, // -1 channels 0..7, 1 channels C-8 .. C-1, 0 -- other channels
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

        if (HW == 0)
        {
            ret();
            return;
        }
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
        if (version == -1)
        {
            vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
            vmovups(ptr[t + 0], xsrc_prev);
        }
        if (version == +1)
        {
            vxorps(xsrc_next, xsrc_next, xsrc_next);
            vmovups(ptr[t + 48], xsrc_next);
        }

        mov(hw, HW);
        L(".lrn_loop");

        if (version != -1) vmovups(xsrc_prev, ptr[src - HW * 32+16]);
        vmovups(ysrc, ptr[src]);
        if (version != +1) vmovups(xsrc_next, ptr[src + HW * 32]);

        if (version != -1) vmovups(ptr[t + 0], xsrc_prev);
        vmovups(ptr[t + 16], ysrc);
        if (version != +1) vmovups(ptr[t + 48], xsrc_next);

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
        float A,
        uint32_t C,
        prop_kind_t pk,
        void *code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , alpha(A)
    {
        assert(C >= 16 && C % 8 == 0);
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

        mov(c, C/8-1);
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

    void nchw_body(int tail, uint32_t HW, prop_kind_t pk,
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
        float A,
        uint32_t C,
        uint32_t HW,
        int tail, // 0 -- no tail, 1 .. 7 -- read/write only that many elements in ymm's
        prop_kind_t pk,
        void* code_ptr = nullptr,
        size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , alpha(A)
    {
            assert(C >= 16);
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

            if (tail != 0)
            {
                mov(imm_addr64, reinterpret_cast<size_t>(&mask[7 - tail]));
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
            if (tail != 0)
                vmaskmovps(yc, ymask, ptr[src + HW * 0]);
            else
                vmovups(yc, ptr[src + HW * 0]);
            if (tail != 0)
                vmaskmovps(yd, ymask, ptr[src + HW * 4]);
            else
                vmovups(yd, ptr[src + HW * 4]);

            vxorps(ysum, ysum, ysum);
            vfmadd231ps(ysum, yc, yc); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2
            vfmadd231ps(ysum, yd, yd);

            mov(c, C - 2);
            L(".lrn_loop");

            if (tail != 0)
                vmaskmovps(ye, ymask, ptr[src + HW * 8]);
            else
                vmovups(ye, ptr[src + HW * 8]);

            nchw_body(tail, HW, pk, ymask, ya, yb, yc, yd, ye, ysum);

            add(src, HW * 4);
            add(dst, HW * 4);
            if (pk != prop_kind::forward_scoring)
                add(scratch, HW * 4);
            dec(c);
            cmp(c, 0);
            jne(".lrn_loop", T_NEAR);

            vxorps(ye, ye, ye);

            nchw_body(tail, HW, pk, ymask, ya, yb, yc, yd, ye, ysum);
            add(src, HW * 4);
            add(dst, HW * 4);
            if (pk != prop_kind::forward_scoring)
                add(scratch, HW * 4);

            nchw_body(tail, HW, pk, ymask, ya, yb, yc, yd, ye, ysum);

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

    const uint32_t C = src_d.dims()[1];
    const uint32_t HW = src_d.dims()[2]*src_d.dims()[3];

    const uint32_t N = src_d.dims()[0];
    if (this->_lpd.dst_primitive_desc.memory_desc.format == nChw8c)
    {
#       pragma omp parallel for collapse(2) schedule(static)
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t c8 = 0; c8 < C / VECTOR_LENGTH; ++c8) {
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
    else if (this->_lpd.dst_primitive_desc.memory_desc.format == nchw)
    {
#       pragma omp parallel for collapse(2) schedule(static)
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t hw8 = 0; hw8 < (HW + VECTOR_LENGTH - 1) / VECTOR_LENGTH; ++hw8) {
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
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t hw = 0; hw < HW; ++hw) {
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

    bool args_ok = true
        && one_of(lrn_d.prop_kind, prop_kind::forward_training,
                prop_kind::forward_scoring)
        && lrn_d.alg_kind == alg_kind::lrn_across_channels
        && src_d.ndims() == 4
        && src_d.dims()[1] % VECTOR_LENGTH == 0
        && src_d.dims()[1] >= 2*VECTOR_LENGTH
        && lrn_d.beta == 0.75
        && lrn_d.local_size == 5
        && one_of(src_d.format(), nChw8c, nchw, nhwc)
        && lrn_d.dst_desc.format == src_d.format();

    return args_ok ? success : unimplemented;
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
    uint32_t H = ppd.src_primitive_desc.memory_desc.tensor_desc.dims[2];
    uint32_t W = ppd.src_primitive_desc.memory_desc.tensor_desc.dims[3];
    uint32_t C = ppd.src_primitive_desc.memory_desc.tensor_desc.dims[1];
    float A = ppd.lrn_desc.alpha / ppd.lrn_desc.local_size;

    if (ppd.src_primitive_desc.memory_desc.format == nChw8c)
    {
        this->jit = new xbyak_lrn(A, H*W, 0, ppd.lrn_desc.prop_kind);
        this->jit_first = new xbyak_lrn(A, H*W, -1, ppd.lrn_desc.prop_kind);
        this->jit_last = new xbyak_lrn(A, H*W, +1, ppd.lrn_desc.prop_kind);
    }
    else if (ppd.src_primitive_desc.memory_desc.format == nchw)
    {
        this->jit = new xbyak_lrn(A, C, H*W, 0, ppd.lrn_desc.prop_kind);
        this->jit_first = nullptr;
        this->jit_last = new xbyak_lrn(A, C, H*W, (H*W) % VECTOR_LENGTH, ppd.lrn_desc.prop_kind);
    }
    else // nhwc
    {
        this->jit = new xbyak_lrn(A, C, ppd.lrn_desc.prop_kind);
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
