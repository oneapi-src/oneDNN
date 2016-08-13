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

#ifndef CPU_JIT_AVX2_GENERATOR_HPP
#define CPU_JIT_AVX2_GENERATOR_HPP

#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"
#include "utils_xbyak.hpp"

namespace mkl_dnn { namespace impl { namespace cpu {

using namespace mkl_dnn::impl::status;
using namespace mkl_dnn::impl::prop_kind;
using namespace mkl_dnn::impl::primitive_kind;

class jit_avx2_generator : public Xbyak::CodeGenerator
{
private:
    Xbyak::Reg64 param          = Xbyak::util::cdecl_param1;
    Xbyak::Reg64 reg_input      = rax;
    Xbyak::Reg64 aux_reg_input  = r8;
    Xbyak::Reg64 reg_kernel     = rdx;
    Xbyak::Reg64 aux_reg_kernel = r9;
    Xbyak::Reg64 reg_output     = rsi;

    Xbyak::Reg64 kj      = r10;
    Xbyak::Reg64 oi_iter = r11;
    Xbyak::Reg64 reg_kh  = rcx;

    inline void hsw_iter(jit_convolution_param_t *params, uint32_t ur_w,
                         int pad_l, int pad_r, const char* kh_lable)
    {
        using Xbyak::Ymm;

        uint32_t iw    = params->iw;
        uint32_t kw    = params->kw;
        uint32_t kh    = params->kh;
        uint32_t ow    = params->ow;
        uint32_t oh    = params->oh;
        uint32_t nb_ic = params->nb_ic;

        uint32_t stride_w = params->stride_w;
        uint32_t nb_oc_block = params->nb_oc_blocking;

        for (uint32_t ii = 0; ii < nb_oc_block; ii++)
            for (uint32_t jj = 0; jj  < ur_w; jj++)
                vmovups(Ymm(ur_w*ii+jj), YWORD[reg_output + sizeof(float)*(ii*oh*ow+jj)*params->oc_block]);

        mov(aux_reg_input , reg_input );
        mov(aux_reg_kernel, reg_kernel);

        mov(kj, reg_kh);
        L(kh_lable); {
            for (uint32_t ki = 0; ki < kw; ki++) {
                uint32_t jj_start = (uint32_t)nstl::max(0, pad_l-(int)ki);
                uint32_t jj_end = ur_w - (uint32_t)nstl::max(0, (int)ki+pad_r - (int)(kw-1));
                for (uint32_t ifm2 = 0; ifm2 < params->ic_block; ifm2++) {
                    for (uint32_t jj =jj_start ; jj < jj_end; jj++) {
                        int aux_input_offset = (ki+jj*stride_w-pad_l)*params->ic_block + ifm2;
                        vbroadcastss(Ymm(nb_oc_block*ur_w+jj), ptr [ aux_reg_input + sizeof(float)*aux_input_offset ]);
                    }
                    for (uint32_t ii = 0; ii < nb_oc_block; ii++) {
                        int aux_kernel_offset = ii*nb_ic*kh*kw*params->ic_block*params->oc_block +
                                                 ki*params->ic_block*params->oc_block+ ifm2*params->oc_block;
                        vmovups(ymm15, ptr [ aux_reg_kernel +  sizeof(float)*aux_kernel_offset ]);
                        for (uint32_t jj = jj_start; jj  < jj_end; jj++)
                            vfmadd231ps(Ymm(ur_w*ii+jj), Ymm(nb_oc_block*ur_w+jj), ymm15);
                    }
                }
            }
            add(aux_reg_kernel, sizeof(float)*kw *params->oc_block*params->ic_block);
            add(aux_reg_input,  sizeof(float)*iw*params->ic_block);

            dec(kj);
            cmp(kj, 0);
            jg(kh_lable, T_NEAR);
        }

        for (uint32_t ii = 0; ii < nb_oc_block; ii++)
            for (uint32_t jj = 0; jj  < ur_w; jj++)
                vmovups(YWORD [ reg_output + sizeof(float)*(ii*oh*ow+jj)*params->oc_block ], Ymm(ur_w*ii+jj));
    }

public:

    void preamble() {
        using Xbyak::util::reg_to_preserve;
        const size_t nregs = sizeof(reg_to_preserve)/sizeof(reg_to_preserve[0]);
        for (size_t i = 0; i < nregs; ++i)
            push(Xbyak::Reg64(reg_to_preserve[i]));
    }
    void postamble() {
        using Xbyak::util::reg_to_preserve;
        const size_t nregs = sizeof(reg_to_preserve)/sizeof(reg_to_preserve[0]);
        for (size_t i = 0; i < nregs; ++i)
            pop(Xbyak::Reg64(reg_to_preserve[nregs - 1 - i]));
        ret();
    }

    jit_avx2_generator(
        jit_convolution_param_t *params,
        void* code_ptr = nullptr,
        size_t code_size = 8 * Xbyak::DEFAULT_MAX_CODE_SIZE
        ) : Xbyak::CodeGenerator(code_size, code_ptr)
    {
        using Xbyak::Ymm;
        preamble();

        mov(reg_input , ptr [ param ]);
        mov(reg_output, ptr [ param + 8]);
        mov(reg_kernel, ptr [ param + 16]);
        mov(reg_kh    , ptr [ param + 48]);

        // NB: works only for params->ur_w == 3 && params->nb_oc % 4 == 0
        uint32_t n_oi = params->ow / params->ur_w;
        xor_(oi_iter, oi_iter);
        int l_pad = params->l_pad;
        int r_pad = nstl::max(0, (int)((params->ow-1)*params->stride_w + params->kw - 1 - (params->iw + params->l_pad - 1 )));
        int r_pad1 = 0;
        if (l_pad > 0) {
            hsw_iter(params, params->ur_w, params->l_pad, 0, ".kh_loop_oimain_padwl");
            add(reg_input,  sizeof(float)*(params->ur_w*params->stride_w-params->l_pad)*params->ic_block);
            add(reg_output,  sizeof(float)*params->ur_w*params->oc_block);
            inc(oi_iter);

            r_pad1 = (params->ur_w*n_oi - 1)*params->stride_w + params->kw -1 - (params->iw + params->l_pad - 1);
            if (r_pad1 > 0)
               n_oi--;
        }

        if ((l_pad <= 0 && n_oi > 0)
          ||(l_pad >  0 && n_oi > 1)) {
            L(".ow_loop"); {
                hsw_iter(params, params->ur_w, 0, 0, ".kh_loop_oimain");
                add(reg_input,  sizeof(float)*params->ur_w*params->stride_w*params->ic_block);
                add(reg_output,  sizeof(float)*params->ur_w*params->oc_block);

                inc(oi_iter);
                cmp(oi_iter, n_oi); jl(".ow_loop", T_NEAR);
            } L(".ow_loop_end");
        }

        if (r_pad1 > 0 ) {
            hsw_iter(params, params->ur_w, 0, r_pad1, ".kh_loop_oimain_padwr");
            add(reg_input,  sizeof(float)*params->ur_w*params->stride_w*params->ic_block);
            add(reg_output,  sizeof(float)*params->ur_w*params->oc_block);
        }

        if (params->ur_w_tail != 0)
            hsw_iter(params, params->ur_w_tail, 0, r_pad, ".kh_loop_oitail");

        postamble();
        return;
    }
};

}}}

#endif
