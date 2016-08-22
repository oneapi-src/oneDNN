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
#include "c_types_map.hpp"
#include "nstl.hpp"

#include "jit_avx2_pooling_generator_f32.hpp"

#define ymm_store_mask Ymm(15)
#define ymm_input      Ymm(14)

#define ymm_index      Ymm(13)
#define xmm_index      Xmm(13)
#define ymm_iw_simd    Ymm(12)
#define xmm_iw_simd    Xmm(12)
#define ymm_simd       Ymm(11)
#define xmm_simd       Xmm(11)
#define ymm_stride_w   Ymm(10)
#define xmm_stride_w   Xmm(10)

#define ymm_tmp        Ymm(9)
#define xmm_tmp        Xmm(9)
#define ymm_init_reg   Ymm(8)
#define xmm_init_reg   Xmm(8)

namespace mkldnn {
namespace impl {
namespace cpu {

inline void jit_avx2_pooling_generator_f32::oh_step(
    jit_pooling_param_t *params, uint32_t ur_w,
    int pad_l, int pad_r, const char* kh_lable)
{
    using Xbyak::Ymm;
    using Xbyak::Xmm;

    unsigned char _cmp = 1;
    float _flt_min = -FLT_MIN;

    uint32_t IW = params->iw;
    uint32_t KW = params->kw;
    uint32_t stride_w = params->stride_w;

    vpxor(ymm_store_mask, ymm_store_mask);
    // Init output
    mov(tmp_gpr, _flt_min);
    movq(xmm_init_reg, tmp_gpr);
    vbroadcastss(ymm_init_reg, xmm_init_reg);    // output

    for (uint32_t jj = 0; jj < ur_w; jj++)
        vmovaps(Ymm(jj), ymm_init_reg);

    mov(aux_reg_input , reg_input);
    xor_(kj, kj);
    L(kh_lable); {
        vpaddd(ymm_index, ymm_index, ymm_iw_simd);
        for (uint32_t ki = 0; ki < KW; ki++) {
            uint32_t jj_start = (uint32_t)nstl::max(0, pad_l-(int)ki);
            uint32_t jj_end   = ur_w -
                (uint32_t)nstl::max(0, (int)ki+pad_r - (int)(KW-1));
            vpaddd(ymm_index, ymm_index, ymm_simd);
            for (uint32_t jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)*params->c_block;
                if (aux_input_offset > (int)IW*(int)params->c_block)
                    continue;
                vmovups(ymm_input,
                    ptr [ aux_reg_input + sizeof(float)*aux_input_offset ]);
                vpaddd(ymm_index, ymm_index, ymm_tmp);
                vcmpps(ymm_store_mask, Ymm(jj), ymm_input, _cmp);
                vblendvps(Ymm(jj), Ymm(jj), ymm_input, ymm_store_mask);
                vblendvps(Ymm(ur_w+jj), Ymm(ur_w+jj), ymm_index, ymm_store_mask);

            }
        }
        add(aux_reg_input,  sizeof(float)*IW*params->c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_lable, T_NEAR);
    }
    for (uint32_t jj = 0; jj < ur_w; jj++) {
        vmovups(YWORD[reg_output +
            sizeof(float)*jj*params->c_block], Ymm(jj));
        vmovdqa(YWORD[reg_index +
            sizeof(uint32_t)*jj*params->c_block], Ymm(ur_w+jj));
    }
}

jit_avx2_pooling_generator_f32::jit_avx2_pooling_generator_f32(
    jit_pooling_param_t *params, void* code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size)
{
    using Xbyak::Ymm;
    this->preamble();

    uint32_t n_oi = params->ow / params->ur_w;

    mov(reg_input , ptr [ this->param1 ]);
    mov(reg_output, ptr [ this->param1 + 8]);
    mov(reg_index , ptr [ this->param1 + 16]);
    mov(reg_kh    , ptr [ this->param1 + 48]);
    mov(reg_arr_init, ptr [ this->param1 + 80]);

    vmovdqu(ymm_index, ptr [ reg_arr_init ]); // array init

    if (params->l_pad > 0) {
        mov(tmp_gpr,(params->l_pad*params->c_block));
        movq(xmm_tmp, tmp_gpr);
        vpbroadcastd(ymm_tmp, xmm_tmp);
        vpsubd(ymm_index, ymm_index, ymm_tmp);
    } else {
        vpxor(ymm_index, ymm_index);
    }
    mov(tmp_gpr,(params->iw*params->c_block));
    movq(xmm_iw_simd, tmp_gpr);
    vpbroadcastd(ymm_iw_simd, xmm_iw_simd);

    mov(tmp_gpr,(params->c_block));
    movq(xmm_simd, tmp_gpr);
    vpbroadcastd(ymm_simd, xmm_simd);

    mov(tmp_gpr,(params->stride_w));
    movq(xmm_stride_w, tmp_gpr);
    vpbroadcastd(ymm_stride_w, xmm_stride_w);

    vpmuldq(ymm_tmp, ymm_stride_w, ymm_simd);

    int r_pad  = nstl::max(0, (int)((params->ow-1)*params->stride_w) +
        (int)params->kw - 1 - (int)(params->iw + params->l_pad - 1 ));
    int r_pad1 = 0;

    xor_(oi_iter, oi_iter);
    if (params->l_pad > 0) {
        oh_step(params, params->ur_w, params->l_pad, 0,
                  ".kh_loop_oimain_padwl");

        add(reg_input,  sizeof(float)*(params->ur_w*params->stride_w -
                               params->l_pad)*params->c_block);
        add(reg_output,  sizeof(float)*params->ur_w*params->c_block);
        add(reg_index, sizeof(uint32_t)*params->ur_w*params->c_block);
        inc(oi_iter);

        r_pad1 = (params->ur_w*n_oi - 1)*params->stride_w +
            params->kw - 1 - (params->iw + params->l_pad - 1);
        if (r_pad1 > 0) n_oi--;
    }

    if ((params->l_pad <= 0 && n_oi > 0)
      ||(params->l_pad >  0 && n_oi > 1)) {
        L(".ow_loop"); {
            oh_step(params, params->ur_w, 0, 0, ".kh_loop_oimain");
            add(reg_input,
               sizeof(float)*params->ur_w*params->stride_w*params->c_block);
            add(reg_output, sizeof(float)*params->ur_w*params->c_block);
            add(reg_index, sizeof(uint32_t)*params->ur_w*params->c_block);

            inc(oi_iter);
            cmp(oi_iter, n_oi); jl(".ow_loop", T_NEAR);
        } L(".ow_loop_end");
    }

    if (r_pad1 > 0 ) {
        oh_step(params, params->ur_w, 0, r_pad1, ".kh_loop_oimain_padwr");
        add(reg_input,
               sizeof(float)*params->ur_w*params->stride_w*params->c_block);
        add(reg_output,sizeof(float)*params->ur_w*params->c_block);
        add(reg_index, sizeof(uint32_t) * params->ur_w * params->c_block);
    }

    if (params->ur_w_tail != 0)
        oh_step(params, params->ur_w_tail, 0, r_pad, ".kh_loop_oitail");

    this->postamble();
    return;
}

}
}
}

#undef ymm_store_mask
#undef ymm_input

#undef ymm_index
#undef xmm_index
#undef ymm_iw_simd
#undef xmm_iw_simd
#undef ymm_simd
#undef xmm_simd
#undef ymm_stride_w
#undef xmm_stride_w

#undef ymm_tmp
#undef xmm_tmp
#undef ymm_init_reg
#undef xmm_init_reg
