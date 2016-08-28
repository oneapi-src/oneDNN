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
#define ymm_input Ymm(14)
#define ymm_tmp Ymm(13)
#define xmm_tmp Xmm(13)
#define ymm_index Ymm(12)
#define xmm_index Xmm(12)
#define ymm_c_block Ymm(11)
#define xmm_c_block Xmm(11)
#define ymm_c_block_stride_w Ymm(10)
#define xmm_c_block_stride_w Xmm(10)
#define ymm_ki_offset Ymm(9)
#define xmm_ki_offset Xmm(9)
#define ymm_ji_offset Ymm(8)
#define xmm_ji_offset Xmm(8)
#define ymm_offset_base Ymm(7)
#define xmm_offset_base Xmm(7)

namespace mkldnn {
namespace impl {
namespace cpu {

inline void jit_avx2_pooling_generator_f32::oh_step(
    jit_pooling_param_t *params, int ur_w,
    int pad_l, int pad_r, const char* kh_lable)
{
    using Xbyak::Ymm;
    using Xbyak::Xmm;

    unsigned char _cmp = 1;
    union {
        float _flt_max;
        int32_t _flt_max_int;
    } cvt;
    cvt._flt_max = -FLT_MAX;

    int iw = params->iw;
    int kw = params->kw;
    int stride_w = params->stride_w;
    int c_block = params->c_block;

    vpxor(ymm_store_mask, ymm_store_mask);

    mov(tmp_gpr, cvt._flt_max_int);
    movq(xmm_tmp, tmp_gpr);
    vbroadcastss(ymm_tmp, xmm_tmp);
    for (int jj = 0; jj < ur_w; jj++)
        vmovaps(Ymm(jj), ymm_tmp);

    mov(aux_reg_input , reg_input);
    xor_(kj, kj);
    L(kh_lable); {
        if (this->_is_training) {
            vpxor(ymm_ki_offset, ymm_ki_offset);
        }
        for (int ki = 0; ki < kw; ki++) {
            int jj_start = nstl::max(0, pad_l - ki);
            int jj_end   = (int)ur_w -
                nstl::max(0, ki + pad_r - (kw-1));
            if (this->_is_training) {
                vmovaps(ymm_index, ymm_ki_offset);
                vmovaps(ymm_ji_offset, ymm_offset_base);
                if (jj_start != 0) {
                    mov(tmp_gpr,(jj_start * stride_w * c_block));
                    movq(xmm_tmp, tmp_gpr);
                    vpbroadcastd(ymm_tmp, xmm_tmp);
                    vpaddd(ymm_ji_offset, ymm_ji_offset, ymm_tmp);
                }
            }
            for (int jj = jj_start; jj  < jj_end; jj++) {
                int aux_input_offset = (ki+jj*stride_w-pad_l)* c_block;
                if (aux_input_offset > iw * c_block)
                    continue;
                if (this->_is_training) {
                    vpaddd(ymm_index, ymm_ki_offset, ymm_ji_offset);
                }
                vmovups(ymm_input,
                    ptr [ aux_reg_input + sizeof(float)*aux_input_offset ]);
                vcmpps(ymm_store_mask, Ymm(jj), ymm_input, _cmp);
                vblendvps(Ymm(jj), Ymm(jj), ymm_input, ymm_store_mask);
                if (this->_is_training) {
                    vblendvps(Ymm(ur_w+jj), Ymm(ur_w+jj), ymm_index,
                        ymm_store_mask);
                    vpaddd(ymm_ji_offset, ymm_ji_offset , ymm_c_block_stride_w);
                }
            }
            if (this->_is_training) {
                vpaddd(ymm_ki_offset, ymm_ki_offset , ymm_c_block);
            }
        }
        add(aux_reg_input,  sizeof(float) * iw * c_block);
        inc(kj);
        cmp(kj, reg_kh);
        jl(kh_lable, T_NEAR);
    }

    for (int jj = 0; jj < ur_w; jj++) {
        vmovups(YWORD[reg_output + sizeof(float)*jj*c_block], Ymm(jj));
        if (this->_is_training)
            vmovdqa(YWORD[reg_index + sizeof(int)*jj*c_block], Ymm(ur_w+jj));
    }
}

jit_avx2_pooling_generator_f32::jit_avx2_pooling_generator_f32(
        jit_pooling_param_t *params, bool is_training, void* code_ptr,
        size_t code_size)
    : jit_generator(code_ptr, code_size)
    , _is_training(is_training)
{
    using Xbyak::Ymm;
    this->preamble();

    int ow = params->ow;
    int iw = params->iw;
    int kw = params->kw;
    int ur_w = params->ur_w;
    int c_block = params->c_block;
    int stride_w = params->stride_w;
    int l_pad = params->l_pad;
    int ur_w_tail = params->ur_w_tail;

    int n_oi = ow / ur_w;

    mov(reg_input , ptr [ this->param1 ]);
    mov(reg_output, ptr [ this->param1 + 8]);
    if (this->_is_training)
        mov(reg_index , ptr [ this->param1 + 16]);
    mov(reg_kh    , ptr [ this->param1 + 48]);
    if (this->_is_training)
        mov(reg_arr_init, ptr [ this->param1 + 80]);

    if (this->_is_training) {
        mov(tmp_gpr,c_block);
        movq(xmm_c_block, tmp_gpr);
        vpbroadcastd(ymm_c_block, xmm_c_block);

        mov(tmp_gpr,(stride_w * c_block));
        movq(xmm_c_block_stride_w, tmp_gpr);
        vpbroadcastd(ymm_c_block_stride_w, xmm_c_block_stride_w);

        vmovdqu(ymm_offset_base, ptr [ reg_arr_init ]);
        if (l_pad > 0) {
            mov(tmp_gpr,(l_pad * c_block));
            movq(xmm_tmp, tmp_gpr);
            vpbroadcastd(ymm_tmp, xmm_tmp);
            vpsubd(ymm_offset_base, ymm_offset_base, ymm_tmp);
        }
    }

    int r_pad  = nstl::max(0, ((ow-1)*stride_w) + kw - 1 - (iw + l_pad - 1 ));
    int r_pad1 = (ur_w*n_oi - 1)*stride_w + kw - 1 - (iw + l_pad - 1);
    if (r_pad1 > 0) n_oi--;

    if (l_pad > 0) {
        n_oi--;
        if (n_oi < 0 && r_pad1 > 0) {
            oh_step(params, ur_w, l_pad, r_pad1, ".kh_loop_oimain_padwl");
        } else  {
            oh_step(params, ur_w, l_pad, 0, ".kh_loop_oimain_padwl");
        }

        add(reg_input,  sizeof(float)*(ur_w*stride_w - l_pad)*c_block);
        add(reg_output,  sizeof(float)*ur_w*c_block);
        if (this->_is_training)
            add(reg_index, sizeof(int)*ur_w*c_block);
    }

    xor_(oi_iter, oi_iter);
    if (n_oi > 0) {
        L(".ow_loop"); {
            oh_step(params, ur_w, 0, 0, ".kh_loop_oimain");
            add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
            add(reg_output, sizeof(float)*ur_w*c_block);
            if (this->_is_training)
                add(reg_index, sizeof(int)*ur_w*c_block);

            inc(oi_iter);
            cmp(oi_iter, n_oi); jl(".ow_loop", T_NEAR);
        } L(".ow_loop_end");
    }

    if (r_pad1 > 0 && n_oi >= 0) {
        oh_step(params, ur_w, 0, r_pad1, ".kh_loop_oimain_padwr");
        add(reg_input, sizeof(float)*ur_w*stride_w*c_block);
        add(reg_output, sizeof(float)*ur_w*c_block);
        if (this->_is_training)
            add(reg_index, sizeof(int) * ur_w * c_block);
    }

    if (ur_w_tail != 0)
        oh_step(params, ur_w_tail, 0, r_pad, ".kh_loop_oitail");

    this->postamble();
    return;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
