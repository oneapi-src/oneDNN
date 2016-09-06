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
#include "type_helpers.hpp"

#include "jit_avx2_batch_norm_generator_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

inline void jit_avx2_batch_norm_generator_f32::mean_compute(int block_size,
    jit_batch_normalization_param_t *params)
{
    using Xbyak::Ymm;

    int block_8 = block_size / 8;
    int block_tail = block_size % 8;
    for (int i = 0; i < block_8; i++) {
        for (int j = 0; j < 8; j++) {
           vaddps(Ymm(j), Ymm(j), ptr [aux_ptr  +
                ((i * 8) + j) * params->c_block * sizeof(float)]);
        }
    }
    for (int j = 0; j < block_tail; j++) {
           vaddps(Ymm(j), Ymm(j), ptr [aux_ptr +
                (block_8*8 + j) * params->c_block * sizeof(float)]);
    }
}

inline void jit_avx2_batch_norm_generator_f32::variance_compute(int block_size,
    jit_batch_normalization_param_t *params)
{
    using Xbyak::Ymm;

    int block_4 = block_size / 4;
    int block_tail = block_size % 4;

    for (int i = 0; i < block_4; i++) {
        for (int j = 0; j < 4; j++) {
            vmovups(Ymm(j+4), ptr [aux_ptr +
                ((i * 4) + j) * params->c_block * sizeof(float)]);
            vsubps(Ymm(j+4), Ymm(j+4), ymm_mean);
            vfmadd231ps(Ymm(j), Ymm(j+4), Ymm(j+4));
        }
    }
    for (int j = 0; j < block_tail; j++) {
       vmovups(Ymm(j+4), ptr [aux_ptr +
            ((block_4 * 4) + j) * params->c_block * sizeof(float)]);
       vsubps(Ymm(j+4), Ymm(j+4), ymm_mean);
       vfmadd231ps(Ymm(j), Ymm(j+4), Ymm(j+4));
    }
}

inline void jit_avx2_batch_norm_generator_f32::dst_compute(int block_size,
    jit_batch_normalization_param_t *params)
{
    using Xbyak::Ymm;

    int block_8 = block_size / 8;
    int block_tail = block_size % 8;
    for (int i = 0; i < block_8; i++) {
        for (int j = 0; j < 8; j++) {
            vmovups(Ymm(j), ptr [aux_ptr +
                ((i * 8) + j) * params->c_block * sizeof(float)]);
            vfmsub213ps(Ymm(j), ymm_variance, ymm_mean_mul_variance);
            vfmadd213ps(Ymm(j), ymm_scale, ymm_shift);
            vmovups(ptr [aux_dst_ptr +
                ((i * 8) + j) * params->c_block * sizeof(float)], Ymm(j));
        }
    }
    for (int j = 0; j < block_tail; j++) {
        vmovups(Ymm(j), ptr [aux_ptr +
            ((block_8 * 8) + j) * params->c_block * sizeof(float)]);
        vfmsub213ps(Ymm(j), ymm_variance, ymm_mean_mul_variance);
        vfmadd213ps(Ymm(j), ymm_scale, ymm_shift);
        vmovups(ptr [aux_dst_ptr +
            ((block_8 * 8) + j) * params->c_block * sizeof(float)], Ymm(j));
    }
}

void jit_avx2_batch_norm_generator_f32::generate() {
    auto params = &this->jbnp;
    using Xbyak::Ymm;
    union {
        float _sp_value;
        int _int_value;
    } cvt;

    int N = params->mb;
    int spatial = params->w*params->h;
    int n_blocks = spatial/params->wh_block;

    this->preamble();

#   define GET_OFF(field) offsetof(jit_batch_normalization_kernel_t, field)
    mov(reg_src, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_scaleshift, ptr[this->param1 + GET_OFF(scaleshift)]);
    if (params->_is_training)
        mov(reg_workspace, ptr[this->param1 + GET_OFF(workspace)]);

    vpxor(ymm_mean, ymm_mean);
    vpxor(ymm_variance, ymm_variance);

    cvt._sp_value = (float)(spatial * N);
    mov(tmp_gpr, cvt._int_value);
    movq(xmm_spatial_n, tmp_gpr);
    vbroadcastss(ymm_spatial_n, xmm_spatial_n);

    cvt._sp_value = (float)(params->eps);
    mov(tmp_gpr, cvt._int_value);
    movq(xmm_epsilon, tmp_gpr);
    vbroadcastss(ymm_epsilon, xmm_epsilon);

    cvt._sp_value = (float)(1.0);
    mov(tmp_gpr, cvt._int_value);
    movq(xmm_one, tmp_gpr);
    vbroadcastss(ymm_one, xmm_one);

    for (int i = 0; i < 8; i++)
        vpxor(Ymm(i), Ymm(i));
    mov(aux_ptr, reg_src);
    xor_(n_iter, n_iter);
    L(".n_mean_loop"); {
        mov(save_ptr, aux_ptr);
        if (n_blocks > 0) {
            xor_(sp_iter, sp_iter);
            L(".spatial_mean_loop"); {
                mean_compute(params->wh_block, params);
                add(aux_ptr, params->c_block*params->wh_block*sizeof(float));

                inc(sp_iter);
                cmp(sp_iter, n_blocks);
                jl(".spatial_mean_loop", T_NEAR);
            }
        }
        mean_compute(params->wh_block_tail, params);
        add(save_ptr, params->c*params->w*params->h*sizeof(float));
        mov(aux_ptr, save_ptr);

        inc(n_iter);
        cmp(n_iter, N);
        jl(".n_mean_loop", T_NEAR);
    }
    for (int i = 0; i < 8; i++)
        vaddps(ymm_mean, ymm_mean, Ymm(i));
    vdivps(ymm_mean, ymm_mean, ymm_spatial_n);

    if (params->_is_training)
        vmovups(ptr [reg_workspace], ymm_mean);

    for (int i = 0; i < 8; i++)
        vpxor(Ymm(i), Ymm(i));
    mov(aux_ptr, reg_src);
    xor_(n_iter, n_iter);
    L(".n_variance_loop"); {
        mov(save_ptr, aux_ptr);
        xor_(sp_iter, sp_iter);
        if (n_blocks > 0) {
            L(".spatial_variance_loop"); {
                variance_compute(params->wh_block, params);
                add(aux_ptr, params->c_block*params->wh_block*sizeof(float));

                inc(sp_iter);
                cmp(sp_iter, n_blocks);
                jl(".spatial_variance_loop", T_NEAR);
            }
        }
        variance_compute(params->wh_block_tail, params);
        add(save_ptr, params->c*params->w*params->h*sizeof(float));
        mov(aux_ptr, save_ptr);

        inc(n_iter);
        cmp(n_iter, N);
        jl(".n_variance_loop", T_NEAR);
    }
    for (int i = 0; i < 4; i++)
        vaddps(ymm_variance, ymm_variance, Ymm(i));
    vdivps(ymm_variance, ymm_variance, ymm_spatial_n);
    vaddps(ymm_variance, ymm_variance, ymm_epsilon);
    vsqrtps(ymm_variance, ymm_variance);
    vdivps(ymm_variance, ymm_one, ymm_variance);

    if (params->_is_training)
        vmovups(ptr [reg_workspace + params->c*sizeof(float)], ymm_variance);
    vmulps(ymm_mean_mul_variance, ymm_mean, ymm_variance);
    vmovups(ymm_scale, ptr [reg_scaleshift]);
    vmovups(ymm_shift, ptr [reg_scaleshift + params->c*sizeof(float)]);

    mov(aux_ptr, reg_src);
    mov(aux_dst_ptr, reg_dst);
    xor_(n_iter, n_iter);
    L(".n_dst_loop"); {
        mov(save_ptr, aux_ptr);
        mov(save_dst_ptr, aux_dst_ptr);
        if (n_blocks > 0) {
            xor_(sp_iter, sp_iter);
            L(".spatial_dst_loop");{
                dst_compute(params->wh_block, params);
                add(aux_ptr, params->c_block*params->wh_block*sizeof(float));
                add(aux_dst_ptr,params->c_block*params->wh_block*sizeof(float));

                inc(sp_iter);
                cmp(sp_iter, n_blocks);
                jl(".spatial_dst_loop", T_NEAR);
            }
        }
        dst_compute(params->wh_block_tail, params);

        add(save_ptr, params->c*params->w*params->h*sizeof(float));
        mov(aux_ptr, save_ptr);
        add(save_dst_ptr, params->c*params->w*params->h*sizeof(float));
        mov(aux_dst_ptr, save_dst_ptr);

        inc(n_iter);
        cmp(n_iter, N);
        jl(".n_dst_loop", T_NEAR);
    }

    this->postamble();
}

void jit_avx2_batch_norm_generator_f32::init_jit_params(
    const batch_normalization_desc_t &bnd, const memory_desc_wrapper &src_d,
    const memory_desc_wrapper &scaleshift_d, const memory_desc_wrapper &dst_d,
    const bool _is_training)
{
    jbnp.mb = src_d.dims()[0];
    jbnp.c = src_d.dims()[1];
    jbnp.h = src_d.dims()[2];
    jbnp.w = src_d.dims()[3];

    jbnp.c_block = 8;
    jbnp.nb_c = jbnp.c / jbnp.c_block;
    jbnp.wh_block = 64;

    int spatial_size = jbnp.h*jbnp.w;
    jbnp.wh_block_tail = spatial_size % jbnp.wh_block;
    jbnp.eps = bnd.epsilon;
    jbnp._is_training = _is_training;
}

jit_avx2_batch_norm_generator_f32::jit_avx2_batch_norm_generator_f32(
    const batch_normalization_primitive_desc_t &bnpd,
    const bool _is_training, void *code_ptr, size_t code_size)
    : jit_generator(code_ptr, code_size)
    , jbnp({})
{
    this->init_jit_params(bnpd.batch_normalization_desc,bnpd.src_primitive_desc,
            bnpd.scaleshift_primitive_desc, bnpd.dst_primitive_desc,
            _is_training);
    this->generate();
    jit_ker = (void (*)(void*))this->getCode();
//TODO: if(jit_ker == nullptr) return nullptr;
}

bool jit_avx2_batch_norm_generator_f32::is_applicable(
    const batch_normalization_desc_t &bnorm_d)
{
    const memory_desc_wrapper src_d(bnorm_d.src_desc),
             scaleshift_d(bnorm_d.scaleshift_desc), dst_d(bnorm_d.dst_desc);

    bool args_ok = true
        && src_d.format() == nChw8c
        && scaleshift_d.format() == nc
        && dst_d.format() == nChw8c;
    if (!args_ok) return false;

    return true;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
