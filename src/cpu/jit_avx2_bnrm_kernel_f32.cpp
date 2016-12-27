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

#include "jit_avx2_bnrm_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

status_t jit_avx2_bnrm_kernel_f32::init_conf(jit_bnrm_conf_t &jbp,
        const batch_normalization_desc_t &bnd,
        const memory_desc_wrapper &data_d,
        const memory_desc_wrapper &scaleshift_d,
        bool is_training, bool stats_is_src, bool use_scaleshift) {
    if (!mayiuse(avx2)) return status::unimplemented;

    bool args_ok = (data_d.format() == memory_format::nChw8c ||
            (data_d.format() == memory_format::nchw
              && data_d.dims()[2] == 1 && data_d.dims()[3] == 1))
        && scaleshift_d.format() == memory_format::nc;
    if (!args_ok) return status::unimplemented;

    jbp.mb = data_d.dims()[0];
    jbp.c = data_d.dims()[1];
    jbp.h = data_d.dims()[2];
    jbp.w = data_d.dims()[3];
    jbp.eps = bnd.batch_norm_epsilon;
    jbp.is_training = is_training;
    jbp.stats_is_src = stats_is_src;
    jbp.use_scaleshift = use_scaleshift;

    jbp.c_block = 8;
    jbp.nb_c = jbp.c / jbp.c_block;
    jbp.wh_block = 64;
    int spatial = jbp.h*jbp.w;
    jbp.wh_block_tail = spatial % jbp.wh_block;

    return status::success;
}

inline void jit_avx2_bnrm_kernel_f32::mean_compute(int block_size) {
    int block_8 = block_size / 8;
    int block_tail = block_size % 8;
    for (int i = 0; i < block_8; i++) {
        for (int j = 0; j < 8; j++) {
            vaddps(Ymm(j), Ymm(j), ptr[aux_ptr  +
                    ((i * 8) + j) * jbp.c_block * sizeof(float)]);
        }
    }
    for (int j = 0; j < block_tail; j++) {
        vaddps(Ymm(j), Ymm(j), ptr[aux_ptr +
                (block_8*8 + j) * jbp.c_block * sizeof(float)]);
    }
}

inline void jit_avx2_bnrm_kernel_f32::variance_compute(int block_size) {
    int block_4 = block_size / 4;
    int block_tail = block_size % 4;

    for (int i = 0; i < block_4; i++) {
        for (int j = 0; j < 4; j++) {
            vmovups(Ymm(j+4), ptr[aux_ptr +
                    ((i * 4) + j) * jbp.c_block * sizeof(float)]);
            vsubps(Ymm(j+4), Ymm(j+4), ymm_mean);
            vfmadd231ps(Ymm(j), Ymm(j+4), Ymm(j+4));
        }
    }
    for (int j = 0; j < block_tail; j++) {
        vmovups(Ymm(j+4), ptr[aux_ptr +
                ((block_4 * 4) + j) * jbp.c_block * sizeof(float)]);
        vsubps(Ymm(j+4), Ymm(j+4), ymm_mean);
        vfmadd231ps(Ymm(j), Ymm(j+4), Ymm(j+4));
    }
}

inline void jit_avx2_bnrm_kernel_f32::dst_compute(int block_size) {
    int block_8 = block_size / 8;
    int block_tail = block_size % 8;
    for (int i = 0; i < block_8; i++) {
        for (int j = 0; j < 8; j++) {
            vmovups(Ymm(j), ptr[aux_ptr +
                    ((i * 8) + j) * jbp.c_block * sizeof(float)]);
            vfmsub213ps(Ymm(j), ymm_variance, ymm_mean_mul_variance);
            if (jbp.use_scaleshift) {
                vfmadd213ps(Ymm(j), ymm_scale, ymm_shift);
            }
            vmovups(ptr[aux_dst_ptr +
                    ((i * 8) + j) * jbp.c_block * sizeof(float)], Ymm(j));
        }
    }
    for (int j = 0; j < block_tail; j++) {
        vmovups(Ymm(j), ptr[aux_ptr +
                ((block_8 * 8) + j) * jbp.c_block * sizeof(float)]);
        vfmsub213ps(Ymm(j), ymm_variance, ymm_mean_mul_variance);
        if (jbp.use_scaleshift) {
            vfmadd213ps(Ymm(j), ymm_scale, ymm_shift);
        }
        vmovups(ptr[aux_dst_ptr +
                ((block_8 * 8) + j) * jbp.c_block * sizeof(float)], Ymm(j));
    }
}

void jit_avx2_bnrm_kernel_f32::generate() {
    int N = jbp.mb;
    int spatial = jbp.w*jbp.h;
    int n_blocks = spatial / jbp.wh_block;

    this->preamble();

#   define GET_OFF(field) offsetof(jit_bnrm_call_s, field)
    mov(reg_src, ptr[this->param1 + GET_OFF(src)]);
    mov(reg_dst, ptr[this->param1 + GET_OFF(dst)]);
    mov(reg_scaleshift, ptr[this->param1 + GET_OFF(scaleshift)]);
    if (jbp.is_training || jbp.stats_is_src){
        mov(reg_mean, ptr[this->param1 + GET_OFF(mean)]);
        mov(reg_variance, ptr[this->param1 + GET_OFF(variance)]);
    }
#   undef GET_OFF

    vpxor(ymm_mean, ymm_mean, ymm_mean);
    vpxor(ymm_variance, ymm_variance, ymm_variance);

    mov(tmp_gpr, float2int(spatial * N));
    movq(xmm_spatial_n, tmp_gpr);
    vbroadcastss(ymm_spatial_n, xmm_spatial_n);

    mov(tmp_gpr, float2int(jbp.eps));
    movq(xmm_epsilon, tmp_gpr);
    vbroadcastss(ymm_epsilon, xmm_epsilon);

    mov(tmp_gpr, float2int(1.0));
    movq(xmm_one, tmp_gpr);
    vbroadcastss(ymm_one, xmm_one);

    if (jbp.stats_is_src) {
        vmovups(ymm_mean, ptr[reg_mean]);
        vmovups(ymm_variance, ptr[reg_variance]);
    } else {
        for (int i = 0; i < 8; i++)
            vpxor(Ymm(i), Ymm(i), Ymm(i));
        mov(aux_ptr, reg_src);
        xor_(n_iter, n_iter);
        L(".n_mean_loop");
        {
            mov(save_ptr, aux_ptr);
            if (n_blocks > 0) {
                xor_(sp_iter, sp_iter);
                L(".spatial_mean_loop");
                {
                    mean_compute(jbp.wh_block);
                    add(aux_ptr, jbp.c_block*jbp.wh_block*sizeof(float));

                    inc(sp_iter);
                    cmp(sp_iter, n_blocks);
                    jl(".spatial_mean_loop", T_NEAR);
                }
            }
            mean_compute(jbp.wh_block_tail);
            add(save_ptr, jbp.c*jbp.w*jbp.h*sizeof(float));
            mov(aux_ptr, save_ptr);

            inc(n_iter);
            cmp(n_iter, N);
            jl(".n_mean_loop", T_NEAR);
        }
        for (int i = 0; i < 8; i++)
            vaddps(ymm_mean, ymm_mean, Ymm(i));
        vdivps(ymm_mean, ymm_mean, ymm_spatial_n);

        if (jbp.is_training) {
            vmovups(ptr[reg_mean], ymm_mean);
        }

        for (int i = 0; i < 8; i++)
            vpxor(Ymm(i), Ymm(i), Ymm(i));
        mov(aux_ptr, reg_src);
        xor_(n_iter, n_iter);
        L(".n_variance_loop");
        {
            mov(save_ptr, aux_ptr);
            xor_(sp_iter, sp_iter);
            if (n_blocks > 0) {
                L(".spatial_variance_loop");
                {
                    variance_compute(jbp.wh_block);
                    add(aux_ptr, jbp.c_block*jbp.wh_block*sizeof(float));

                    inc(sp_iter);
                    cmp(sp_iter, n_blocks);
                    jl(".spatial_variance_loop", T_NEAR);
                }
            }
            variance_compute(jbp.wh_block_tail);
            add(save_ptr, jbp.c*jbp.w*jbp.h*sizeof(float));
            mov(aux_ptr, save_ptr);

            inc(n_iter);
            cmp(n_iter, N);
            jl(".n_variance_loop", T_NEAR);
        }
        for (int i = 0; i < 4; i++)
            vaddps(ymm_variance, ymm_variance, Ymm(i));
        vdivps(ymm_variance, ymm_variance, ymm_spatial_n);
        if (jbp.is_training) {
            vmovups(ptr[reg_variance], ymm_variance);
        }
    }

    vaddps(ymm_variance, ymm_variance, ymm_epsilon);
    vsqrtps(ymm_variance, ymm_variance);
    vdivps(ymm_variance, ymm_one, ymm_variance);

    vmulps(ymm_mean_mul_variance, ymm_mean, ymm_variance);
    if (jbp.use_scaleshift) {
        vmovups(ymm_scale, ptr[reg_scaleshift]);
        vmovups(ymm_shift, ptr[reg_scaleshift + jbp.c*sizeof(float)]);
    }

    mov(aux_ptr, reg_src);
    mov(aux_dst_ptr, reg_dst);
    xor_(n_iter, n_iter);
    L(".n_dst_loop");
    {
        mov(save_ptr, aux_ptr);
        mov(save_dst_ptr, aux_dst_ptr);
        if (n_blocks > 0) {
            xor_(sp_iter, sp_iter);
            L(".spatial_dst_loop");
            {
                dst_compute(jbp.wh_block);
                add(aux_ptr, jbp.c_block*jbp.wh_block*sizeof(float));
                add(aux_dst_ptr, jbp.c_block*jbp.wh_block*sizeof(float));

                inc(sp_iter);
                cmp(sp_iter, n_blocks);
                jl(".spatial_dst_loop", T_NEAR);
            }
        }
        dst_compute(jbp.wh_block_tail);

        add(save_ptr, jbp.c*jbp.w*jbp.h*sizeof(float));
        mov(aux_ptr, save_ptr);
        add(save_dst_ptr, jbp.c*jbp.w*jbp.h*sizeof(float));
        mov(aux_dst_ptr, save_dst_ptr);

        inc(n_iter);
        cmp(n_iter, N);
        jl(".n_dst_loop", T_NEAR);
    }

    this->postamble();
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
