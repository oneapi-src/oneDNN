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
#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include <math.h>

#include "jit_avx512_common_conv_winograd_kernel_f32.hpp"

#ifndef KERNEL_SIZE_THRESHOLD
#define KERNEL_SIZE_THRESHOLD 16
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_format;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

int L1_cache_size = get_cache_size(1, true);
int L2_cache_size = get_cache_size(2, true);

int get_largest_divisor_lower_than(int number, int bound) {
    int res = 1;
    for (int divisor = 2; divisor <= ::sqrt(number); divisor++) {
        if (number % divisor == 0) {
            /* if we reached the bound, we return the previously found divisor
             */
            if (divisor > bound)
                return res;

            /* if not, number/divisor is a divisor too, so we try it */
            int alt_res = number / divisor;
            if (alt_res <= bound)
                return alt_res;

            /* if number/divisor is greater than the bound, we update
               the current greater divisor lower than bound */
            res = divisor;
        }
    }
    return res;
}

void _jit_avx512_common_conv_winograd_data_kernel_f32::gemm_compute_kernel(
        bool first_iter, bool last_iter) {
    /* This fonction computes a GEMM with:
       - srcA of dimension 16 x 16
       - srcB of dimension dim_kernel x 16
       - dst of dimension  dim_kernel x 16
       All matrices are in column major format.
     */

    auto ker = [=](int ur) {
        if (first_iter) {
            /* Initializing the output 16x16 tile to zero */
            for (int i = 0; i < ur; i++) {
                Zmm zmm(jcp.zmm_start + i);
                vpxord(zmm, zmm, zmm);
            }
        } else {
            /* Load the input from memory */
            for (int i = 0; i < ur; i++) {
                Zmm zmm(jcp.zmm_start + i);
                vmovups(zmm, zword[reg_dstC + 64 * i]);
            }
        }

        /* Computing the GEMM */

        if (jcp.double_buffering)
            vmovups(zmm0, zword[reg_srcA]);

        const int inc_i = jcp.ver == ver_4fma ? 4 : 1;
        for (int i = 0; i < 16; i += inc_i) {
            Zmm current;
            if (!jcp.load_U) {
                if (jcp.double_buffering) {
                    if (i < 15)
                        vmovups(Zmm((i + 1) % 2),
                                zword[reg_srcA + 64 * (i + 1)]);
                    current = Zmm(i % 2);
                } else {
                    for (int j = 0; j < inc_i; j++)
                        vmovups(Zmm(i + j), zword[reg_srcA + 64 * (i + j)]);
                    current = Zmm(i);
                }
            } else {
                current = Zmm(i);
            }
            for (int j = 0; j < ur; j++) {
                Zmm zmm(jcp.zmm_start + j);
                if (jcp.ver == ver_4fma)
                    v4fmaddps(zmm, Zmm(i), EVEX_compress_addr(reg_srcB, 64*j + i*4, false));
                else
                    vfmadd231ps(zmm, current, EVEX_compress_addr(reg_srcB, 64*j + i*4, true));
            }
        }

        /* Writing the result to memory */
        for (int i = 0; i < ur; i++) {
            Zmm zmm(jcp.zmm_start + i);
            vmovups(zword[reg_dstC + 64 * i], zmm);
        }
    };

    int quotient_dim = jcp.dim_kernel / jcp.nb_reg;
    int remainder_dim = jcp.dim_kernel % jcp.nb_reg;

    for (int t = 0; t < quotient_dim; t++) {
        ker(jcp.nb_reg);
        add(reg_srcB, jcp.nb_reg * 16 * 4);
        add(reg_dstC, jcp.nb_reg * 16 * 4);
    }
    if (remainder_dim != 0) {
        ker(remainder_dim);
        add(reg_srcB, remainder_dim * 16 * 4);
        add(reg_dstC, remainder_dim * 16 * 4);
    }
}

void _jit_avx512_common_conv_winograd_data_kernel_f32::gemm_loop_generate() {
    Label nbXc_loop;

    auto load_A = [=]() {
        for (int i = 0; i < 16; i++)
            vmovups(Zmm(i), zword[reg_srcA + 64 * i]);
    };

    auto inner_loops = [=](bool first_iter) {
        Label loop_label;

        if (jcp.load_U)
            load_A();

        // for (int img1 = 0; img1 < jcp.bimg; img1++)
        // for (int tj = 0; tj < jcp.jtiles; tj++)
        // for (int ti = 0; ti < jcp.itiles; ti++)
        {
            if (jcp.nb_iter > 1) {
                mov(reg_loop_cpt, jcp.nb_iter);
                L(loop_label);
            }

            gemm_compute_kernel(first_iter, false);

            if (jcp.nb_iter > 1) {
                sub(reg_loop_cpt, 1);
                jnz(loop_label);
            }
        }
    };

    // register used to handle long fma encoding
    push(reg_EVEX_max_8b_offt);
    mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);

    // first iteration on ifm1 for fwd and ofm1 for bwd_data: GEMM with beta=0
    mov(reg_dstC, reg_dstC_const);
    inner_loops(true);

    // subsequent iteration of GEMMs with beta=1
    if (jcp.nb_Xc > 1) {
        // fwd: for (int ifm1 = 1; ifm1 < jcp.nb_ic; ifm1++)
        // bwd_data: for (int ofm1 = 1; ofm1 < jcp.nb_oc; ofm1++)
        mov(reg_nb_Xc, jcp.nb_Xc - 1);
        L(nbXc_loop);

        /* dstC matrix is independent of ifm1 and serves as accumulator */
        mov(reg_dstC, reg_dstC_const);
        /* We get the next U matrix */
        add(reg_srcA, 16 * 16 * 4);

        inner_loops(false);

        sub(reg_nb_Xc, 1);
        jnz(nbXc_loop);
    }

    pop(reg_EVEX_max_8b_offt);
    ret();
}

status_t _jit_avx512_common_conv_winograd_data_kernel_f32::init_conf_common(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d) {

    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int simd_w = 16;

    // Initializing jcp
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.r_pad =
        nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);
    jcp.b_pad =
        nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih - jcp.t_pad);
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;

    // Winograd specific initialization
    jcp.alpha = 6;
    const int tile_size = jcp.alpha - 2;
    jcp.itiles = (jcp.ow + tile_size - 1) / tile_size;
    jcp.jtiles = (jcp.oh + tile_size - 1) / tile_size;

    // Checking conditions not supported by these kernels
    if (jcp.ngroups != 1)
        return status::unimplemented;
    if ((jcp.kh != 3) || (jcp.kw != 3))
        return status::unimplemented;
    if ((jcp.stride_h != 1) || (jcp.stride_w != 1))
        return status::unimplemented;
    if ((jcp.ic % simd_w) != 0 || (jcp.oc % simd_w) != 0)
        return status::unimplemented;

    if (src_d.format() != nChw16c)
        return status::unimplemented;
    if (weights_d.format() != (with_groups ? gOIhw16i16o : OIhw16i16o))
        return status::unimplemented;
    if (dst_d.format() != nChw16c)
        return status::unimplemented;

    // Setting kernel parameters
    {
        /* Conditions on bimg:
           - it should divide minibatch
           - the matrices of the gemm should fit in the L1 cache
             ([2*(16xbimg*itiles*jtiles)]*4 Bytes < 32KB) (hence the 128 below)
           - minibatch/bimg * alpha^2 should always be at least the number of
           cores
        */

        const int num_threads = omp_get_max_threads();
        int cache_threshold = (L1_cache_size / 128) / (jcp.itiles * jcp.jtiles);
        int num_threads_threshold =
            (jcp.mb * jcp.alpha * jcp.alpha) / num_threads;

        int candidate1 =
            get_largest_divisor_lower_than(jcp.mb, cache_threshold);
        int candidate2 =
            get_largest_divisor_lower_than(jcp.mb, num_threads_threshold);
        jcp.bimg = (candidate1 < candidate2) ? candidate1 : candidate2;

        /* we pick the unrolling factor to prevent tail computation and
         * excessive unrolling */
        int kernel_size_threshold = KERNEL_SIZE_THRESHOLD;
        jcp.dim_kernel = get_largest_divisor_lower_than(
            jcp.bimg * jcp.itiles * jcp.jtiles, kernel_size_threshold);
        jcp.nb_iter = (jcp.bimg * jcp.jtiles * jcp.itiles) / jcp.dim_kernel;
    }

    jcp.ver = mayiuse(avx512_mic_4ops) ? ver_4fma : ver_fma;

    jcp.load_U = true;
    /* we do double buffering if A is not loaded in registers */
    jcp.double_buffering = !(jcp.ver == ver_4fma) && !jcp.load_U;

    if (jcp.load_U) {
        /* The 16 first registers are reserved to load U */
        jcp.zmm_start = 16;
    } else {
        if (jcp.ver == ver_4fma)
            jcp.zmm_start = 4;
        else {
            if (jcp.double_buffering)
                jcp.zmm_start = 2;
            else
                jcp.zmm_start = 1;
        }
    }

    jcp.nb_reg = 32 - jcp.zmm_start;

    jcp.ic_block = (jcp.ic % simd_w != 0) ? jcp.ic : simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    return status::success;
}

status_t jit_avx512_common_conv_winograd_fwd_kernel_f32::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, bool with_relu,
        double relu_negative_slope) {
    status_t st = init_conf_common(jcp, cd, src_d, weights_d, dst_d);

    if (st != status::success)
        return st;

    jcp.with_bias = cd.bias_desc.format != memory_format::undef;
    jcp.nb_Xc = jcp.nb_ic;
    jcp.with_relu = with_relu;
    jcp.relu_negative_slope = relu_negative_slope;

    return status::success;
}

status_t jit_avx512_common_conv_winograd_bwd_data_kernel_f32::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &diff_src_d,
        const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &diff_dst_d) {
    status_t st = init_conf_common(jcp, cd, diff_src_d, weights_d, diff_dst_d);

    if (st != status::success)
        return st;

    jcp.nb_Xc = jcp.nb_oc;

    return status::success;
}

void jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::
    gemm_compute_kernel() {
    /* This function computes a GEMM with:
       - srcA of dimension dim_ker*nb_iter x 16
       - srcB of dimension 16 x dim_ker*nb_iter
       - dst of dimension 16 x 16
       All matrices are in column major format (srcB is transposed).
     */
    Label label_loop_gemm;

    auto ker = [=](int ur) {
        /* Computing the GEMM */
        /* we assume the output is in the first 16 Zmms */
        int num_4fma = jcp.ver == ver_4fma ? ur / 4 : 0;
        int num_fma = jcp.ver == ver_4fma ? ur % 4 : ur;
        int nrowsA = 16;
        int ncolsA = 4;
        int dsize = sizeof(float);
        int cols_dst = 16;

        if (num_4fma) {
            /* Multiply 16 x 4 blocks for each nb_iter.
             * 4 columns of A are multiplied with 4 values of B for each 4fma */
            for (int i = 0; i < num_4fma; ++i){
                for (int j = 0; j < ncolsA; ++j) {
                    vmovups(Zmm(jcp.zmm_start + i * ncolsA + j),
                            zword[reg_srcA + nrowsA * dsize * (ncolsA * i + j)]);
                }
            }

            for (int i = 0; i < num_4fma; ++i){
                int zmm_start_b = jcp.zmm_start + num_4fma * ncolsA;
                for (int j = 0; j < ncolsA; ++j) {
                    vmovups(Zmm(zmm_start_b + j),
                            zword[reg_srcB + nrowsA * dsize * (ncolsA * i + j)]);
                }

                 /* Transposes 4x4 blocks in 16x4 block of B instead of
                  * full 16x4 block, since we only need
                  * 4 contiguous values for one 4fma.*/
                vunpcklps(Zmm(zmm_start_b + 4), Zmm(zmm_start_b),
                        Zmm(zmm_start_b + 1));
                vunpcklps(Zmm(zmm_start_b + 5), Zmm(zmm_start_b + 2),
                        Zmm(zmm_start_b + 3));
                vunpckhps(Zmm(zmm_start_b), Zmm(zmm_start_b),
                        Zmm(zmm_start_b + 1));
                vunpckhps(Zmm(zmm_start_b + 1), Zmm(zmm_start_b + 2),
                        Zmm(zmm_start_b + 3));

                vunpcklpd(Zmm(zmm_start_b + 2), Zmm(zmm_start_b + 4),
                        Zmm(zmm_start_b + 5));
                vunpckhpd(Zmm(zmm_start_b + 3), Zmm(zmm_start_b + 4),
                        Zmm(zmm_start_b + 5));

                vunpcklpd(Zmm(zmm_start_b + 4), Zmm(zmm_start_b),
                        Zmm(zmm_start_b + 1));
                vunpckhpd(Zmm(zmm_start_b + 5), Zmm(zmm_start_b),
                        Zmm(zmm_start_b + 1));

                for (int j = ncolsA - 1; j >= 0; --j) {
                    sub(reg_sp, nrowsA * dsize);
                    /* push transpose to stack instead of original location
                     * to avoid overwrites by other threads*/
                    vmovups(zword[reg_sp], Zmm(zmm_start_b + 2 + j));
                }


                /* Naive approach: prefetch 16x4 blocks for next nb_iter to
                 * avoid L2 misses for A and B*/
                if (i == 0) {
                    for (int j = 0; j < ncolsA; ++j) {
                        prefetcht1(ptr[reg_srcA + nrowsA * dsize
                                * (ncolsA * num_4fma + num_fma + j)]);
                        prefetcht0(ptr[reg_srcB + nrowsA * dsize
                                * (ncolsA * num_4fma + num_fma + j)]);
                    }
                }

                int rowB = 0;
                for (int j = 0; j < cols_dst; ++j) {
                     /* partially transposed 4x16 block of B is multiplied with
                     * columns of A (c1 - column 1, c2 - column 2 and so on...)
                     * in order as shown,
                     * B: c1, c2, c3, c4, c17, c18, c19, c20, ...
                     *    c5, c6, c7, c8, c21, c22, c23, c24, ...
                     *    c9, c10, c11, c12, c25, c26, c27, c28, ...
                     *    c13, c14, c15, c16, c29, c30, c31, c32, ...*/
                    if (j != 0 && j % ncolsA == 0)
                        rowB += ncolsA * dsize;
                    int quadB = (j % ncolsA) * nrowsA * dsize + rowB;
                    v4fmaddps(Zmm(j), Zmm(jcp.zmm_start + i * ncolsA),
                            EVEX_compress_addr(reg_sp, quadB, false));
                }

                for (int j = 0; j < ncolsA; ++j) {
                    add(reg_sp, nrowsA * dsize);
                }
            }
        }

        if (num_fma && jcp.double_buffering)
            vmovups(Zmm(jcp.zmm_start),
                    zword[reg_srcA + num_4fma * ncolsA * nrowsA * dsize]);

        for (int i = 0; i < num_fma; ++i) {
            Zmm current;
            if (jcp.double_buffering) {
                if (i < num_fma - 1)
                    vmovups(Zmm(jcp.zmm_start + ((i + 1) % 2)),
                            zword[reg_srcA
                            + nrowsA * dsize * (num_4fma * ncolsA + i + 1)]);
                current = Zmm(jcp.zmm_start + (i % 2));
            } else {
                vmovups(Zmm(jcp.zmm_start),
                            zword[reg_srcA + nrowsA * dsize * i]);
                current = Zmm(jcp.zmm_start);
            }

            for (int j = 0; j < cols_dst; j++) {
                vfmadd231ps(Zmm(j), current, EVEX_compress_addr(
                            reg_srcB,
                            ((num_4fma * ncolsA + i) * nrowsA  + j) * dsize,
                            true));
            }
        }
    };

    // for (int img1 = 0; img1 < jcp.bimg; img1++)
    // for (int tj = 0; tj < jcp.jtiles; tj++)
    // for (int ti = 0; ti < jcp.itiles; ti++)
    {
        // dim_ker * nb_iter = jcp.bimg * jcp.jtiles * jcp.itiles
        if (jcp.nb_iter > 1) {
            mov(reg_loop_cpt, jcp.nb_iter);
            L(label_loop_gemm);
        }

        ker(jcp.dim_kernel);

        add(reg_srcA, jcp.dim_kernel * 16 * 4);
        add(reg_srcB, jcp.dim_kernel * 16 * 4);

        if (jcp.nb_iter > 1) {
            sub(reg_loop_cpt, 1);
            jnz(label_loop_gemm);
        }
    }
}

void jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::
    gemm_loop_generate(bool first_img) {
    Label nbic_loop;

    // register used to handle long fma encoding
    push(reg_EVEX_max_8b_offt);
    mov(reg_EVEX_max_8b_offt, 2 * EVEX_max_8b_offt);

    if (jcp.nb_ic > 1) {
        // for (int ifm1 = 1; ifm1 < jcp.nb_ic; ifm1++)
        mov(reg_nb_ic, jcp.nb_ic);
        L(nbic_loop);
    }

    {
        /* srcA matrix is independent of ifm1 */
        mov(reg_srcA, reg_srcA_const);

        /* We load U in registers */
        for (int i = 0; i < 16; i++) {
            Zmm zmm(i);
            if (first_img) { /* Initializing the output 16x16 tile to zero */
                vpxord(zmm, zmm, zmm);
            } else { /* We load it from memory hoping it is in cache */
                vmovups(zmm, zword[reg_dstC + 64 * i]);
            }
        }

        gemm_compute_kernel();

        /* Writing U for this ic to memory */
        for (int i = 0; i < 16; i++) {
            Zmm zmm(i);
            vmovups(zword[reg_dstC + 64 * i], zmm);
        }
    }

    if (jcp.nb_ic > 1) {
        /* We get the next U matrix */
        add(reg_dstC, 16 * 16 * 4);

        sub(reg_nb_ic, 1);
        jnz(nbic_loop);
    }

    pop(reg_EVEX_max_8b_offt);
    ret();
}

status_t jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::init_conf(
        jit_conv_winograd_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &diff_dst_d,
        const memory_desc_wrapper &diff_weights_d) {
    if (!mayiuse(avx512_common))
        return status::unimplemented;

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    const int simd_w = 16;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = diff_dst_d.dims()[2];
    jcp.ow = diff_dst_d.dims()[3];
    jcp.kh = diff_weights_d.dims()[with_groups + 2];
    jcp.kw = diff_weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.r_pad =
        nstl::max(0, (jcp.ow - 1) * jcp.stride_w + jcp.kw - jcp.iw - jcp.l_pad);
    jcp.b_pad =
        nstl::max(0, (jcp.oh - 1) * jcp.stride_h + jcp.kh - jcp.ih - jcp.t_pad);
    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.with_bias = (cd.diff_bias_desc.format != memory_format::undef);

    jcp.ic_block = simd_w;
    jcp.nb_ic = jcp.ic / jcp.ic_block;
    jcp.oc_block = simd_w;
    jcp.nb_oc = jcp.oc / jcp.oc_block;

    jcp.ver = mayiuse(avx512_mic_4ops) ? ver_4fma : ver_fma;

    // Winograd specific initialization
    jcp.alpha = 6;
    const int tile_size = jcp.alpha - 2;
    jcp.itiles = (jcp.ow + tile_size - 1) / tile_size;
    jcp.jtiles = (jcp.oh + tile_size - 1) / tile_size;

    // Winograd kernel works only for 3x3 convolution with stride 1
    if (jcp.ngroups != 1)
        return status::unimplemented;
    if ((jcp.kh != 3) || (jcp.kw != 3))
        return status::unimplemented;
    if ((jcp.stride_h != 1) || (jcp.stride_w != 1))
        return status::unimplemented;
    if ((jcp.ic % simd_w) != 0 || (jcp.oc % simd_w) != 0)
        return status::unimplemented;
    if (src_d.format() != nChw16c)
        return status::unimplemented;
    if (diff_weights_d.format() != (with_groups ? gOIhw16i16o : OIhw16i16o))
        return status::unimplemented;
    if (diff_dst_d.format() != nChw16c)
        return status::unimplemented;

    {
        /* Conditions on bimg:
           - it should divide minibatch
           - the matrices of the gemm should fit in the L1 cache
             ([2*(16xbimg*itiles*jtiles)]*4 Bytes < 32KB) (hence the 248 below)
           - minibatch/bimg * alpha^2 should always be at least the number of
           cores
        */

        int num_threads = omp_get_max_threads();

        int cache_threshold = (L1_cache_size / 64) / (jcp.itiles * jcp.jtiles);
        int num_threads_threshold =
            (jcp.mb * jcp.alpha * jcp.alpha) / num_threads;

        int candidate1 =
            get_largest_divisor_lower_than(jcp.mb, cache_threshold);
        int candidate2 =
            get_largest_divisor_lower_than(jcp.mb, num_threads_threshold);
        jcp.bimg = (candidate1 < candidate2) ? candidate1 : candidate2;

        /* Prevents aggressive unrolling */
        int kernel_size_threshold = jcp.ver == ver_4fma ? 4 : 2;
        jcp.dim_kernel = get_largest_divisor_lower_than(
            jcp.bimg * jcp.itiles * jcp.jtiles, kernel_size_threshold);
        jcp.nb_iter = (jcp.bimg * jcp.jtiles * jcp.itiles) / jcp.dim_kernel;
    }

    /* The 16 first registers are reserved to write U */
    jcp.zmm_start = 16;
    jcp.nb_reg = 32 - jcp.zmm_start;
    jcp.double_buffering = true;
    int regs_transV = jcp.ver == ver_4fma ? 6 : 0;
    int regs_loadM = jcp.dim_kernel * (jcp.double_buffering ? 2 : 1);
    assert(jcp.nb_reg >= regs_loadM + regs_transV);

    return status::success;
}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
