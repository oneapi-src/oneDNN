/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#ifndef JIT_PRIMITIVE_CONF_HPP
#define JIT_PRIMITIVE_CONF_HPP

#include <stdint.h>

namespace mkldnn {
namespace impl {
namespace cpu {

enum conv_version_t {ver_unused, ver_fma, ver_4fma, ver_4vnni};
enum conv_loop_order_t {loop_cgn, loop_gnc, loop_ngc};
enum conv_1x1_loop_order_t {loop_rbl, loop_rlb, loop_lbr, loop_lrb, loop_blr,
                            loop_brl};

struct jit_conv_conf_t {
    prop_kind_t prop_kind;
    conv_version_t ver;
    conv_loop_order_t loop_order;

    int mb;
    int ngroups, ic, oc;
    int ih, iw, oh, ow;
    int l_pad, t_pad;
    int r_pad, b_pad;
    int kh, kw;
    int stride_h, stride_w;
    int dilate_h, dilate_w;
    memory_format_t src_fmt;
    bool with_bias, with_relu;
    double relu_negative_slope;

    int ihp, iwp, ohp, owp;
    int nb_ic, ic_block;
    int nb_oc, oc_block;
    int nb_ic_blocking, nb_oc_blocking; // blocking of nb_ic and nb_ic
    int nb_ic_blocking_max;
    int ur_h, ur_w;
    int ur_w_tail;
    bool is_1stconv;
    /* 4fma */
    bool transpose_src;
    int tr_iw;
    /* 4vnni */
    size_t typesize_in;
    size_t typesize_out;
    /* avx512_u8s8u8 */
    int ic_nb1, ic_nb2;
    int oc_nb1;
    int ur_ow_max, ur_ow, ur_ow_tail;
    int ur_ow_nsteps;
    data_type_t bia_dt;
};


struct jit_conv_winograd_conf_t : public jit_conv_conf_t {
    //Winograd specific attributes
    //alpha determines the tile size
    int alpha;
    //number of tiles in x dimension
    int itiles;
    //number of tiles in y dimension
    int jtiles;
    //number of images in a block
    int bimg;

    int nb_Xc;
    int dim_kernel;
    int nb_iter;

    bool double_buffering;
    bool load_U;
    int zmm_start;
    int nb_reg;
};

struct jit_conv_call_s {
    const void *src; /* hack, non-const for backward_data */
    const void *dst; /* hack, non-const for forward */
    const void *filt; /* hack, non-const for backward_weights */
    const void *bias; /* hack, non-const for backward_bias */
    const void *src_prf;
    const void *dst_prf;
    const void *filt_prf;
    const void *bias_prf;
    size_t kh_padding;
    size_t kh_padding_prf;
    size_t kw_padding;
    size_t channel;
    size_t channel_prf;
    size_t oc_blocks;
    int ic_flag;
};

struct jit_1x1_conv_conf_t {
    prop_kind_t prop_kind;
    conv_version_t ver;

    int mb;
    int ngroups, ic, oc;
    int iw, ih, ow, oh;
    int l_pad, t_pad;
    int kh, kw;
    int stride_h, stride_w;
    memory_format_t src_fmt;
    bool with_bias, with_relu;
    double relu_negative_slope;

    int is, os;
    int ic_block, oc_block;

    int ur, ur_tail;

    int reduce_dim, reduce_block, nb_reduce,
        nb_reduce_blocking, nb_reduce_blocking_max;
    int load_dim, load_block, nb_load,
        nb_load_blocking, nb_load_blocking_max;
    int bcast_dim, bcast_block, nb_bcast,
        nb_bcast_blocking, nb_bcast_blocking_max;

    int reduce_loop_unroll, reduce_loop_bcast_step, reduce_loop_load_step;
    int load_loop_load_step, load_loop_iter_step;
    int bcast_loop_output_step, bcast_loop_output_substep;
    int bcast_loop_bcast_step, bcast_loop_bcast_substep;
    int fma_step;
    int load_grp_count;
    conv_1x1_loop_order_t loop_order;
    bool use_vmovntps;
};

struct jit_gemm_conv_conf_t {
    prop_kind_t prop_kind;

    int mb;
    int ngroups, ic, oc;
    int iw, ih, ow, oh;
    int l_pad, t_pad;
    int kh, kw;
    int stride_h, stride_w;
    int dilate_h, dilate_w;
    memory_format_t src_fmt;
    bool with_bias, with_relu;
    double relu_negative_slope;

    int is, os, ks;
    int ic_block, oc_block;
    bool need_im2col;
    size_t im2col_size;
};

struct jit_1x1_conv_call_s {
    const float *bcast_data;
    const float *load_data;
    const float *output_data;
    const float *bias_data; // used in forward and backward_weights only

    size_t load_dim;
    size_t bcast_dim;
    size_t reduce_dim;

    size_t output_stride; // used in backward_weights only

    size_t reduce_pos_flag;
};

}
}
}

#endif
