/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_attr.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
namespace ocl {

#define MAX_NDIMS 6

struct jit_offsets {
    int src_off[4][MAX_NDIMS];
    int wht_off[4][MAX_NDIMS];
    int dst_off[4][MAX_NDIMS];
    int bias_off[4][MAX_NDIMS];
};

struct jit_rnn_offsets {
    int src_layer_off[3][MAX_NDIMS];
    int src_iter_off[3][MAX_NDIMS];
    int src_iter_c_off[3][MAX_NDIMS];
    int weights_layer_off[3][MAX_NDIMS];
    int weights_iter_off[3][MAX_NDIMS];
    int bias_off[3][MAX_NDIMS];
    int dst_layer_off[3][MAX_NDIMS];
    int dst_iter_off[3][MAX_NDIMS];
    int dst_iter_c_off[3][MAX_NDIMS];
    int diff_src_layer_off[3][MAX_NDIMS];
    int diff_src_iter_off[3][MAX_NDIMS];
    int diff_src_iter_c_off[3][MAX_NDIMS];
    int diff_weights_layer_off[3][MAX_NDIMS];
    int diff_weights_iter_off[3][MAX_NDIMS];
    int diff_bias_off[3][MAX_NDIMS];
    int diff_dst_layer_off[3][MAX_NDIMS];
    int diff_dst_iter_off[3][MAX_NDIMS];
    int diff_dst_iter_c_off[3][MAX_NDIMS];
    int ws_off[3][MAX_NDIMS];
};

/* convolution */
enum conv_version_t {
    ver_unused,
    ver_1stconv,
    ver_16mb16c,
    ver_8ow16c,
};

struct jit_conv_conf_t {
    prop_kind_t prop_kind;

    int ndims;
    int mb;
    int ngroups, ic, oc, oc_without_padding, ic_without_padding;
    int id, ih, iw, od, oh, ow;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;

    int od_block, oh_block, ow_block;
    int id_block, ih_block, iw_block;
    int oc_block, ic_block, nchunk;
    int ocb;
    int oh_chunk, mb_chunk, mb_block, slm_ic;
    size_t wht_slm_size, src_slm_size;
    int sub_group_size;
    size_t gws_d[3], lws_d[3];

    bool with_bias, with_sum, with_sum_relu, with_groups;

    bool with_eltwise;
    bool with_relu;
    post_ops_t::entry_t::eltwise_t eltwise;

    bool is_depthwise;
    float relu_negative_slope;
    float sum_scale;
    int scale_idx_mult, rmode;
    int ver;
    format_tag_t src_tag, dst_tag, wei_tag;
    bool is_nchw;
    bool is_nhwc;

    data_type_t src_data_type;
    data_type_t weights_data_type;
    data_type_t bias_data_type;
    data_type_t dst_data_type;
    data_type_t acc_data_type;
};

/* pooling */
struct jit_pool_conf_t {
    int ndims;
    int mb, c;
    int id, ih, iw, od, oh, ow;
    int stride_d, stride_h, stride_w;
    int kd, kh, kw;
    int f_pad, t_pad, l_pad;
    data_type_t src_dt;
    alg_kind_t alg;
    bool is_training, is_backward;
    bool use_16mb_unroll, use_16c_unroll;
    size_t gws_d[3], lws_d[3];
    int sub_group_size;
};

/* inner_product */
struct jit_inner_product_conf_t {
    int ndims;
    int mb, oc, ic, ic_total;
    int id, ih, iw, od, oh, ow;
    int kd, kh, kw;
    bool with_bias, has_spatial;
    bool is_forward, is_backward_data, is_backward_weights;
    data_type_t src_dt;
};

/* rnn */
struct jit_rnn_conf_t {
    int cell_kind;
    int activation_kind;
    int direction_kind;
    bool with_bias;
    bool with_src_iter;
    bool with_src_iter_c;
    bool with_dst_iter;
    bool with_dst_iter_c;
    bool is_lbr;
    bool is_forward;
    data_type_t src_dt;
    data_type_t wei_dt;


    int n_layer;
    int n_direction;
    int n_iter;
    int n_gates;
    int n_bias;
    int n_states;
    int n_weights_input;
    int n_weights_state;
    int batch;
    int slc;
    int sic;
    int dic;
    int dlc;
    int wic;
    int n_parts_wei_st, n_parts_wei_i;
    int src_layer_ndims;
    int src_iter_ndims;
    int src_iter_c_ndims;
    int weights_layer_ndims;
    int weights_iter_ndims;
    int dst_layer_ndims;
    int dst_iter_ndims;
    int dst_iter_c_ndims;
    int bias_ndims;
    int diff_src_layer_ndims;
    int diff_src_iter_ndims;
    int diff_src_iter_c_ndims;
    int diff_weights_layer_ndims;
    int diff_weights_iter_ndims;
    int diff_dst_layer_ndims;
    int diff_dst_iter_ndims;
    int diff_dst_iter_c_ndims;
    int diff_bias_ndims;

    size_t ws_gates_offset;
    size_t ws_states_offset;
    size_t ws_diff_states_offset;
    size_t ws_grid_comp_offset;
    size_t ws_cell_comp_offset;
};

/* bnorm */
struct jit_bnorm_conf_t {
    data_type_t data_type;

    int ndims;
    int mb, ic, mb_chunk, sp_chunk, mb_block;
    int id, ih, iw;
    bool with_relu, use_16mb_unroll;
    bool is_forward, is_backward;
    bool use_scaleshift, save_stats, is_training;
    bool fuse_norm_relu, calculate_stats, calculate_diff_stats;
    bool diff_scaleshift;
    float relu_negative_slope, eps;
    size_t gws_d[3], lws_d[3];
};

/* simple sum */
struct jit_simple_sum_conf_t {
    int ndims;
};

/* simple reorder */
struct jit_reorder_conf_t {
    bool do_reorder, is_alpha_beta, with_group, has_padding;
    int ndims, par_dims, ker_dims, last_dims;
    size_t nelems;
    size_t gws_d[3], lws_d[3];
    int block[3];
    int sub_group_size;
};

/* eltwise */
struct jit_eltwise_conf_t {
    int ndims;
    data_type_t data_type;
    alg_kind_t alg;
    bool is_forward;
    size_t gws_d[3];
};

/* shuffle */
struct jit_shuffle_conf_t {
    data_type_t data_type;
    int axis;
    int axis_size;
    int group_size;
    int transpose_row;
    int transpose_col;
    size_t outer_size;
    size_t inner_size;
    size_t dim;
    int ndims;
    size_t gws_d[3];
};

inline void set_default_conf(jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const primitive_attr_t &attr) {

    const memory_desc_wrapper src_mdw(&src_md);
    const memory_desc_wrapper weights_mdw(&weights_md);
    const memory_desc_wrapper dst_mdw(&dst_md);

    const bool with_groups = weights_mdw.ndims() == src_mdw.ndims() + 1;
    int ndims = src_mdw.ndims();

    jcp = utils::zero<decltype(jcp)>();
    jcp.with_groups = with_groups;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_mdw.dims()[0] : 1;
    jcp.mb = src_mdw.dims()[0];
    jcp.oc_without_padding = dst_mdw.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = src_mdw.dims()[1] / jcp.ngroups;
    jcp.id = (ndims == 5) ? src_mdw.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_mdw.dims()[ndims - 2];
    jcp.iw = src_mdw.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_mdw.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_mdw.dims()[ndims - 2];
    jcp.ow = dst_mdw.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_mdw.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_mdw.dims()[with_groups + ndims - 2];
    jcp.kw = weights_mdw.dims()[with_groups + ndims - 1];

    jcp.oc = dst_mdw.dims()[1] / jcp.ngroups;
    jcp.ic = src_mdw.dims()[1] / jcp.ngroups;

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.back_pad = (ndims == 5) ? cd.padding[1][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.b_pad = (ndims == 3) ? 0 : cd.padding[1][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.r_pad = cd.padding[1][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];
    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    const auto &p = attr.post_ops_;
    jcp.with_sum = p.find(primitive_kind::sum) != -1;
    const int sum_idx = p.find(primitive_kind::sum);
    jcp.sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 1.0;

    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise)
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;
    jcp.with_relu
            = (jcp.with_eltwise && jcp.eltwise.alg == alg_kind::eltwise_relu);
    if (jcp.with_relu)
        jcp.relu_negative_slope = jcp.eltwise.alpha;
    if (p.len_ == 2 && sum_idx != -1) {
        jcp.with_sum_relu = p.entry_[sum_idx].is_sum(jcp.sum_scale == 1.0)
                && p.entry_[1].is_relu();
    } else {
        jcp.with_sum_relu = 0;
    }

    jcp.scale_idx_mult = attr.output_scales_.mask_ == (1 << 1);
}

inline void set_offsets(
        ocl_jit_t &jit, const memory_desc_wrapper &md, const char *str) {
    char tempstr[32];

    dim_t block_dims[MKLDNN_MAX_NDIMS];
    dim_t strides_compat[2][MKLDNN_MAX_NDIMS];

    md.compute_blocks(block_dims);
    md.compute_strides_compat(strides_compat);

    for (int d = 0; d < md.ndims(); ++d) {
        const int block = block_dims[d];

        snprintf(tempstr, 32, "%s_B%d", str, d);
        jit.define_int(tempstr, block);

        snprintf(tempstr, 32, "%s_S%d", str, d);
        jit.define_int(tempstr, strides_compat[0][d]);

        snprintf(tempstr, 32, "%s_SB%d", str, d);
        jit.define_int(tempstr, strides_compat[1][d]);
    }
    for (int d = md.ndims(); d < 6; ++d) {

        snprintf(tempstr, 32, "%s_B%d", str, d);
        jit.define_int(tempstr, 1);

        snprintf(tempstr, 32, "%s_S%d", str, d);
        jit.define_int(tempstr, 0);

        snprintf(tempstr, 32, "%s_SB%d", str, d);
        jit.define_int(tempstr, 0);
    }

    snprintf(tempstr, 32, "%s_OFFSET_PAD", str);
    jit.define_int(tempstr, md.md_->offset0);
}

inline void set_offsets(const memory_desc_wrapper &md, int offs[3][MAX_NDIMS]) {
    dim_t block_dims[MKLDNN_MAX_NDIMS];
    dim_t strides_compat[2][MKLDNN_MAX_NDIMS];

    md.compute_blocks(block_dims);
    md.compute_strides_compat(strides_compat);
    const dims_t &dims = md.dims();

    for (int d = 0; d < md.ndims(); ++d) {
        const int block = block_dims[d];

        offs[0][d] = block;
        offs[1][d] = strides_compat[0][d];
        offs[2][d] = strides_compat[1][d];
        offs[3][d] = dims[d];
    }
}

inline void def_offsets(const int offs[4][MAX_NDIMS], ocl_jit_t &jit,
        const char *str, const int ndims) {

    for (int d = 0; d < ndims; d++) {
        char tempstr[32];
        snprintf(tempstr, 32, "%s_B%d", str, d);
        jit.define_int(tempstr, offs[0][d]);

        snprintf(tempstr, 32, "%s_S%d", str, d);
        jit.define_int(tempstr, offs[1][d]);

        snprintf(tempstr, 32, "%s_SB%d", str, d);
        jit.define_int(tempstr, offs[2][d]);

        snprintf(tempstr, 32, "%s_D%d", str, d);
        jit.define_int(tempstr, offs[3][d]);
    }
    for (int d = ndims; d < 6; ++d) {
        char tempstr[32];
        snprintf(tempstr, 32, "%s_B%d", str, d);
        jit.define_int(tempstr, 1);

        snprintf(tempstr, 32, "%s_S%d", str, d);
        jit.define_int(tempstr, 0);

        snprintf(tempstr, 32, "%s_SB%d", str, d);
        jit.define_int(tempstr, 0);

        snprintf(tempstr, 32, "%s_D%d", str, d);
        jit.define_int(tempstr, 0);
    }
}

inline void def_postops(ocl_jit_t &jit, alg_kind_t alg) {
    jit.define_int("RELU", alg_kind::eltwise_relu);
    jit.define_int("LINEAR", alg_kind::eltwise_linear);
    jit.define_int("BOUNDED_RELU", alg_kind::eltwise_bounded_relu);
    jit.define_int("SOFT_RELU", alg_kind::eltwise_soft_relu);
    jit.define_int("LOGISTIC", alg_kind::eltwise_logistic);
    jit.define_int("TANH", alg_kind::eltwise_tanh);
    jit.define_int("ELU", alg_kind::eltwise_elu);
    jit.define_int("SQUARE", alg_kind::eltwise_square);
    jit.define_int("SQRT", alg_kind::eltwise_sqrt);
    jit.define_int("ABS", alg_kind::eltwise_abs);
    jit.define_int("EXP", alg_kind::eltwise_exp);
    jit.define_int("ALG_KIND", alg);
}

inline void def_data_type(ocl_jit_t &jit, data_type_t dt, const char *str) {
    char tempstr[32];
    switch (dt) {
    case data_type::f16:
        snprintf(tempstr, 32, "-D%s_DATA_T=half -D%s_DT_F16", str, str);
        jit.add_option(tempstr);
        break;
    case data_type::f32:
        snprintf(tempstr, 32, "-D%s_DATA_T=float -D%s_DT_F32", str, str);
        jit.add_option(tempstr);
        break;
    case data_type::s8:
        snprintf(tempstr, 32, "-D%s_DATA_T=char -D%s_DT_S8", str, str);
        jit.add_option(tempstr);
        break;
    case data_type::u8:
        snprintf(tempstr, 32, "-D%s_DATA_T=uchar -D%s_DT_U8", str, str);
        jit.add_option(tempstr);
        break;
    case data_type::s32:
        snprintf(tempstr, 32, "-D%s_DATA_T=int -D%s_DT_S32", str, str);
        jit.add_option(tempstr);
        break;
    default: assert(!"unsupported data type"); break;
    }
}

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
