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
#include "common/utils.hpp"
#include "compute/compute.hpp"
#include "ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

#define MAX_NDIMS 6

struct jit_memory_desc_info_t {
    // Max 2 levels of blocking
    static const int nlevels = 2;

    int ndims;
    data_type_t data_type;

    int offset0;
    int dims[MAX_NDIMS];
    int padded_dims[MAX_NDIMS];
    int blocks[MAX_NDIMS][nlevels + 1];
    int strides[MAX_NDIMS][nlevels + 1];

    static jit_memory_desc_info_t create(const memory_desc_wrapper &mdw) {
        auto jit_md_info = jit_memory_desc_info_t();

        jit_md_info.ndims = mdw.ndims();
        jit_md_info.data_type = mdw.data_type();
        jit_md_info.offset0 = mdw.offset0();

        auto &blk = mdw.blocking_desc();
        dim_t blk_stride
                = utils::array_product(blk.inner_blks, blk.inner_nblks);

        for (int d = 0; d < mdw.ndims(); ++d) {
            utils::array_set(jit_md_info.blocks[d], 1, nlevels + 1);
            utils::array_set(jit_md_info.strides[d], 0, nlevels + 1);
        }

        for (int d = 0; d < mdw.ndims(); ++d) {
            jit_md_info.dims[d] = mdw.dims()[d];
            jit_md_info.padded_dims[d] = mdw.padded_dims()[d];
            jit_md_info.strides[d][0] = blk.strides[d];
        }

        int levels[MAX_NDIMS] = {0};
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            int d = blk.inner_idxs[iblk];
            ++levels[d];

            jit_md_info.blocks[d][levels[d]] = blk.inner_blks[iblk];
            blk_stride /= blk.inner_blks[iblk];
            jit_md_info.strides[d][levels[d]] = blk_stride;
        }

        // Permute inner blocks for O dimension in OIhw4o8i8o4i and
        // gOIhw4o8i8o4i formats.
        //
        // This is specific for GPU and required for the
        // implementations relying on the subgroup extension.
        if (mdw.matches_one_of_tag(
                    format_tag::OIhw4o8i8o4i, format_tag::gOIhw4o8i8o4i)) {
            int d = (levels[0] == 2) ? 0 : 1;
            nstl::swap(jit_md_info.blocks[d][2], jit_md_info.blocks[d][1]);
            nstl::swap(jit_md_info.strides[d][2], jit_md_info.strides[d][1]);
        }
        return jit_md_info;
    }
};

struct jit_offsets {
    int src_off[4][MAX_NDIMS];
    int wht_off[4][MAX_NDIMS];
    int dst_off[4][MAX_NDIMS];
    int bias_off[4][MAX_NDIMS];
};

struct jit_rnn_offsets {
    int src_layer_off[4][MAX_NDIMS];
    int src_iter_off[4][MAX_NDIMS];
    int src_iter_c_off[4][MAX_NDIMS];
    int weights_layer_off[4][MAX_NDIMS];
    int weights_iter_off[4][MAX_NDIMS];
    int bias_off[4][MAX_NDIMS];
    int dst_layer_off[4][MAX_NDIMS];
    int dst_iter_off[4][MAX_NDIMS];
    int dst_iter_c_off[4][MAX_NDIMS];
    int diff_src_layer_off[4][MAX_NDIMS];
    int diff_src_iter_off[4][MAX_NDIMS];
    int diff_src_iter_c_off[4][MAX_NDIMS];
    int diff_weights_layer_off[4][MAX_NDIMS];
    int diff_weights_iter_off[4][MAX_NDIMS];
    int diff_bias_off[4][MAX_NDIMS];
    int diff_dst_layer_off[4][MAX_NDIMS];
    int diff_dst_iter_off[4][MAX_NDIMS];
    int diff_dst_iter_c_off[4][MAX_NDIMS];
    int ws_off[4][MAX_NDIMS];
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
    int icb;
    int ocb;
    int oh_chunk, mb_chunk, mb_block, slm_ic;
    size_t wht_slm_size, src_slm_size;
    int sub_group_size;
    size_t gws_d[3], lws_d[3];
    compute::dispatch_t dispatch;

    bool with_bias, with_sum, with_sum_relu, with_groups;
    bool with_scales, with_common_scales, with_per_oc_scales;

    bool with_eltwise;
    bool with_post_sum_eltwise;
    bool eltwise_alg_relu;
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
    compute::dispatch_t dispatch;
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
    compute::dispatch_t dispatch;

    data_type_t src_dt;
    data_type_t wei_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
    data_type_t acc_dt;
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
    bool is_fwd;
    bool copy_bias;
    bool is_int8;
    bool is_testmode;
    data_type_t src_dt;
    data_type_t wei_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
    data_type_t acc_dt;
    data_type_t precise_dt;
    data_type_t input_dt;
    data_type_t output_dt;

    int n_layer;
    int n_dir;
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
    int n_parts_weights_iter, n_parts_weights_layer;
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
    int states_ws_ld, gates_ws_ld;

    int wei_qparam_mask;

    size_t ws_gates_offset;
    size_t ws_states_offset;
    size_t ws_diff_states_offset;
    size_t ws_grid_comp_offset;
    size_t ws_cell_comp_offset;
    size_t ws_h_state_offset;
    size_t ws_c_state_offset;
    size_t ws_bias_offset;
    size_t scratchpad_size;
    size_t workspace_size;
};
struct jit_rnn_reorder_conf_t {
    bool do_reorder, with_group, has_padding;
    bool with_sum_ab, with_sum_a;
    bool use_ref_impl;
    int ndims;
    size_t nelems;
    compute::dispatch_t dispatch;
    int block[3];
    int sub_group_size;
    int mask;
    size_t scales_count;
};

/* bnorm */
struct jit_bnorm_conf_t {
    data_type_t data_type;

    int ndims;
    int mb, ic, mb_block;
    int reduce_stat_nblocks;
    int id, ih, iw;
    bool with_relu, use_16mb_unroll;
    bool is_forward, is_backward;
    bool use_scaleshift, save_stats, is_training;
    bool fuse_norm_relu, calculate_stats, calculate_diff_stats;
    bool diff_scaleshift;
    float relu_negative_slope, eps;

    compute::dispatch_t dispatch_calc_stat;
    compute::dispatch_t dispatch_reduce_stat;
    compute::dispatch_t dispatch;
};

/* lnorm */
struct jit_lnorm_conf_t {
    data_type_t data_type;

    bool is_fwd;
    int ndims;
    int norm_axis;

    jit_memory_desc_info_t src_md_info;
    jit_memory_desc_info_t dst_md_info;
    jit_memory_desc_info_t stat_md_info;

    bool use_scaleshift;
    bool calculate_stats;
    bool save_stats;
    float eps;

    compute::dispatch_t dispatch_scaleshift;
    compute::dispatch_t dispatch;
};

/* simple sum */
struct jit_simple_sum_conf_t {
    int ndims;
};

/* binary */
struct jit_binary_conf_t {
    int ndims;
    data_type_t data_type;
    bool is_mul;
    bool is_add;
    bool is_tensor_op;
    compute::dispatch_t dispatch;
    int dim0[MAX_NDIMS];
    int bcast_dims[MAX_NDIMS];
    bool is_dense;
    bool is_same_md;
    jit_memory_desc_info_t src0_md_info;
    jit_memory_desc_info_t src1_md_info;
    jit_memory_desc_info_t dst_md_info;
};

/* simple reorder */
struct jit_reorder_conf_t {
    bool do_reorder, with_group, has_padding;
    bool scale_quant, with_sum_ab, with_sum_a;
    bool use_ref_impl;
    int ndims;
    size_t nelems;

    compute::dispatch_t dispatch;

    int sub_group_size;
    int scale_mask;
    size_t scales_num;

    jit_memory_desc_info_t src_md_info;
    jit_memory_desc_info_t dst_md_info;
};

/* eltwise */
struct jit_eltwise_conf_t {
    int ndims;
    bool with_zero_padding;
    data_type_t data_type;
    alg_kind_t alg;
    bool is_forward;
    compute::dispatch_t dispatch;
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

    jcp.is_depthwise = jcp.with_groups && jcp.oc_without_padding == 1
            && jcp.ic_without_padding == 1;
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
    jcp.with_eltwise = eltwise_ind == 0;
    jcp.with_post_sum_eltwise = eltwise_ind == 1;
    if (jcp.with_eltwise || jcp.with_post_sum_eltwise)
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    jcp.eltwise_alg_relu
            = (eltwise_ind != -1 && jcp.eltwise.alg == alg_kind::eltwise_relu);
    if (jcp.eltwise_alg_relu) jcp.relu_negative_slope = jcp.eltwise.alpha;

    jcp.with_scales = !attr.output_scales_.has_default_values();
    jcp.scale_idx_mult = attr.output_scales_.mask_ == (1 << 1);
    jcp.with_common_scales = jcp.with_scales && attr.output_scales_.mask_ == 0;
    jcp.with_per_oc_scales = jcp.with_scales && jcp.scale_idx_mult;
}

inline void set_offsets(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_wrapper &md, const char *str) {
    dim_t block_dims[DNNL_MAX_NDIMS];
    dim_t strides_compat[2][DNNL_MAX_NDIMS];

    md.compute_blocks(block_dims);
    md.compute_strides_compat(strides_compat);

    for (int d = 0; d < MAX_NDIMS; ++d) {
        const int block = block_dims[d];

        kernel_ctx.define_int(
                utils::format("%s_B%d", str, d), (d < md.ndims()) ? block : 1);
        kernel_ctx.define_int(utils::format("%s_S%d", str, d),
                (d < md.ndims()) ? strides_compat[0][d] : 0);
        kernel_ctx.define_int(utils::format("%s_SB%d", str, d),
                (d < md.ndims()) ? strides_compat[1][d] : 0);
    }

    kernel_ctx.define_int(utils::format("%s_OFFSET_PAD", str), md.md_->offset0);
}

inline void set_offsets(const memory_desc_wrapper &md, int offs[3][MAX_NDIMS]) {
    dim_t block_dims[DNNL_MAX_NDIMS];
    dim_t strides_compat[2][DNNL_MAX_NDIMS];

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

inline void def_offsets(const int offs[4][MAX_NDIMS],
        compute::kernel_ctx_t &kernel_ctx, const char *str, const int ndims) {

    for (int d = 0; d < MAX_NDIMS; d++) {
        kernel_ctx.define_int(
                utils::format("%s_B%d", str, d), (d < ndims) ? offs[0][d] : 1);
        kernel_ctx.define_int(
                utils::format("%s_S%d", str, d), (d < ndims) ? offs[1][d] : 0);
        kernel_ctx.define_int(
                utils::format("%s_SB%d", str, d), (d < ndims) ? offs[2][d] : 0);
        kernel_ctx.define_int(
                utils::format("%s_D%d", str, d), (d < ndims) ? offs[3][d] : 0);
    }
}

inline void def_postops(compute::kernel_ctx_t &kernel_ctx, alg_kind_t alg) {
    kernel_ctx.define_int("RELU", alg_kind::eltwise_relu);
    kernel_ctx.define_int("LINEAR", alg_kind::eltwise_linear);
    kernel_ctx.define_int("BOUNDED_RELU", alg_kind::eltwise_bounded_relu);
    kernel_ctx.define_int("SOFT_RELU", alg_kind::eltwise_soft_relu);
    kernel_ctx.define_int("LOGISTIC", alg_kind::eltwise_logistic);
    kernel_ctx.define_int("TANH", alg_kind::eltwise_tanh);
    kernel_ctx.define_int("ELU", alg_kind::eltwise_elu);
    kernel_ctx.define_int("SQUARE", alg_kind::eltwise_square);
    kernel_ctx.define_int("SQRT", alg_kind::eltwise_sqrt);
    kernel_ctx.define_int("ABS", alg_kind::eltwise_abs);
    kernel_ctx.define_int("EXP", alg_kind::eltwise_exp);
    kernel_ctx.define_int("GELU", alg_kind::eltwise_gelu);
    kernel_ctx.define_int("SWISH", alg_kind::eltwise_swish);
    kernel_ctx.define_int("LOG", alg_kind::eltwise_log);
    kernel_ctx.define_int("CLIP", alg_kind::eltwise_clip);
    kernel_ctx.define_int("ALG_KIND", alg);
}

inline void def_data_type(
        compute::kernel_ctx_t &kernel_ctx, data_type_t dt, const char *str) {
    switch (dt) {
        case data_type::bf16:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=ushort -D%s_DT_BF16", str, str));
            break;
        case data_type::f16:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=half -D%s_DT_F16", str, str));
            break;
        case data_type::f32:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=float -D%s_DT_F32", str, str));
            break;
        case data_type::s8:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=char -D%s_DT_S8", str, str));
            break;
        case data_type::u8:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=uchar -D%s_DT_U8", str, str));
            break;
        case data_type::s32:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=int -D%s_DT_S32", str, str));
            break;
        default: assert(!"unsupported data type"); break;
    }
}

inline void def_memory_desc_info(compute::kernel_ctx_t &kernel_ctx,
        const jit_memory_desc_info_t &jit_md_info, const char *prefix) {
    def_data_type(kernel_ctx, jit_md_info.data_type, prefix);

    kernel_ctx.define_int(
            utils::format("%s_OFFSET0", prefix), jit_md_info.offset0);
    kernel_ctx.define_int(utils::format("%s_NDIMS", prefix), jit_md_info.ndims);

    for (int d = 0; d < MAX_NDIMS; ++d) {
        int dim = (d < jit_md_info.ndims) ? jit_md_info.dims[d] : 0;
        int padded_dim
                = (d < jit_md_info.ndims) ? jit_md_info.padded_dims[d] : 0;
        kernel_ctx.define_int(utils::format("%s_D%d", prefix, d), dim);
        kernel_ctx.define_int(utils::format("%s_PD%d", prefix, d), padded_dim);

        for (int l = 0; l < jit_md_info.nlevels + 1; ++l) {
            int block = (d < jit_md_info.ndims) ? jit_md_info.blocks[d][l] : 1;
            int stride
                    = (d < jit_md_info.ndims) ? jit_md_info.strides[d][l] : 0;
            kernel_ctx.define_int(
                    utils::format("%s_B%d_%d", prefix, d, l), block);
            kernel_ctx.define_int(
                    utils::format("%s_S%d_%d", prefix, d, l), stride);
        }
    }
}

inline void def_dispatch(compute::kernel_ctx_t &kernel_ctx,
        const compute::dispatch_t &dispatch) {
    dispatch.def_kernel_macros(kernel_ctx);
}

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif
