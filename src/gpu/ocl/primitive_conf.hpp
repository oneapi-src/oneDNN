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

#ifndef GPU_OCL_PRIMITIVE_CONF_HPP
#define GPU_OCL_PRIMITIVE_CONF_HPP

#include <stdint.h>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive_attr.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

#define MAX_NDIMS 6

struct memory_desc_info_t {
    // Max 2 levels of blocking
    static const int nlevels = 2;

    int ndims;
    data_type_t data_type;

    int offset0;
    int dims[MAX_NDIMS];
    int padded_dims[MAX_NDIMS];
    int blocks[MAX_NDIMS][nlevels + 1];
    int strides[MAX_NDIMS][nlevels + 1];

    static memory_desc_info_t create(const memory_desc_wrapper &mdw) {
        auto md_info = memory_desc_info_t();

        md_info.ndims = mdw.ndims();
        md_info.data_type = mdw.data_type();
        md_info.offset0 = mdw.offset0();

        auto &blk = mdw.blocking_desc();
        dim_t blk_stride
                = utils::array_product(blk.inner_blks, blk.inner_nblks);

        for (int d = 0; d < mdw.ndims(); ++d) {
            utils::array_set(md_info.blocks[d], 1, nlevels + 1);
            utils::array_set(md_info.strides[d], 0, nlevels + 1);
        }

        for (int d = 0; d < mdw.ndims(); ++d) {
            md_info.dims[d] = mdw.dims()[d];
            md_info.padded_dims[d] = mdw.padded_dims()[d];
            md_info.strides[d][0] = blk.strides[d];
        }

        int levels[MAX_NDIMS] = {0};
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            int d = blk.inner_idxs[iblk];
            ++levels[d];

            md_info.blocks[d][levels[d]] = blk.inner_blks[iblk];
            blk_stride /= blk.inner_blks[iblk];
            md_info.strides[d][levels[d]] = blk_stride;
        }

        // Permute inner blocks for O dimension in OIhw4o8i8o4i and
        // gOIhw4o8i8o4i formats.
        //
        // This is specific for GPU and required for the
        // implementations relying on the subgroup extension.
        if (mdw.matches_one_of_tag(
                    format_tag::OIhw4o8i8o4i, format_tag::gOIhw4o8i8o4i)) {
            int d = (levels[0] == 2) ? 0 : 1;
            nstl::swap(md_info.blocks[d][2], md_info.blocks[d][1]);
            nstl::swap(md_info.strides[d][2], md_info.strides[d][1]);
        }
        return md_info;
    }
};

struct offsets_t {
    int src_off[4][MAX_NDIMS];
    int wei_off[4][MAX_NDIMS];
    int dst_off[4][MAX_NDIMS];
    int bias_off[4][MAX_NDIMS];
};

struct rnn_offsets_t {
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

// Convolution
enum conv_version_t {
    ver_unused,
    ver_1stconv,
    ver_16mb16c,
    ver_8ow16c,
};

struct conv_conf_t {
    prop_kind_t prop_kind;

    int ndims;
    int mb;
    int ngroups, ic, oc;
    int ngroups_without_padding, oc_without_padding, ic_without_padding;
    int id, ih, iw, od, oh, ow;
    int f_pad, l_pad, t_pad;
    int back_pad, r_pad, b_pad;
    int kd, kh, kw;
    int stride_d, stride_h, stride_w;
    int dilate_d, dilate_h, dilate_w;

    int od_block, oh_block, ow_block;
    int id_block, ih_block, iw_block;
    int oc_block, ic_block, nchunk;
    int odb, ohb, owb;
    int icb;
    int ocb;
    int osp_chunk, mb_chunk, mb_block, slm_ic;
    size_t wei_slm_size, src_slm_size;
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

// Pooling
struct pool_conf_t {
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

// Inner Product
struct inner_product_conf_t {
    int ndims;
    int src_ndims, wei_ndims, dst_ndims;
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

// RNN
struct rnn_conf_t {
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
    bool is_training;
    data_type_t src_dt;
    data_type_t wei_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
    data_type_t acc_dt;
    data_type_t aux_dt;
    data_type_t input_dt;
    data_type_t output_dt;
    data_type_t diff_dt;

    int n_layer;
    int n_dir;
    int n_iter;
    int n_iter_scratch_gates;
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
    int states_ws_ld, gates_ws_ld, diff_states_ws_ld, scratch_gates_ld;

    int wei_qparam_mask;

    size_t ws_gates_offset;
    size_t ws_states_offset;
    size_t ws_diff_states_offset;
    size_t ws_grid_comp_offset;
    size_t ws_cell_comp_offset;
    size_t ws_h_state_offset;
    size_t ws_c_state_offset;
    size_t ws_bias_offset;
    size_t scratch_gates_offset;
    size_t scratchpad_size;
    size_t workspace_size;
};

struct rnn_reorder_conf_t {
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

// Batch Normalization
struct bnorm_conf_t {
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

// Layer Normalization
struct lnorm_conf_t {
    data_type_t data_type;

    bool is_fwd;
    int ndims;
    int norm_axis;

    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
    memory_desc_info_t stat_md_info;

    bool use_scaleshift;
    bool calculate_stats;
    bool save_stats;
    float eps;

    compute::dispatch_t dispatch_scaleshift;
    compute::dispatch_t dispatch;
};

// Binary
struct binary_conf_t {
    int ndims;
    data_type_t data_type;
    bool is_mul;
    bool is_add;
    bool is_max;
    bool is_min;
    bool is_tensor_op;
    compute::dispatch_t dispatch;
    int dim0[MAX_NDIMS];
    int bcast_dims[MAX_NDIMS];
    bool is_dense;
    bool is_same_md;
    bool with_eltwise;
    post_ops_t::entry_t::eltwise_t eltwise;
    bool with_sum;
    float sum_scale;
    memory_desc_info_t src0_md_info;
    memory_desc_info_t src1_md_info;
    memory_desc_info_t dst_md_info;
};

// Reorder
struct reorder_conf_t {
    bool do_reorder, with_group, has_padding;
    bool scale_quant, with_sum_ab, with_sum_a;
    bool use_ref_impl;
    int ndims;
    size_t nelems;

    compute::dispatch_t dispatch;

    int sub_group_size;
    int scale_mask;
    size_t scales_num;

    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
};

// Elementwise
struct eltwise_conf_t {
    int ndims;
    bool with_zero_padding;
    data_type_t data_type;
    alg_kind_t alg;
    bool is_forward;
    compute::dispatch_t dispatch;
};

// Shuffle
struct shuffle_conf_t {
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

inline void set_default_conf(conv_conf_t &conf, const convolution_desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr) {

    const memory_desc_wrapper src_mdw(&src_md);
    const memory_desc_wrapper weights_mdw(&weights_md);
    const memory_desc_wrapper dst_mdw(&dst_md);
    const memory_desc_wrapper bias_mdw(&bias_md);

    const bool with_groups = weights_mdw.ndims() == src_mdw.ndims() + 1;
    int ndims = src_mdw.ndims();

    conf = utils::zero<decltype(conf)>();
    conf.with_groups = with_groups;
    conf.ndims = ndims;
    conf.prop_kind = cd.prop_kind;
    conf.ngroups = with_groups ? weights_mdw.dims()[0] : 1;
    conf.mb = src_mdw.dims()[0];
    conf.oc_without_padding = dst_mdw.dims()[1] / conf.ngroups;
    conf.ic_without_padding = src_mdw.dims()[1] / conf.ngroups;
    conf.id = (ndims == 5) ? src_mdw.dims()[2] : 1;
    conf.ih = (ndims == 3) ? 1 : src_mdw.dims()[ndims - 2];
    conf.iw = src_mdw.dims()[ndims - 1];
    conf.od = (ndims == 5) ? dst_mdw.dims()[2] : 1;
    conf.oh = (ndims == 3) ? 1 : dst_mdw.dims()[ndims - 2];
    conf.ow = dst_mdw.dims()[ndims - 1];
    conf.kd = (ndims == 5) ? weights_mdw.dims()[with_groups + 2] : 1;
    conf.kh = (ndims == 3) ? 1 : weights_mdw.dims()[with_groups + ndims - 2];
    conf.kw = weights_mdw.dims()[with_groups + ndims - 1];

    conf.is_depthwise = conf.with_groups && conf.oc_without_padding == 1
            && conf.ic_without_padding == 1;
    conf.oc = dst_mdw.dims()[1] / conf.ngroups;
    conf.ic = src_mdw.dims()[1] / conf.ngroups;

    conf.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    conf.back_pad = (ndims == 5) ? cd.padding[1][0] : 0;
    conf.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    conf.b_pad = (ndims == 3) ? 0 : cd.padding[1][ndims - 4];
    conf.l_pad = cd.padding[0][ndims - 3];
    conf.r_pad = cd.padding[1][ndims - 3];
    conf.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    conf.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    conf.stride_w = cd.strides[ndims - 3];
    conf.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    conf.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    conf.dilate_w = cd.dilates[ndims - 3];

    conf.with_bias = bias_mdw.format_kind() != format_kind::undef;

    conf.src_data_type = src_mdw.data_type();
    conf.weights_data_type = weights_mdw.data_type();
    conf.dst_data_type = dst_mdw.data_type();
    conf.acc_data_type = cd.accum_data_type;
    conf.bias_data_type
            = conf.with_bias ? bias_mdw.data_type() : data_type::f32;

    const auto &p = attr.post_ops_;
    conf.with_sum = p.find(primitive_kind::sum) != -1;
    const int sum_idx = p.find(primitive_kind::sum);
    conf.sum_scale = (sum_idx != -1) ? p.entry_[sum_idx].sum.scale : 1.0;

    const int eltwise_ind = p.find(primitive_kind::eltwise);
    conf.with_eltwise = eltwise_ind == 0;
    conf.with_post_sum_eltwise = eltwise_ind == 1;
    if (conf.with_eltwise || conf.with_post_sum_eltwise)
        conf.eltwise = p.entry_[eltwise_ind].eltwise;

    conf.eltwise_alg_relu
            = (eltwise_ind != -1 && conf.eltwise.alg == alg_kind::eltwise_relu);
    if (conf.eltwise_alg_relu) conf.relu_negative_slope = conf.eltwise.alpha;

    conf.with_scales = !attr.output_scales_.has_default_values();
    conf.scale_idx_mult = attr.output_scales_.mask_ == (1 << 1);
    conf.with_common_scales
            = conf.with_scales && attr.output_scales_.mask_ == 0;
    conf.with_per_oc_scales = conf.with_scales && conf.scale_idx_mult;
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

inline void set_offsets(const memory_desc_wrapper &md, int offs[4][MAX_NDIMS]) {
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
    kernel_ctx.define_int("GELU_TANH", alg_kind::eltwise_gelu_tanh);
    kernel_ctx.define_int("SWISH", alg_kind::eltwise_swish);
    kernel_ctx.define_int("LOG", alg_kind::eltwise_log);
    kernel_ctx.define_int("CLIP", alg_kind::eltwise_clip);
    kernel_ctx.define_int("POW", alg_kind::eltwise_pow);
    kernel_ctx.define_int("GELU_ERF", alg_kind::eltwise_gelu_erf);

    kernel_ctx.define_int("RELU_DST", alg_kind::eltwise_relu_use_dst_for_bwd);
    kernel_ctx.define_int(
            "LOGISTIC_DST", alg_kind::eltwise_logistic_use_dst_for_bwd);
    kernel_ctx.define_int("TANH_DST", alg_kind::eltwise_tanh_use_dst_for_bwd);
    kernel_ctx.define_int("ELU_DST", alg_kind::eltwise_elu_use_dst_for_bwd);
    kernel_ctx.define_int("SQRT_DST", alg_kind::eltwise_sqrt_use_dst_for_bwd);
    kernel_ctx.define_int("EXP_DST", alg_kind::eltwise_exp_use_dst_for_bwd);

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
        const memory_desc_info_t &md_info, const char *prefix) {
    def_data_type(kernel_ctx, md_info.data_type, prefix);

    kernel_ctx.define_int(utils::format("%s_OFFSET0", prefix), md_info.offset0);
    kernel_ctx.define_int(utils::format("%s_NDIMS", prefix), md_info.ndims);

    for (int d = 0; d < MAX_NDIMS; ++d) {
        int dim = (d < md_info.ndims) ? md_info.dims[d] : 0;
        int padded_dim = (d < md_info.ndims) ? md_info.padded_dims[d] : 0;
        kernel_ctx.define_int(utils::format("%s_D%d", prefix, d), dim);
        kernel_ctx.define_int(utils::format("%s_PD%d", prefix, d), padded_dim);

        for (int l = 0; l < md_info.nlevels + 1; ++l) {
            int block = (d < md_info.ndims) ? md_info.blocks[d][l] : 1;
            int stride = (d < md_info.ndims) ? md_info.strides[d][l] : 0;
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
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
