/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_PRIMITIVE_CONF_HPP
#define GPU_INTEL_PRIMITIVE_CONF_HPP

#include <stdint.h>

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/memory_storage.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/gpu_utils.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/compute/kernel_arg_list.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

inline bool memory_desc_ndims_ok(const memory_desc_t *md) {
    return md->ndims > MAX_NDIMS;
}

template <typename T, typename... Rest>
bool memory_desc_ndims_ok(const T *first, const Rest *...rest) {
    return memory_desc_ndims_ok(first) || memory_desc_ndims_ok(rest...);
}

struct memory_desc_info_t {
    // Max levels of blocking
    static const int max_nlevels = 3;

    int ndims;
    data_type_t data_type;

    size_t size;
    dim_t offset0;
    dim_t dims[MAX_NDIMS];
    dim_t padded_dims[MAX_NDIMS];

    int nlevels;
    dim_t blocks[MAX_NDIMS][max_nlevels + 1];
    dim_t strides[MAX_NDIMS][max_nlevels + 1];

    static memory_desc_info_t create(const memory_desc_wrapper &mdw) {
        using namespace format_tag;

        auto md_info = memory_desc_info_t();

        md_info.nlevels = 2;

        md_info.ndims = mdw.ndims();
        md_info.data_type = mdw.data_type();
        md_info.size = mdw.size();
        md_info.offset0 = mdw.offset0();

        auto &blk = mdw.blocking_desc();
        dim_t blk_stride
                = utils::array_product(blk.inner_blks, blk.inner_nblks);

        for (int d = 0; d < mdw.ndims(); ++d) {
            utils::array_set(md_info.blocks[d], 1, md_info.nlevels + 1);
            utils::array_set(md_info.strides[d], 0, md_info.nlevels + 1);
        }

        for (int d = 0; d < mdw.ndims(); ++d) {
            md_info.dims[d] = mdw.dims()[d];
            md_info.padded_dims[d] = mdw.padded_dims()[d];
            md_info.strides[d][0] = blk.strides[d];
        }

        int levels[MAX_NDIMS] = {0};
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk) {
            dim_t d = blk.inner_idxs[iblk];
            ++levels[d];

            md_info.blocks[d][levels[d]] = blk.inner_blks[iblk];
            blk_stride /= blk.inner_blks[iblk];
            md_info.strides[d][levels[d]] = blk_stride;
        }
        return md_info;
    }
};

struct attr_info_t {
    static attr_info_t create(const primitive_attr_t *attr) {
        const auto &po = attr->post_ops_;

        attr_info_t attr_info;

        attr_info.binary_idx = po.find(primitive_kind::binary);
        attr_info.with_binary = (attr_info.binary_idx != -1);

        // Eltwise
        attr_info.eltwise_idx = po.find(primitive_kind::eltwise);
        attr_info.with_eltwise = (attr_info.eltwise_idx != -1);

        if (attr_info.with_eltwise) {
            auto &eltwise = po.entry_[attr_info.eltwise_idx].eltwise;
            attr_info.eltwise_alg = eltwise.alg;
            attr_info.eltwise_scale = eltwise.scale;
            attr_info.eltwise_alpha = eltwise.alpha;
            attr_info.eltwise_beta = eltwise.beta;
        } else {
            attr_info.eltwise_alg = alg_kind::undef;
            attr_info.eltwise_scale = 1.0f;
            attr_info.eltwise_alpha = 1.0f;
            attr_info.eltwise_beta = 0.0f;
        }

        // Sum
        attr_info.sum_idx = po.find(primitive_kind::sum);
        attr_info.sum_scale = (attr_info.sum_idx != -1
                        ? po.entry_[attr_info.sum_idx].sum.scale
                        : 0.0f);
        attr_info.sum_data_type = (attr_info.sum_idx != -1)
                ? po.entry_[attr_info.sum_idx].sum.dt
                : dnnl_data_type_undef;
        attr_info.with_sum
                = (attr_info.sum_idx != -1) && (attr_info.sum_scale != 0.0f);

        const auto &src_scales = attr->scales_.get(DNNL_ARG_SRC);
        attr_info.with_src_scales = !src_scales.has_default_values();
        attr_info.with_src0_scale = !src_scales.has_default_values();
        attr_info.src_scales_mask = src_scales.mask_;

        const auto &src1_scales = attr->scales_.get(DNNL_ARG_SRC_1);
        attr_info.with_src1_scale = !src1_scales.has_default_values();
        gpu_assert(src1_scales.mask_ == 0);

        const auto &wei_scales = attr->scales_.get(DNNL_ARG_WEIGHTS);
        attr_info.with_wei_scales = !wei_scales.has_default_values();
        attr_info.wei_scales_mask = wei_scales.mask_;

        const auto &dst_scales = attr->scales_.get(DNNL_ARG_DST);
        attr_info.with_dst_scales = !dst_scales.has_default_values();
        gpu_assert(dst_scales.mask_ == 0);

        // zero points
        const auto &zp = attr->zero_points_;
        attr_info.with_src_zpoints = !zp.has_default_values(DNNL_ARG_SRC);
        attr_info.with_wei_zpoints = !zp.has_default_values(DNNL_ARG_WEIGHTS);
        attr_info.with_dst_zpoints = !zp.has_default_values(DNNL_ARG_DST);
        attr_info.src_zpoints_data_type = zp.get_data_type(DNNL_ARG_SRC);
        attr_info.wei_zpoints_data_type = zp.get_data_type(DNNL_ARG_WEIGHTS);
        attr_info.dst_zpoints_data_type = zp.get_data_type(DNNL_ARG_DST);

        attr_info.with_per_ic_src_zpoints = attr_info.with_src_zpoints
                && !zp.has_default_values(DNNL_ARG_SRC)
                && !zp.common(DNNL_ARG_SRC);

        attr_info.with_per_oc_dst_zpoints = attr_info.with_dst_zpoints
                && !zp.has_default_values(DNNL_ARG_DST)
                && !zp.common(DNNL_ARG_DST);

        attr_info.initialized = true;
        return attr_info;
    }

    bool initialized = false;

    bool with_binary;
    bool with_eltwise;
    int eltwise_idx;
    int binary_idx;
    alg_kind_t eltwise_alg;
    float eltwise_scale;
    float eltwise_alpha;
    float eltwise_beta;

    bool with_sum;
    int sum_idx;
    float sum_scale;
    data_type_t sum_data_type;

    bool with_src0_scale;
    bool with_src1_scale;
    bool with_src_scales;
    bool with_wei_scales;
    bool with_dst_scales;
    bool src_scales_mask;
    bool wei_scales_mask;

    bool with_src_zpoints;
    bool with_wei_zpoints;
    bool with_dst_zpoints;
    bool with_per_ic_src_zpoints;
    bool with_per_oc_dst_zpoints;
    data_type_t src_zpoints_data_type;
    data_type_t wei_zpoints_data_type;
    data_type_t dst_zpoints_data_type;
};

template <size_t ndims>
using strides_t = std::array<dim_t, ndims>;
template <>
struct compute::scalar_type_traits<strides_t<2>> {
    static const auto type = scalar_type_t::_int64x3_t;
};
template <>
struct compute::scalar_type_traits<strides_t<3>> {
    static const auto type = scalar_type_t::_int64x3_t;
};
template <>
struct compute::scalar_type_traits<strides_t<4>> {
    static const auto type = scalar_type_t::_int64x4_t;
};
template <>
struct compute::scalar_type_traits<strides_t<5>> {
    static const auto type = scalar_type_t::_int64x5_t;
};
template <>
struct compute::scalar_type_traits<strides_t<6>> {
    static const auto type = scalar_type_t::_int64x5_t;
};

struct offsets_t {
    dim_t src_off[4][MAX_NDIMS];
    dim_t wei_off[4][MAX_NDIMS];
    dim_t dst_off[4][MAX_NDIMS];
};

struct rnn_offsets_t {
    strides_t<3> src_layer;
    strides_t<4> src_iter;
    strides_t<4> src_iter_c;
    strides_t<5> weights_layer;
    strides_t<5> weights_iter;
    dim_t weights_layer_comp_off;
    dim_t weights_iter_comp_off;
    strides_t<4> bias;
    strides_t<3> dst_layer;
    strides_t<4> dst_iter;
    strides_t<4> dst_iter_c;
    strides_t<3> diff_src_layer;
    strides_t<4> diff_src_iter;
    strides_t<4> diff_src_iter_c;
    strides_t<5> diff_weights_layer;
    strides_t<5> diff_weights_iter;
    strides_t<4> diff_bias;
    strides_t<3> diff_dst_layer;
    strides_t<4> diff_dst_iter;
    strides_t<4> diff_dst_iter_c;
};

// Convolution
enum conv_version_t {
    ver_unused,
    ver_1stconv,
    ver_16mb16c,
    ver_32mb16c,
    ver_32mb32c,
    ver_32c,
    ver_8ow16c,
    ver_nhwc,
    ver_nchw,
    ver_mb_block,
    ver_ow_block,

    // Xe_HP-specific versions.
    ver_v1,
    ver_v2
};

struct conv_conf_t {
    prop_kind_t prop_kind;

    int ndims;
    dim_t mb;
    dim_t ngroups, ic, oc;
    dim_t ngroups_without_padding, oc_without_padding, ic_without_padding;
    dim_t id, ih, iw, od, oh, ow;
    dim_t f_pad, l_pad, t_pad;
    dim_t back_pad, r_pad, b_pad;
    dim_t kd, kh, kw, kwb;
    dim_t stride_d, stride_h, stride_w;
    dim_t dilate_d, dilate_h, dilate_w;

    int oh_block, ow_block;
    int oc_block, ic_block;
    dim_t ocb;
    int mb_block;
    int iw_tail;
    size_t wei_slm_size, src_slm_size, dst_slm_size;
    int sub_group_size;

    compute::range_t gws_d = compute::range_t::empty();
    compute::range_t lws_d = compute::range_t::empty();
    compute::dispatch_t dispatch;

    bool with_bias, with_groups;

    attr_info_t attr_info;

    bool is_depthwise;
    bool is_nhwc;
    bool reorder_wei = false;
    bool reorder_bias = false;
    int ver;
    format_tag_t src_tag, dst_tag, wei_tag;
    bool is_nchw;
    bool is_src_nchw, is_src_nhwc;
    bool is_dst_nhwc;

    int tile_size;
    int wino_m;
    int wino_r;
    dim_t wino_ih, wino_oh;
    dim_t wino_iw, wino_ow;
    dim_t wino_ic;
    dim_t wino_oc;
    int wino_ic_block;
    int wino_oc_block;
    int vect_size;
    compute::range_t U_gws_d = compute::range_t::empty();
    compute::range_t U_lws_d = compute::range_t::empty();
    compute::range_t V_gws_d = compute::range_t::empty();
    compute::range_t V_lws_d = compute::range_t::empty();
    compute::range_t M_gws_d = compute::range_t::empty();
    compute::range_t M_lws_d = compute::range_t::empty();
    bool is_fused;

    data_type_t src_data_type;
    data_type_t weights_data_type;
    data_type_t bias_data_type;
    data_type_t dst_data_type;
    data_type_t acc_data_type;

    memory_desc_info_t src_md_info;
    memory_desc_info_t wei_md_info;
    memory_desc_info_t dst_md_info;
};

// Pooling
struct pool_conf_t {
    int ndims;
    dim_t mb, c;
    dim_t mb_padded;
    dim_t c_padded;
    dim_t id, ih, iw, od, oh, ow;
    dim_t stride_d, stride_h, stride_w;
    dim_t kd, kh, kw;
    dim_t dd, dh, dw;
    dim_t f_pad, t_pad, l_pad;
    data_type_t src_dt;
    data_type_t dst_dt;
    alg_kind_t alg;
    bool is_plain;
    bool is_training, is_backward;
    bool use_mb_c_block, use_only_c_block;
    int unroll_mb_count = 1;
    bool vectorize = true;
    int chunks_per_c_block, chunks_per_mb_block;
    int vect_dt_n;
    int nvect;
    compute::dispatch_t dispatch;
    int sub_group_size;
    dim_t global_pool_spatial_chunk;
    dim_t num_batches = 1;
    int mb_block_size = 16;

    attr_info_t attr_info;
    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
};

// Prelu
struct prelu_conf_t {
    bool is_forward;
    bool reduce_diff_weights;
    compute::dispatch_t dispatch;

    attr_info_t attr_info;
    memory_desc_info_t src_md_info;
    memory_desc_info_t wei_md_info;
    memory_desc_info_t dst_md_info;
    memory_desc_info_t diff_src_md_info;
    memory_desc_info_t diff_wei_md_info;
};

// Inner Product
struct inner_product_conf_t {
    dim_idx_t ndims;
    dim_idx_t src_ndims, wei_ndims, dst_ndims;
    dim_t mb, oc, ic, ic_total;
    dim_t id, ih, iw, od, oh, ow;
    dim_t kd, kh, kw;
    bool with_bias, has_spatial;
    bool is_forward, is_backward_data, is_backward_weights;
    compute::dispatch_t dispatch;
    bool reorder_dst = false;

    data_type_t src_dt;
    data_type_t wei_dt;
    data_type_t bia_dt;
    data_type_t dst_dt;
    data_type_t acc_dt;

    attr_info_t attr_info;
};

struct rnn_reorder_conf_t {
    bool do_reorder, with_group, has_padding;
    bool with_sum_ab, with_sum_a;
    bool use_ref_impl;
    int ndims;
    size_t nelems;
    compute::dispatch_t dispatch;
    int sub_group_size;
    int mask;
    size_t scales_count;
};

// Batch Normalization
enum bn_impl_t {
    unknown = 0,
    ref,
    simple,
    reusable,
    gen9,
    nhwc_opt,
    nhwc_reusable
};

struct bnorm_conf_t {
    data_type_t data_type;
    size_t elsz;
    dim_idx_t ndims;
    dim_t mb, ic, id, ih, iw;
    int mb_block;
    dim_idx_t reduce_dim_idx;
    dim_t reduce_dim;
    dim_t nn, sp, sp_tail;
    int vect_size;
    dim_t stat_sp_nblocks, stat_sp_tail;
    dim_t update_sp_nblocks, update_sp_tail;
    dim_t reduce_stat_nblocks;
    bool with_relu;
    dim_t stat_ic;
    bool is_forward, is_backward;
    bool use_scale, use_shift, save_stats, is_training;
    bool calculate_stats, calculate_diff_stats;
    bool fuse_norm_relu, fuse_norm_add_relu;
    bool diff_scale, diff_shift;
    float relu_negative_slope, eps;
    int sub_group_size;
    bool skip_reduce_stat;
    bool use_stats_one_pass;
    dim_t calc_stat_ic;
    int max_ic_block;
    bn_impl_t impl = bn_impl_t::unknown;
};

// Layer Normalization
struct lnorm_conf_t {
    data_type_t src_dt, dst_dt;
    data_type_t weights_data_type = data_type::f32;

    bool is_fwd;
    dim_idx_t ndims;
    dim_idx_t norm_axis;
    dim_idx_t across_axis;
    int norm_block;
    int num_norm_blocks;
    int norm_block_fused;
    int num_norm_blocks_fused;
    int across_block;
    int num_across_blocks;

    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
    memory_desc_info_t stat_md_info;

    bool use_scale;
    bool use_shift;
    bool use_fused;
    bool calculate_stats;
    bool save_stats;
    bool vectorize_calc_stats;
    bool vectorize_bwd;
    bool vectorize_bwd_scaleshift;
    float eps;
    int sub_group_size;
    int vect_dt_n;
    int vect_size_fused;
    int shift_off;
    int n_chunk_size;
    dim_t finalize_n_chunks;
    dim_t n_chunks;
    int vector_size_scaleshift;
    bool use_src_buffer;

    compute::dispatch_t dispatch_scaleshift;
    compute::dispatch_t dispatch_scaleshift_finalize;
    compute::dispatch_t dispatch;
    compute::dispatch_t dispatch_fused;
};

// Binary
struct binary_conf_t {
    int ndims, nvect;
    bool use_unroll_16b, src0_unroll_16b;
    bool is_plain_layout;
    bool plain_to_ABcd4a4b;
    bool isXa16b;
    data_type_t src0_data_type;
    data_type_t src1_data_type;
    data_type_t dst_data_type;
    alg_kind_t alg;
    // bool is_ne;
    bool is_tensor_op;
    compute::dispatch_t dispatch;
    int mb_block;
    int has_tail;
    int dim0[MAX_NDIMS];
    int src0_bcast_dims[MAX_NDIMS];
    int src1_bcast_dims[MAX_NDIMS];
    bool is_dense;
    bool is_same_md;
    bool same_src_dt;
    bool with_binary_post_op;
    bool is_src1_broadcast;
    bool is_src0_blocked;

    memory_desc_info_t src0_md_info;
    memory_desc_info_t src1_md_info;
    memory_desc_info_t dst_md_info;

    attr_info_t attr_info;
};

// Reduction
struct reduction_conf_t {
    // Used by reference implementation
    alg_kind_t alg;
    int ndims, div;
    float eps, power;
    dim_t src_dims[MAX_NDIMS], reduce_dims[MAX_NDIMS], dst_dims[MAX_NDIMS];
    bool is_reduction_dim[MAX_NDIMS];
    int hwd_reduction_size, hwd_size;
    data_type_t src_type, dst_type;
    memory_desc_info_t src_md_info, dst_md_info;
    compute::dispatch_t dispatch;
    offsets_t off;
    attr_info_t attr_info;
};

// Reorder
enum reorder_kernel_t {
    none,
    dense_vector,
    unroll_16b,
    unroll_16b16c,
    unroll_16a16b,
    plain_to_ABcd84a42b,
    vectorize_last_dim,
    plain_to_ABxx8ayb,
    plain_xFxE_to_abcdef,
    transpose8x8,
    transpose16x16,
    local8x8,
    local16x16,
    reorder_nchw,
    unaligned_sizes,
    reorder_alt,
    vectorize_groups,
    pad_innermost,
    xb_to_xab_xba
};

// Resampling
struct resampling_conf_t {
    dim_idx_t ndims;
    offsets_t off;
    dim_t MB, C;
    dim_t ID, IH, IW;
    dim_t OD, OH, OW;
    float FD, FH, FW;
    int vect_size;
    dims_t padded_strides;
    compute::range_t gws = compute::range_t::empty();
    compute::range_t lws = compute::range_t::empty();
    int sub_group_size;
    dim_t padded_c;
    attr_info_t attr_info;
    compute::dispatch_t dispatch;
};

struct block_desc_t {
    dim_idx_t dim_idx;
    int blk_size;
    int step_size;
};

#define LOOP_NEST_LEVEL 4
struct vectorize_last_dim_t {
    dim_idx_t vector_dim;
    int rescale_coeff;
    // composition of data within 16-item packet
    block_desc_t src_vct[LOOP_NEST_LEVEL];
    block_desc_t dst_vct[LOOP_NEST_LEVEL];
    // dimensions to loop over when accessing packets defined above
    block_desc_t src_blk[LOOP_NEST_LEVEL];
    block_desc_t dst_blk[LOOP_NEST_LEVEL];
    int src_blk_limits[MAX_NDIMS];
    int dst_blk_limits[MAX_NDIMS];
    int src_vect_limit;
    int dst_vect_limit;
};

struct vectorize_group_t {
    dim_idx_t vector_dim;
    dim_idx_t src_loop_dim;
    dim_idx_t dst_loop_dim;
    int group_size;
    int innermost_size;
};

struct xb_to_xab_xba_t {
    int vd;
    dim_t blk_size;
    dim_idx_t src_blk_dim;
    dim_t src_blk_coeff;
    dim_idx_t dst_blk_dim;
    dim_t dst_blk_coeff;
};

union reorder_implementation {
    vectorize_group_t vg;
    xb_to_xab_xba_t ab;
    vectorize_last_dim_t vld;
};

struct quantization_t : public gpu::quantization_t {
public:
    using gpu::quantization_t::quantization_t;

    void define_macros(
            compute::kernel_ctx_t &kernel_ctx, const std::string &name) const {
        if (with_scale()) {
            kernel_ctx.define_int("WITH_" + name + "_SCALE", 1);
            kernel_ctx.define_int(name + "_SCALE_MASK", scale_mask());
            kernel_ctx.define_int(name + "_NUM_SCALES", num_scales());
        }

        if (with_zp()) {
            kernel_ctx.define_int("WITH_" + name + "_ZPOINT", 1);
            kernel_ctx.define_int(name + "_ZPOINT_MASK", zp_mask());
            kernel_ctx.define_int(name + "_NUM_ZPOINTS", num_zps());
        }
    }
};

struct sum_quantization_t : public gpu::sum_quantization_t {
public:
    using gpu::sum_quantization_t::sum_quantization_t;

    void define_macros(
            compute::kernel_ctx_t &kernel_ctx, const std::string &name) const {
        if (with_scale()) kernel_ctx.define_int("WITH_" + name + "_SCALE", 1);
        if (with_zp()) kernel_ctx.define_int("WITH_" + name + "_ZPOINT", 1);
    }
};

struct reorder_conf_t {
    bool has_padding;

    quantization_t src_quant, dst_quant;
    sum_quantization_t sum_quant;

    reorder_kernel_t implementation;
    int ndims;
    size_t nelems;
    bool subbyte_pack = false;

    compute::dispatch_t dispatch;

    int sub_group_size;
    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;

    reorder_implementation aux_data;
};

// Concat
struct concat_conf_t {
    dim_t dst_extern_dim_size;
    dim_t src_extern_dim_sizes[64];
    dim_t offset[64];
    dim_t padded_offset[64];
    dim_t n_blocks;
    dim_t blocks[6];
    dim_t strides[6];
    dim_t inner_axis;
    dim_t dst_concat_axis;
    dim_t dst_padded_concat_axis;
    dim_t dst_offset0;
    dim_t read_block;
    dim_t write_block;
    dim_t gws0_block;
    dim_t read_overlap;
    int n;
    int simd;
    int data_type_size;
    compute::range_t gws_d = compute::range_t::one();
    compute::range_t lws_d;

    data_type_t src_type, dst_type;
    compute::dispatch_t dispatch;
    int ndims;
    memory_desc_info_t src_md_infos[16]; // simple concat does not use this
    memory_desc_info_t dst_md_info;
    int concat_axis;
    int sub_group_size;
    int iter_dim_idx, iter_dim_chunk;
    scales_query_t scale_src[64];
    uint64_t scales_mask;
    bool use_large_index = true;
};

// Shuffle
struct shuffle_conf_t {
    data_type_t data_type;
    dim_idx_t axis;
    dim_t transpose_row;
    dim_t transpose_col;
    compute::dispatch_t dispatch;
    memory_desc_info_t src_md_info;
    memory_desc_info_t dst_md_info;
};

inline void set_default_pool_conf(pool_conf_t &conf, const pooling_desc_t &desc,
        const memory_desc_t &src_md, const memory_desc_t &dst_md,
        const primitive_attr_t &attr) {
    const memory_desc_wrapper src_mdw(src_md);
    const memory_desc_wrapper dst_mdw(dst_md);

    const auto &src_dims = src_mdw.dims();
    const auto &dst_dims = dst_mdw.dims();

    int ndims = src_mdw.ndims();
    conf.ndims = ndims;

    conf.mb = src_dims[0];

    conf.c = src_dims[1];
    conf.mb_padded = src_mdw.padded_dims()[0];
    conf.c_padded = src_mdw.padded_dims()[1];
    conf.id = (ndims == 5) ? src_dims[2] : 1;
    conf.ih = (ndims == 3) ? 1 : src_dims[ndims - 2];
    conf.iw = src_dims[ndims - 1];
    conf.od = (ndims == 5) ? dst_dims[2] : 1;
    conf.oh = (ndims == 3) ? 1 : dst_dims[ndims - 2];
    conf.ow = dst_dims[ndims - 1];

    conf.stride_d = (ndims == 5) ? desc.strides[0] : 1;
    conf.stride_h = (ndims == 3) ? 1 : desc.strides[ndims - 4];
    conf.stride_w = desc.strides[ndims - 3];
    conf.kd = (ndims == 5) ? desc.kernel[0] : 1;
    conf.kh = (ndims == 3) ? 1 : desc.kernel[ndims - 4];
    conf.kw = desc.kernel[ndims - 3];

    conf.dd = (ndims == 5) ? desc.dilation[0] : 0;
    conf.dh = (ndims == 3) ? 0 : desc.dilation[ndims - 4];
    conf.dw = desc.dilation[ndims - 3];

    conf.f_pad = (ndims == 5) ? desc.padding[0][0] : 0;
    conf.t_pad = (ndims == 3) ? 0 : desc.padding[0][ndims - 4];
    conf.l_pad = desc.padding[0][ndims - 3];

    conf.alg = desc.alg_kind;

    conf.src_dt = src_mdw.data_type();
    conf.dst_dt = dst_mdw.data_type();

    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);

    conf.is_training = desc.prop_kind == prop_kind::forward_training;
    conf.is_backward = desc.prop_kind == prop_kind::backward_data;

    conf.attr_info = attr_info_t::create(&attr);
}

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

    if (!src_mdw.format_any())
        conf.src_md_info = memory_desc_info_t::create(src_mdw);
    if (!weights_mdw.format_any())
        conf.wei_md_info = memory_desc_info_t::create(weights_mdw);
    if (!dst_mdw.format_any())
        conf.dst_md_info = memory_desc_info_t::create(dst_mdw);

    conf.attr_info = attr_info_t::create(&attr);
}

inline void set_offsets(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_wrapper &md, const char *str) {
    dim_t block_dims[DNNL_MAX_NDIMS];
    dim_t strides_compat[2][DNNL_MAX_NDIMS];

    md.compute_blocks(block_dims);
    md.compute_strides_compat(strides_compat);

    for (int d = 0; d < MAX_NDIMS; ++d) {
        const dim_t block = block_dims[d];

        kernel_ctx.define_int(
                utils::format("%s_B%d", str, d), (d < md.ndims()) ? block : 1);
        kernel_ctx.define_int(utils::format("%s_S%d", str, d),
                (d < md.ndims()) ? strides_compat[0][d] : 0);
        kernel_ctx.define_int(utils::format("%s_SB%d", str, d),
                (d < md.ndims()) ? strides_compat[1][d] : 0);
    }

    kernel_ctx.define_int(utils::format("%s_OFFSET_PAD", str), md.md_->offset0);
}

inline void set_offsets(
        const memory_desc_wrapper &md, dim_t offs[4][MAX_NDIMS]) {
    dim_t block_dims[DNNL_MAX_NDIMS];
    dim_t strides_compat[2][DNNL_MAX_NDIMS];

    md.compute_blocks(block_dims);
    md.compute_strides_compat(strides_compat);
    const dims_t &dims = md.dims();

    for (int d = 0; d < md.ndims(); ++d) {
        const dim_t block = block_dims[d];

        offs[0][d] = block;
        offs[1][d] = strides_compat[0][d];
        offs[2][d] = strides_compat[1][d];
        offs[3][d] = dims[d];
    }
}

struct outer_strides_getter_t {
    template <size_t ndims>
    operator strides_t<ndims>() const {
        strides_t<ndims> ret;
        gpu_assert(into<dim_t>(ndims) >= md.ndims());
        for (int d = ndims - 1; d >= 0; d--) {
            // Assumes size 1 dimensions are dense with respect to the neighboring
            // dimension so they can be used for size calculations in some layouts
            ret[d] = [&]() {
                if (d >= md.ndims())
                    return static_cast<dim_t>(0);
                else if (md.padded_dims()[d] > 1)
                    return md.strides()[d];
                else if (d == md.ndims() - 1)
                    return static_cast<dim_t>(1);
                else
                    return ret[d + 1] * md.padded_dims()[d + 1];
            }();
        }
        return ret;
    }

    const memory_desc_wrapper &md;
};

inline outer_strides_getter_t get_outer_strides(const memory_desc_wrapper &md) {
    return {md};
}

inline block_layout_t get_inner_layout(const memory_desc_wrapper &md) {
    block_layout_t inner_layout(md, /* inner_only */ true);

    block_layout_t ret;
    // Explicitly initialize to size-1 blocks
    for (int d = 0; d < MAX_NDIMS; d++) {
        ret.append(block_t(d, 1, 0));
    }

    // Overwrite inner blocks with their actual values
    for (const auto &block : inner_layout) {
        ret[block.dim_idx] = block;
    }

    return ret;
}

inline void def_offsets(const dim_t offs[4][MAX_NDIMS],
        compute::kernel_ctx_t &kernel_ctx, const char *str,
        const dim_idx_t ndims) {

    for (dim_idx_t d = 0; d < MAX_NDIMS; d++) {
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

inline void def_block_offsets(const block_layout_t &layout,
        compute::kernel_ctx_t &kernel_ctx, const char *str) {

    for (const block_t &b : layout) {
        kernel_ctx.define_int(utils::format("%s_B%d", str, b.dim_idx), b.block);
        kernel_ctx.define_int(
                utils::format("%s_SB%d", str, b.dim_idx), b.stride);
    }
}

inline void def_data_type(compute::kernel_ctx_t &kernel_ctx, data_type_t dt,
        const char *str, bool with_punning = true) {
    const char *bf16_name = with_punning ? "ushort" : "bf16";
    const char *bf8_name = with_punning ? "uchar" : "f8_e5m2";
    const char *hf8_name = with_punning ? "uchar" : "f8_e4m3";
    const char *f4_e2m1_name = with_punning ? "uchar" : "f4_e2m1";
    const char *e8m0_name = with_punning ? "uchar" : "e8m0";
    const char *u4_name = with_punning ? "uchar" : "u4";
    const char *s4_name = with_punning ? "uchar" : "s4";

    switch (dt) {
        case data_type::undef:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=void -D%s_DT_UNDEF", str, str));
            break;
        case data_type::bf16:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_BF16", str, bf16_name, str));
            break;
        case data_type::f16:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=half -D%s_DT_F16", str, str));
            break;
        case data_type::f32:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=float -D%s_DT_F32", str, str));
            break;
        case data_type::f64:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=double -D%s_DT_F64", str, str));
            break;
        case data_type::s8:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=char -D%s_DT_S8", str, str));
            break;
        case data_type::u8:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=uchar -D%s_DT_U8", str, str));
            break;
        case data_type::f8_e4m3:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_HF8", str, hf8_name, str));
            break;
        case data_type::f8_e5m2:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_BF8", str, bf8_name, str));
            break;
        case data_type::f4_e2m1:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_F4_E2M1", str, f4_e2m1_name, str));
            break;
        case data_type::e8m0:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_E8M0", str, e8m0_name, str));
            break;
        case data_type::s4:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_S4", str, s4_name, str));
            break;
        case data_type::u4:
            kernel_ctx.add_option(utils::format(
                    "-D%s_DATA_T=%s -D%s_DT_U4", str, u4_name, str));
            break;
        case data_type::s32:
            kernel_ctx.add_option(
                    utils::format("-D%s_DATA_T=int -D%s_DT_S32", str, str));
            break;
        default:
            gpu_error_not_expected()
                    << "Unexpected data type " << dnnl_dt2str(dt);
            break;
    }
}

inline void def_memory_desc_info(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_info_t &md_info, const char *prefix) {
    def_data_type(kernel_ctx, md_info.data_type, prefix);
    kernel_ctx.register_buffer_size(md_info.size);

    kernel_ctx.define_int(utils::format("%s_OFFSET0", prefix), md_info.offset0);
    kernel_ctx.define_int(utils::format("%s_NDIMS", prefix), md_info.ndims);

    kernel_ctx.define_int(utils::format("%s_NLEVELS", prefix), md_info.nlevels);

    for (int d = 0; d < MAX_NDIMS; ++d) {
        dim_t dim = (d < md_info.ndims) ? md_info.dims[d] : 1;
        dim_t padded_dim = (d < md_info.ndims) ? md_info.padded_dims[d] : 1;
        kernel_ctx.define_int(utils::format("%s_D%d", prefix, d), dim);
        kernel_ctx.define_int(utils::format("%s_PD%d", prefix, d), padded_dim);

        for (int l = 0; l < md_info.nlevels + 1; ++l) {
            dim_t block = (d < md_info.ndims) ? md_info.blocks[d][l] : 1;
            dim_t stride = (d < md_info.ndims) ? md_info.strides[d][l] : 0;
            kernel_ctx.define_int(
                    utils::format("%s_B%d_%d", prefix, d, l), block);
            kernel_ctx.define_int(
                    utils::format("%s_S%d_%d", prefix, d, l), stride);
        }
    }
}

inline void def_binary_alg_kinds(compute::kernel_ctx_t &kernel_ctx) {
    kernel_ctx.define_int("BINARY_ADD", alg_kind::binary_add);
    kernel_ctx.define_int("BINARY_MUL", alg_kind::binary_mul);
    kernel_ctx.define_int("BINARY_MIN", alg_kind::binary_min);
    kernel_ctx.define_int("BINARY_MAX", alg_kind::binary_max);
    kernel_ctx.define_int("BINARY_DIV", alg_kind::binary_div);
    kernel_ctx.define_int("BINARY_SUB", alg_kind::binary_sub);
    kernel_ctx.define_int("BINARY_GE", alg_kind::binary_ge);
    kernel_ctx.define_int("BINARY_GT", alg_kind::binary_gt);
    kernel_ctx.define_int("BINARY_LE", alg_kind::binary_le);
    kernel_ctx.define_int("BINARY_LT", alg_kind::binary_lt);
    kernel_ctx.define_int("BINARY_EQ", alg_kind::binary_eq);
    kernel_ctx.define_int("BINARY_NE", alg_kind::binary_ne);
}

inline void def_eltwise_alg_kinds(compute::kernel_ctx_t &kernel_ctx) {
    kernel_ctx.define_int("RELU", alg_kind::eltwise_relu);
    kernel_ctx.define_int("LINEAR", alg_kind::eltwise_linear);
    kernel_ctx.define_int("SOFT_RELU", alg_kind::eltwise_soft_relu);
    kernel_ctx.define_int("MISH", alg_kind::eltwise_mish);
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
    kernel_ctx.define_int("CLIP_V2", alg_kind::eltwise_clip_v2);
    kernel_ctx.define_int("POW", alg_kind::eltwise_pow);
    kernel_ctx.define_int("GELU_ERF", alg_kind::eltwise_gelu_erf);
    kernel_ctx.define_int("ROUND", alg_kind::eltwise_round);
    kernel_ctx.define_int("HARDSWISH", alg_kind::eltwise_hardswish);
    kernel_ctx.define_int("HARDSIGMOID", alg_kind::eltwise_hardsigmoid);

    kernel_ctx.define_int("RELU_DST", alg_kind::eltwise_relu_use_dst_for_bwd);
    kernel_ctx.define_int(
            "LOGISTIC_DST", alg_kind::eltwise_logistic_use_dst_for_bwd);
    kernel_ctx.define_int("TANH_DST", alg_kind::eltwise_tanh_use_dst_for_bwd);
    kernel_ctx.define_int("ELU_DST", alg_kind::eltwise_elu_use_dst_for_bwd);
    kernel_ctx.define_int("SQRT_DST", alg_kind::eltwise_sqrt_use_dst_for_bwd);
    kernel_ctx.define_int("EXP_DST", alg_kind::eltwise_exp_use_dst_for_bwd);
    kernel_ctx.define_int(
            "CLIP_V2_DST", alg_kind::eltwise_clip_v2_use_dst_for_bwd);
}

inline bool post_ops_with_binary_ok(const primitive_attr_t *attr,
        const data_type_t dst_dt, const int max_ndims_supported = 2,
        const int prelu_mask_supported = 3) {
    const auto &p = attr->post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
    auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false); };
    auto is_binary = [&](int idx) { return p.entry_[idx].is_binary(); };
    auto is_prelu = [&](int idx) { return p.entry_[idx].is_prelu(); };

    bool is_po_ok = true;
    for (int po_idx = 0; po_idx < p.len(); ++po_idx) {
        is_po_ok = is_po_ok
                && (is_eltwise(po_idx) || is_sum(po_idx) || is_binary(po_idx)
                        || is_prelu(po_idx));
        if (is_binary(po_idx)) {
            const auto &bin_desc = p.entry_[po_idx].binary.src1_desc;
            if (bin_desc.ndims > max_ndims_supported) {
                // accept descriptor if unsupported dims are equal to 1.
                for (int dim_idx = max_ndims_supported;
                        dim_idx < bin_desc.ndims; ++dim_idx) {
                    if (bin_desc.dims[dim_idx] != 1) is_po_ok = false;
                }
            }
        }
        if (is_prelu(po_idx)) {
            if (p.entry_[po_idx].prelu.mask > prelu_mask_supported)
                is_po_ok = false;
        }
        if (is_sum(po_idx)) {
            if (p.entry_[po_idx].sum.zero_point != 0) return false;
            if (p.entry_[po_idx].sum.dt != dnnl_data_type_undef
                    && types::data_type_size(p.entry_[po_idx].sum.dt)
                            != types::data_type_size(dst_dt))
                return false;
        }
    }

    if (p.len() > MAX_POST_OPS_SUPPORTED) is_po_ok = false;
    if (dst_dt == dnnl_f64 && !p.has_default_values()) is_po_ok = false;

    return is_po_ok;
}

constexpr int prelu_max_ndims = 5;
inline status_t get_prelu_md(int prelu_mask, const dim_t *dst_dims,
        memory_desc_t &weight_mem_desc, int weight_ndims) {
    format_tag_t weights_tag;
    dims_t weight_dims {};
    for (int d = 0; d < weight_ndims; ++d) {
        if (((prelu_mask >> d) & 0x1) == 1) {
            weight_dims[d] = dst_dims[d];
        } else {
            weight_dims[d] = 1;
        }
    }
    switch (weight_ndims) {
        case 1: weights_tag = format_tag_t::dnnl_a; break;
        case 2: weights_tag = format_tag_t::dnnl_ab; break;
        case 3: weights_tag = format_tag_t::dnnl_acb; break;
        case 4: weights_tag = format_tag_t::dnnl_acdb; break;
        case 5: weights_tag = format_tag_t::dnnl_acdeb; break;
        default: weights_tag = format_tag_t::dnnl_format_tag_undef; break;
    }
    CHECK(memory_desc_init_by_tag(weight_mem_desc, weight_ndims, weight_dims,
            data_type_t::dnnl_f32, weights_tag));
    return status::success;
}

inline status_t def_post_ops_cfg(compute::kernel_ctx_t &kernel_ctx,
        const post_ops_t &post_ops, const memory_desc_t &dst_md) {
    const int po_nop_id = 0;
    const int po_binary_id = 1;
    const int po_eltwise_id = 2;
    const int po_sum_id = 3;

    kernel_ctx.define_int("PO_BINARY", po_binary_id);
    kernel_ctx.define_int("PO_ELTWISE", po_eltwise_id);
    kernel_ctx.define_int("PO_SUM", po_sum_id);

    std::string po_kernel_args = "-DPOST_OP_ARGS=\"";
    int nof_supported_post_ops = 0;

    bool post_op_uses_bf16 = false;
    bool post_op_uses_bf8 = false;
    bool post_op_uses_hf8 = false;

    auto add_po_defines = [&](const std::string &bin_arg_name,
                                  const post_ops_t::entry_t &e, int idx) {
        auto define_binary_po_type = [&](data_type_t type) {
            def_data_type(kernel_ctx, type,
                    utils::format("%s_ACTUAL", bin_arg_name).c_str(), false);
            post_op_uses_bf16 |= (type == data_type::bf16);
            post_op_uses_bf8 |= (type == data_type::f8_e5m2);
            post_op_uses_hf8 |= (type == data_type::f8_e4m3);
        };
        if (e.is_binary()) {
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_KIND", po_binary_id);
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_ALG", e.binary.alg);

            const memory_desc_wrapper src1_mdw(e.binary.src1_desc);
            const auto mdi = memory_desc_info_t::create(src1_mdw);
            def_memory_desc_info(kernel_ctx, mdi, bin_arg_name.c_str());
            define_binary_po_type(mdi.data_type);
        } else if (e.is_prelu()) {
            // binary && eltwise relu = prelu post op
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_KIND", po_binary_id);
            kernel_ctx.define_int("PO_" + std::to_string(idx) + "_ALG",
                    alg_kind_t::dnnl_eltwise_relu);

            memory_desc_t weight_mem_desc;
            int weight_ndims = dst_md.ndims;
            CHECK(get_prelu_md(
                    e.prelu.mask, dst_md.dims, weight_mem_desc, weight_ndims));
            const memory_desc_wrapper weight_mdw(weight_mem_desc);
            const auto mdi = memory_desc_info_t::create(weight_mdw);
            def_memory_desc_info(kernel_ctx, mdi, bin_arg_name.c_str());

            // prelu weights are assumed to be f32
            define_binary_po_type(data_type::f32);
        } else {
            memory_desc_t empty_mem_desc;
            dnnl_dims_t empty_dims = {1, 1, 1, 1};
            CHECK(memory_desc_init_by_tag(empty_mem_desc, 4, empty_dims,
                    data_type_t::dnnl_s8, format_tag_t::dnnl_nchw));
            const memory_desc_wrapper src1_mdw(empty_mem_desc);
            const auto mdi = memory_desc_info_t::create(src1_mdw);
            def_memory_desc_info(kernel_ctx, mdi, bin_arg_name.c_str());

            // unused - just need any type that's convertible to float
            define_binary_po_type(data_type::f32);
        }
        if (e.is_eltwise(false)) {
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_KIND", po_eltwise_id);
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_ALG", e.eltwise.alg);
            kernel_ctx.define_float(
                    ("PO_" + std::to_string(idx) + "_ELTWISE_ALPHA").c_str(),
                    e.eltwise.alpha);
            kernel_ctx.define_float(
                    ("PO_" + std::to_string(idx) + "_ELTWISE_BETA").c_str(),
                    e.eltwise.beta);
            kernel_ctx.define_float(
                    ("PO_" + std::to_string(idx) + "_ELTWISE_SCALE").c_str(),
                    e.eltwise.scale);
        } else {
            kernel_ctx.define_float(
                    ("PO_" + std::to_string(idx) + "_ELTWISE_ALPHA").c_str(),
                    1.0f);
            kernel_ctx.define_float(
                    ("PO_" + std::to_string(idx) + "_ELTWISE_BETA").c_str(),
                    0.0f);
            kernel_ctx.define_float(
                    ("PO_" + std::to_string(idx) + "_ELTWISE_SCALE").c_str(),
                    1.0f);
        }
        if (e.is_sum(false)) {
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_KIND", po_sum_id);
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_ALG", alg_kind::undef);
            kernel_ctx.define_float(
                    ("PO_" + std::to_string(idx) + "_SUM_SCALE").c_str(),
                    e.sum.scale);
        } else {
            kernel_ctx.define_float(
                    ("PO_" + std::to_string(idx) + "_SUM_SCALE").c_str(), 1.0f);
        }
        if (!(e.is_binary() || e.is_eltwise(false) || e.is_sum(false)
                    || e.is_prelu())) {
            // empty post op
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_KIND", po_nop_id);
            // *_ALG need to be set but it's unused when kind is NOP
            kernel_ctx.define_int(
                    "PO_" + std::to_string(idx) + "_ALG", alg_kind::undef);
            --nof_supported_post_ops;
        }
        po_kernel_args += ", const __global PO_" + std::to_string(idx)
                + "_BIN_ARG_DATA_T *po_" + std::to_string(idx) + "_binary_arg";
        return status::success;
    };

    for (int idx = 0; idx < post_ops.len(); ++idx, ++nof_supported_post_ops) {
        const std::string bin_arg_name
                = "PO_" + std::to_string(idx) + "_BIN_ARG";
        CHECK(add_po_defines(bin_arg_name, post_ops.entry_[idx], idx));
    }

    kernel_ctx.define_int("POST_OP_CHAIN_LENGTH", nof_supported_post_ops);
    if (post_op_uses_bf16) kernel_ctx.define_int("POST_OP_USING_BF16", 1);
    if (post_op_uses_bf8) kernel_ctx.define_int("POST_OP_USING_BF8", 1);
    if (post_op_uses_hf8) kernel_ctx.define_int("POST_OP_USING_HF8", 1);

    po_kernel_args += "\"";
    kernel_ctx.add_option(po_kernel_args);
    return status::success;
}

inline int append_post_ops_to_arg_list_base(const exec_args_t &args,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops) {
    auto set_arg_entry = [&](const post_ops_t::entry_t &e, int po_idx) {
        if (e.is_binary()) {
            auto arg = args.at(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx) | DNNL_ARG_SRC_1);
            gpu_assert(arg.is_const);

            auto &binary_arg = arg.mem
                    ? *(arg.mem->memory_storage())
                    : dnnl::impl::memory_storage_t::empty_storage();
            arg_list.set(post_op_idx++, binary_arg);
        } else if (e.is_prelu()) {
            auto arg = args.at(
                    DNNL_ARG_ATTR_MULTIPLE_POST_OP(po_idx) | DNNL_ARG_WEIGHTS);
            gpu_assert(arg.is_const);
            auto &prelu_wei_arg = arg.mem
                    ? *(arg.mem->memory_storage())
                    : dnnl::impl::memory_storage_t::empty_storage();
            arg_list.set(post_op_idx++, prelu_wei_arg);
        } else {
            arg_list.set(post_op_idx++, memory_storage_t::empty_storage());
        }
    };

    for (int idx = 0; idx < post_ops.len(); ++idx) {
        set_arg_entry(post_ops.entry_[idx], idx);
    }
    return post_op_idx;
}
inline int append_post_ops_to_arg_list_gemm(const exec_args_t &args,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops) {
    return append_post_ops_to_arg_list_base(
            args, arg_list, post_op_idx, post_ops);
}
inline int append_post_ops_to_arg_list(const exec_ctx_t &ctx,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops) {
    exec_args_t args;
    return append_post_ops_to_arg_list_base(
            ctx.args(), arg_list, post_op_idx, post_ops);
}

inline bool post_ops_preserves_zeroes(
        const exec_ctx_t &ctx, const post_ops_t &post_ops) {
    bool preserve_zeroes = true;
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        const post_ops_t::entry_t &po_entry = post_ops.entry_[idx];
        if (po_entry.is_binary()) {
            // only binary mul is preserving zeroes
            preserve_zeroes &= po_entry.binary.alg
                    == dnnl::impl::alg_kind_t::dnnl_binary_mul;
        }
        if (po_entry.is_eltwise(false)) {
            preserve_zeroes &= gpu_eltwise_fwd_pd_t::eltwise_preserves_zero(
                    po_entry.eltwise.alg, po_entry.eltwise.alpha,
                    po_entry.eltwise.beta);
        }
    }
    return preserve_zeroes;
}

inline status_t def_attr_info_impl(compute::kernel_ctx_t &kernel_ctx,
        const attr_info_t &attr_info, const post_ops_t &post_ops,
        const memory_desc_t &dst_md) {
    gpu_assert(attr_info.initialized);

    kernel_ctx.define_int("WITH_POST_OP", post_ops.len() > 0);

    kernel_ctx.define_int("WITH_ELTWISE", attr_info.with_eltwise);
    kernel_ctx.define_int("ELTWISE_IDX", attr_info.eltwise_idx);
    kernel_ctx.define_int("ELTWISE_ALG", attr_info.eltwise_alg);

    kernel_ctx.define_int("WITH_SUM", attr_info.with_sum);
    kernel_ctx.define_int("SUM_IDX", attr_info.sum_idx);
    kernel_ctx.define_float("SUM_SCALE", attr_info.sum_scale);
    kernel_ctx.define_int("SUM_SCALE1", attr_info.sum_scale == 1.0f);

    kernel_ctx.define_int("WITH_SRC0_SCALE", attr_info.with_src0_scale);
    kernel_ctx.define_int("WITH_SRC1_SCALE", attr_info.with_src1_scale);

    kernel_ctx.define_int("WITH_SRC_SCALES", attr_info.with_src_scales);
    kernel_ctx.define_int("WITH_WEI_SCALES", attr_info.with_wei_scales);
    kernel_ctx.define_int("WITH_DST_SCALES", attr_info.with_dst_scales);
    kernel_ctx.define_int("SRC_SCALES_MASK", attr_info.src_scales_mask);
    kernel_ctx.define_int("WEI_SCALES_MASK", attr_info.wei_scales_mask);

    kernel_ctx.define_int("WITH_SRC_ZPOINTS", attr_info.with_src_zpoints);
    kernel_ctx.define_int("WITH_WEI_ZPOINTS", attr_info.with_wei_zpoints);
    kernel_ctx.define_int("WITH_DST_ZPOINTS", attr_info.with_dst_zpoints);
    kernel_ctx.define_int(
            "WITH_SRC_ZPOINTS_PER_IC", attr_info.with_per_ic_src_zpoints);
    kernel_ctx.define_int(
            "WITH_DST_ZPOINTS_PER_OC", attr_info.with_per_oc_dst_zpoints);
    kernel_ctx.define_int("WITH_WEI_ZPOINTS_DT_S8",
            attr_info.wei_zpoints_data_type == dnnl_s8);
    kernel_ctx.define_int("WITH_WEI_ZPOINTS_DT_U8",
            attr_info.wei_zpoints_data_type == dnnl_u8);

    def_binary_alg_kinds(kernel_ctx);
    def_eltwise_alg_kinds(kernel_ctx);

    return def_post_ops_cfg(kernel_ctx, post_ops, dst_md);
}

inline status_t def_attr_info(compute::kernel_ctx_t &kernel_ctx,
        const attr_info_t &attr_info, const post_ops_t &post_ops,
        const memory_desc_t &dst_md) {
    return def_attr_info_impl(kernel_ctx, attr_info, post_ops, dst_md);
}

inline void def_dispatch(compute::kernel_ctx_t &kernel_ctx,
        const compute::dispatch_t &dispatch) {
    dispatch.def_kernel_macros(kernel_ctx);
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
