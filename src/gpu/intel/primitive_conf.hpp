/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#include "gpu/gpu_utils.hpp"

#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/compute/dispatch.hpp"
#include "gpu/intel/compute/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

bool memory_desc_ndims_ok(const memory_desc_t *md);

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

    static memory_desc_info_t create(const memory_desc_wrapper &mdw);
};

struct attr_info_t {
    static attr_info_t create(const primitive_attr_t *attr);

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
    int wei_scales_mask;
    int dst_scales_mask;
    data_type_t src_scales_data_type;
    data_type_t wei_scales_data_type;
    data_type_t dst_scales_data_type;

    bool with_src_zpoints;
    bool with_wei_zpoints;
    bool with_dst_zpoints;
    bool with_per_ic_src_zpoints;
    bool with_per_oc_dst_zpoints;
    data_type_t src_zpoints_data_type;
    data_type_t wei_zpoints_data_type;
    data_type_t dst_zpoints_data_type;
    bool with_dst_sround;
};

template <size_t ndims>
using strides_t = std::array<dim_t, ndims>;
template <>
struct compute::scalar_type_traits_t<strides_t<2>> {
    static const auto type = scalar_type_t::_int64x3_t;
};
template <>
struct compute::scalar_type_traits_t<strides_t<3>> {
    static const auto type = scalar_type_t::_int64x3_t;
};
template <>
struct compute::scalar_type_traits_t<strides_t<4>> {
    static const auto type = scalar_type_t::_int64x4_t;
};
template <>
struct compute::scalar_type_traits_t<strides_t<5>> {
    static const auto type = scalar_type_t::_int64x5_t;
};
template <>
struct compute::scalar_type_traits_t<strides_t<6>> {
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
    bool stochastic_round = false;
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
            compute::kernel_ctx_t &kernel_ctx, const std::string &name) const;
};

struct sum_quantization_t : public gpu::sum_quantization_t {
public:
    using gpu::sum_quantization_t::sum_quantization_t;

    void define_macros(
            compute::kernel_ctx_t &kernel_ctx, const std::string &name) const;
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

void set_default_pool_conf(pool_conf_t &conf, const pooling_desc_t &desc,
        const memory_desc_t &src_md, const memory_desc_t &dst_md,
        const primitive_attr_t &attr);

void set_default_conf(conv_conf_t &conf, const convolution_desc_t &cd,
        const memory_desc_t &src_md, const memory_desc_t &weights_md,
        const memory_desc_t &dst_md, const memory_desc_t &bias_md,
        const primitive_attr_t &attr);

void set_offsets(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_wrapper &md, const char *str);

void set_offsets(const memory_desc_wrapper &md, dim_t offs[4][MAX_NDIMS]);

struct outer_strides_getter_t {
    template <size_t ndims>
    operator strides_t<ndims>() const {
        strides_t<ndims> ret;
        gpu_assert(into<dim_t>(ndims) >= md.ndims());
        for (int d = ndims - 1; d >= 0; d--) {
            // Assumes size 1 dimensions are dense w.r.t. the neighboring dims
            // so they can be used for size calculations in some layouts.
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

outer_strides_getter_t get_outer_strides(const memory_desc_wrapper &md);

block_layout_t get_inner_layout(const memory_desc_wrapper &md);

void def_offsets(const dim_t offs[4][MAX_NDIMS],
        compute::kernel_ctx_t &kernel_ctx, const char *str,
        const dim_idx_t ndims);

void def_block_offsets(const block_layout_t &layout,
        compute::kernel_ctx_t &kernel_ctx, const char *str);

void def_data_type(compute::kernel_ctx_t &kernel_ctx, data_type_t dt,
        const char *str, bool with_punning = true);
void def_data_type(compute::kernel_ctx_t &kernel_ctx, data_type_t dt,
        const std::string &str, bool with_punning = true);

void def_memory_desc_info(compute::kernel_ctx_t &kernel_ctx,
        const memory_desc_info_t &md_info, const char *prefix,
        bool with_punning = true);

void def_binary_alg_kinds(compute::kernel_ctx_t &kernel_ctx);

void def_eltwise_alg_kinds(compute::kernel_ctx_t &kernel_ctx);

bool post_ops_with_binary_ok(const primitive_attr_t *attr,
        const data_type_t dst_dt, const int max_ndims_supported = 2);

constexpr int prelu_max_ndims = 5;
status_t get_prelu_md(int prelu_mask, const dim_t *dst_dims,
        memory_desc_t &weight_mem_desc, int weight_ndims);

status_t def_post_ops_cfg(compute::kernel_ctx_t &kernel_ctx,
        const post_ops_t &post_ops, const memory_desc_t &dst_md);

int append_post_ops_to_arg_list_base(const exec_args_t &args,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops);
int append_post_ops_to_arg_list_gemm(const exec_args_t &args,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops);
int append_post_ops_to_arg_list(const exec_ctx_t &ctx,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops);

bool post_ops_preserves_zeroes(
        const exec_ctx_t &ctx, const post_ops_t &post_ops);

status_t def_attr_info_impl(compute::kernel_ctx_t &kernel_ctx,
        const attr_info_t &attr_info, const post_ops_t &post_ops,
        const memory_desc_t &dst_md, bool with_punning = true);

status_t def_attr_info(compute::kernel_ctx_t &kernel_ctx,
        const attr_info_t &attr_info, const post_ops_t &post_ops,
        const memory_desc_t &dst_md, bool with_punning = true);

void def_dispatch(
        compute::kernel_ctx_t &kernel_ctx, const compute::dispatch_t &dispatch);

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
