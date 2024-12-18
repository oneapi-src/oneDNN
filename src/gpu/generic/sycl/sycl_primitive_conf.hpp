/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_SYCL_PRIMITIVE_CONF_HPP
#define GPU_GENERIC_SYCL_SYCL_PRIMITIVE_CONF_HPP

#include "common/broadcast_strategy.hpp"
#include "gpu/generic/sycl/sycl_post_ops.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct sycl_binary_conf_t {
    xpu::sycl::md_t src0_md;
    xpu::sycl::md_t src1_md;
    xpu::sycl::md_t dst_md;

    alg_kind_t alg_kind;

    bool do_scale_src0;
    bool do_scale_src1;

    int broadcast_dims0[xpu::sycl::md_t::max_dims];
    int broadcast_dims1[xpu::sycl::md_t::max_dims];
    int ndims;
    bool is_tensor_op;

    int wk_size;

    sycl_post_ops_t post_ops;
};

struct sycl_convolution_common_conf_t {
    bool has_bias = false;
    data_type_t bias_dt;

    int padding[3];
    int strides[3];
    int dilation[3];

    bool do_scale_data;
    bool do_scale_weights;
    bool do_scale_dst;
    bool single_weight_scale;

    bool use_data_zeropoints;
    bool use_dst_zeropoints;
    bool single_data_zeropoint;
    bool single_dst_zeropoint;

    int ndims;

    int wk_size;
    bool has_groups;

    bool is_deconvolution;

    sycl_post_ops_t post_ops;
};

struct sycl_convolution_fwd_conf_t : sycl_convolution_common_conf_t {
    xpu::sycl::md_t data_md;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t weights_md;
};

struct sycl_convolution_bwd_data_conf_t : sycl_convolution_common_conf_t {
    xpu::sycl::md_t weights_md;
    xpu::sycl::md_t diff_data_md;
    xpu::sycl::md_t diff_dst_md;
};

struct sycl_convolution_bwd_weights_conf_t : sycl_convolution_common_conf_t {
    xpu::sycl::md_t data_md;
    xpu::sycl::md_t diff_dst_md;
    xpu::sycl::md_t diff_weights_md;
};

struct sycl_eltwise_conf_t {
    prop_kind_t prop_kind;
    xpu::sycl::md_t src_md;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t diff_src_md;
    xpu::sycl::md_t diff_dst_md;
    alg_kind_t alg_kind;
    float alpha;
    float beta;
    dim_t mb;
    dim_t c;
    dim_t d;
    dim_t h;
    dim_t w;
    dim_t wk_size;
    sycl_post_ops_t post_ops;
};

struct sycl_matmul_conf_t {
    xpu::sycl::md_t data_md;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t weights_md;
    xpu::sycl::md_t bias_md;
    alg_kind_t alg_kind;
    bool transpose_data; //TODO can we remove?
    bool transpose_dst;
    bool transpose_weights;
    bool transpose_bias;
    sycl_post_ops_t post_ops;
    int wk_size;

    int data_mask;
    int weights_mask;
    int bias_mask;

    bool do_scale_data;
    bool do_scale_weights;
    bool do_scale_dst;
    bool single_weights_scale;

    bool use_data_zeropoints;
    bool use_weights_zeropoints;
    bool use_dst_zeropoints;

    bool use_dropout;
};

struct sycl_prelu_conf_t {
    prop_kind_t prop_kind;
    xpu::sycl::md_t data_md;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t weights_md;
    xpu::sycl::md_t diff_data_md;
    xpu::sycl::md_t diff_dst_md;
    xpu::sycl::md_t diff_weights_md;
    dim_t work_amount;
    dim_t work_amount_wei;
    dim_t work_amount_src;
    dim_t work_load;
    int mask;
    float sum;
    broadcasting_strategy_t bcast_type;
    int ndims;
    int block_size;
    int wg_size;
    size_t n_thr;
    size_t i_thr;
};

struct sycl_shuffle_conf_t {
    xpu::sycl::md_t src_md;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t stat_md;
    xpu::sycl::md_t axis_md;
    dim_t transpose_col;
    dim_t transpose_row;
    dim_t group_size;

    dim_t outer_size;
    dim_t inner_size;
    dim_t stride_m;
    dim_t blksize;
    format_tag_t tag;
    int axis;
    int axis_size;
    dim_t MB;
    dim_t C;
    dim_t H;
    dim_t W;
    dim_t D;
    dim_t HW;
    dim_t SP;

    int ndims;
    int ndims_d;
    dim_t dims;
    size_t nthr;
    int block_size;
    int wg_size;
    dim_t work_amount;
};

struct sycl_reorder_conf_t {
    xpu::sycl::md_t src_md;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t scales;

    bool do_scale_src;
    int scale_src_mask;
    bool do_scale_dst;
    int scale_dst_mask;

    int ndims;

    int wk_size;

    sycl_post_ops_t post_ops;
};

struct sycl_resampling_conf_t {
    dims_t dst_dims;
    int dst_ndims;
    size_t work_amount;

    xpu::sycl::md_t src_md;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t diff_src_md;
    xpu::sycl::md_t diff_dst_md;

    alg_kind_t alg;
    float src_scale;
    bool do_scale_src;
    int broadcast_dims[xpu::sycl::md_t::max_dims];
    bool is_tensor_op;

    int block_size;
    int wg_size;
    size_t n_thr;

    sycl_post_ops_t post_ops;
};

struct sycl_layer_normalization_conf_t {
    prop_kind_t prop_kind;
    xpu::sycl::md_t data_md;
    xpu::sycl::md_t diff_data_md;
    xpu::sycl::md_t data_scaleshift_md;
    xpu::sycl::md_t diff_data_scaleshift_md;
    xpu::sycl::md_t scale;
    xpu::sycl::md_t shift;
    xpu::sycl::md_t stat_md;
    data_type_t var_dt;
    data_type_t scales_src_dt;
    data_type_t scales_dst_dt;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t diff_dst_md;
    dim_t wk_size;
    bool is_fwd;
    bool src_def;
    bool dst_def;
    size_t n_thr;
    size_t n_thr2;
    size_t diff_shift_off;
    bool rt_scaling = false;
    int scale_count;
    float oscale = 1.0f;
    dim_t N;
    dim_t C;
    bool use_ss;
    bool use_scale;
    bool use_shift;
    dim_t wei_shift_off;
    bool calculate_stats;
    bool calculate_diff_stats;
    bool save_stats;
    int shift_off;
    bool zero_dims;
    int ss_off;
    float layer_norm_epsilon;
    unsigned flags;
    int ndims;
    int block_size;
    int wg_size;
};

struct sycl_batch_normalization_conf_t {
    prop_kind_t prop_kind;
    int ndims;
    size_t n_thr;
    unsigned flags;
    size_t wk_size;
    int block_size;
    int wg_size;
    bool use_scale;
    bool use_shift;
    float alpha;
    bool dir;
    xpu::sycl::md_t data_md;
    xpu::sycl::md_t src1_md;
    xpu::sycl::md_t diff_data_md;
    data_type_t diff_src1_dt;
    xpu::sycl::md_t data_scaleshift_md;
    xpu::sycl::md_t diff_data_scaleshift_md;
    xpu::sycl::md_t stat_md;
    xpu::sycl::md_t var_md;
    data_type_t ws_dt;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t diff_dst_md;
    dim_t N;
    dim_t C;
    dim_t D;
    dim_t H;
    dim_t W;
    float batch_norm_epsilon;
    bool save_stats;
    bool calculate_stats;
    bool calculate_diff_stats;
    bool fuse_norm_relu;
    bool fuse_norm_add_relu;
    bool zero_dims;
    bool is_training;
    bool with_relu;
};

struct sycl_softmax_conf_t {
    prop_kind_t prop_kind;
    xpu::sycl::md_t src_md;
    xpu::sycl::md_t dst_md;

    xpu::sycl::md_t diff_md;
    xpu::sycl::md_t diff_src_md;
    xpu::sycl::md_t diff_dst_md;
    alg_kind_t alg_kind;
    dim_t wk_size;

    dim_t axis;
    dim_t axis_size;
    dim_t inner_size;
    dim_t outer_size;
    dim_t channels;
    bool do_scale_src;
    bool do_scale_dst;

    sycl_post_ops_t post_ops;
};

struct sycl_lrn_conf_t {
    xpu::sycl::md_t src_md;
    xpu::sycl::md_t dst_md;
    xpu::sycl::md_t diff_dst_md;
    xpu::sycl::md_t diff_src_md;
    alg_kind_t alg_kind;

    dim_t mb;
    dim_t c;
    dim_t d;
    dim_t h;
    dim_t w;
    dim_t stride_mb;
    dim_t ndims;
    dim_t tag_blk_sz;
    dim_t size;
    dim_t compute_n_summands;
    float alpha;
    float beta;
    float k;

    int wk_size;
};

struct sycl_pooling_base_conf_t {
    xpu::sycl::md_t ws_md;
    int ndims;
    bool zero_dims;
    int block_size;
    int wg_size;
    size_t n_thr;
    alg_kind_t alg;
    dim_t MB;
    dim_t OC;
    dim_t OD;
    dim_t OH;
    dim_t OW;
    dim_t ID;
    dim_t IH;
    dim_t IW;
    dim_t KD;
    dim_t KH;
    dim_t KW;
    dim_t SD;
    dim_t SH;
    dim_t SW;
    dim_t padF;
    dim_t padT;
    dim_t padL;
    dim_t DD;
    dim_t DH;
    dim_t DW;
};

struct sycl_pooling_fwd_conf_t : public sycl_pooling_base_conf_t {
    xpu::sycl::md_t src_md;
    xpu::sycl::md_t dst_md;
    sycl_post_ops_t post_ops;
};

// Intel GPU kernel fails to run with more than 8 tensors.
#define DNNL_REF_SUM_MAX_NUM_TENSORS 8

struct sycl_sum_conf_t {
    xpu::sycl::md_t src_md[DNNL_REF_SUM_MAX_NUM_TENSORS];
    xpu::sycl::md_t dst_md;
    float src_scales[DNNL_REF_SUM_MAX_NUM_TENSORS];
    int n;
    int wk_size;
};

struct sycl_pooling_bwd_conf_t : public sycl_pooling_base_conf_t {
    xpu::sycl::md_t diff_src_md;
    xpu::sycl::md_t diff_dst_md;
};

struct sycl_simple_reduction_conf_t {
    dnnl_alg_kind_t alg = dnnl_alg_kind_undef;
    xpu::sycl::md_t src_md;
    xpu::sycl::md_t dst_md;
    float p;
    float eps;
    sycl_post_ops_t post_ops;
    dim_t reduce_dims[xpu::sycl::md_t::max_dims];
    int reduce_size = 1;
};

CHECK_SYCL_KERNEL_ARG_TYPE(sycl_binary_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_prelu_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_shuffle_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_resampling_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_batch_normalization_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_softmax_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_layer_normalization_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_eltwise_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_lrn_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_sum_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_pooling_base_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_pooling_fwd_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_pooling_bwd_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_convolution_fwd_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_convolution_bwd_data_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_convolution_bwd_weights_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_simple_reduction_conf_t);

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
