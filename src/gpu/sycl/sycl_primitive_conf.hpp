/*******************************************************************************
* Copyright 2022-2023 Intel Corporation
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

#ifndef GPU_SYCL_SYCL_PRIMITIVE_CONF_HPP
#define GPU_SYCL_SYCL_PRIMITIVE_CONF_HPP

#include "common/broadcast_strategy.hpp"
#include "gpu/sycl/sycl_post_ops.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

struct sycl_binary_conf_t {
    sycl_md_t src0_md;
    sycl_md_t src1_md;
    sycl_md_t dst_md;

    alg_kind_t alg_kind;

    bool do_scale_src0;
    bool do_scale_src1;

    int broadcast_dims[sycl_md_t::max_dims];
    int ndims;
    bool is_tensor_op;

    int block_size;
    int wg_size;
    int wk_size;

    sycl_post_ops_t post_ops;
};

struct sycl_eltwise_conf_t {
    prop_kind_t prop_kind;
    sycl_md_t src_md;
    sycl_md_t dst_md;
    sycl_md_t diff_src_md;
    sycl_md_t diff_dst_md;
    alg_kind_t alg_kind;
    float alpha;
    float beta;
    dim_t mb;
    dim_t c;
    dim_t d;
    dim_t h;
    dim_t w;
    dim_t block_size;
    dim_t wg_size;
    dim_t wk_size;
    dim_t post_po_len;
    sycl_md_t binary_src_arr[8];
    sycl_post_ops_t post_ops;
};

struct sycl_prelu_conf_t {
    prop_kind_t prop_kind;
    sycl_md_t data_md;
    sycl_md_t dst_md;
    sycl_md_t weights_md;
    sycl_md_t diff_data_md;
    sycl_md_t diff_dst_md;
    sycl_md_t diff_weights_md;
    dim_t work_amount;
    dim_t work_amount_wei;
    dim_t work_amount_src;
    dim_t work_load;
    bool reduce_diff_weights = 0;
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
    sycl_md_t src_md;
    sycl_md_t dst_md;
    sycl_md_t stat_md;
    sycl_md_t axis_md;
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

struct sycl_resampling_conf_t {
    dim_t MB;
    dim_t C;
    dim_t ID;
    dim_t IH;
    dim_t IW;
    dim_t OD;
    dim_t OH;
    dim_t OW;
    dims_t dst_dims;
    int dst_ndims;
    int po_len;
    size_t work_amount;

    data_type_t src_dt;
    data_type_t dst_dt;

    sycl_md_t src_md;
    sycl_md_t src1_md[8];
    sycl_md_t dst_md;
    sycl_md_t diff_src_md;
    sycl_md_t diff_dst_md;

    alg_kind_t alg;
    float src_scale;
    bool do_scale_src;
    int broadcast_dims[sycl_md_t::max_dims];
    int ndims;
    bool is_tensor_op;

    int block_size;
    int wg_size;
    size_t n_thr;

    sycl_post_ops_t post_ops;
};

struct sycl_layer_normalization_conf_t {
    prop_kind_t prop_kind;
    sycl_md_t data_md;
    sycl_md_t diff_data_md;
    sycl_md_t data_scaleshift_md;
    sycl_md_t diff_data_scaleshift_md;
    sycl_md_t scale;
    sycl_md_t shift;
    sycl_md_t stat_md;
    sycl_md_t stat_d;
    sycl_md_t var_md;
    sycl_md_t dst_md;
    sycl_md_t diff_dst_md;
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
    sycl_md_t data_md;
    sycl_md_t src1_md;
    sycl_md_t dst1_md;
    sycl_md_t diff_data_md;
    sycl_md_t diff_src1_md;
    sycl_md_t data_scaleshift_md;
    sycl_md_t diff_data_scaleshift_md;
    sycl_md_t stat_md;
    sycl_md_t var_md;
    sycl_md_t ws_md;
    sycl_md_t dst_md;
    sycl_md_t diff_dst_md;
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
    sycl_md_t src_md;
    sycl_md_t dst_md;

    sycl_md_t diff_md;
    sycl_md_t diff_src_md;
    sycl_md_t diff_dst_md;
    alg_kind_t alg_kind;
    dim_t block_size;
    dim_t wg_size;
    dim_t wk_size;

    dim_t axis;
    dim_t axis_size;
    dim_t inner_size;
    dim_t outer_size;
    dim_t channels;
    bool do_scale_src;
    bool do_scale_dst;
};

struct sycl_lrn_conf_t {
    sycl_md_t src_md;
    sycl_md_t dst_md;
    sycl_md_t diff_dst_md;
    sycl_md_t diff_src_md;
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

    int block_size;
    int wg_size;
    int wk_size;
};

struct sycl_pooling_conf_t {
    sycl_md_t src_md;
    sycl_md_t src1_md[8];
    sycl_md_t dst_md;
    sycl_md_t ws_md;
    sycl_md_t diff_src_md;
    sycl_md_t diff_dst_md;
    int ndims;
    int po_len;
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
    dims_t dst_dims;
    int dst_ndims;
    sycl_post_ops_t post_ops;
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
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_pooling_conf_t);

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
