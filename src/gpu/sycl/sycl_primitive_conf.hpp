/*******************************************************************************
* Copyright 2023 Intel Corporation
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

    sycl_post_ops_t post_ops;
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

CHECK_SYCL_KERNEL_ARG_TYPE(sycl_binary_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_lrn_conf_t);

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
