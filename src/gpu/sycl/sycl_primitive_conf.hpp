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

CHECK_SYCL_KERNEL_ARG_TYPE(sycl_prelu_conf_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_binary_conf_t);

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
