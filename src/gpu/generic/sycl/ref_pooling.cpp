/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "gpu/generic/sycl/ref_pooling.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "gpu/generic/sycl/pooling_kernels.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_pooling_fwd_t::pd_t::init_conf() {
    using namespace primitive_kind;
    conf_ = sycl_pooling_fwd_conf_t();
    conf_.ndims = ndims();
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.src_md = xpu::sycl::md_t(src_md(0));
    conf_.dst_md = xpu::sycl::md_t(dst_md(0));
    conf_.ws_md = !types::is_zero_md(workspace_md())
            ? xpu::sycl::md_t(workspace_md(0))
            : xpu::sycl::md_t {};
    conf_.zero_dims = has_zero_dim_memory();
    auto nelems_A = memory_desc_wrapper(src_md(0)).nelems();
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;
    conf_.alg = desc()->alg_kind;
    conf_.MB = MB(); //n: denotes the mini-batch dimension
    conf_.OC = OC(); //c:outputchannel dimension
    conf_.OD = OD(); //d: output spatial depth
    conf_.OH = OH(); //h: ouput height
    conf_.OW = OW(); //w: output width
    conf_.ID = ID(); //d: input spatial depth
    conf_.IH = IH(); //h: input height
    conf_.IW = IW(); //w: input width
    conf_.KD = KD(); //K: kernel D: spatial depth
    conf_.KH = KH(); //k: kernel H: Height
    conf_.KW = KW(); //K: kernel W:width
    conf_.SD = KSD(); //K:kernel S:strides D:Depth
    conf_.SH = KSH(); //K:kernel S:strides H:Height
    conf_.SW = KSW(); //K:kernel S:strides W:Wdith
    conf_.padF = padFront();
    conf_.padT = padT();
    conf_.padL = padL();
    conf_.DD = KDD(); //K:kernel D:Dilation D:Depth
    conf_.DH = KDH(); //K:kernel D:Dilation H:Height
    conf_.DW = KDW(); //K:kernel D:Dilation W:Weight

    conf_.post_ops = sycl_post_ops_t(attr(), dst_md());
    return status::success;
}

status_t ref_pooling_fwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<pooling_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_pooling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    // XXX: Add support for 0-dim src
    for (auto &md : {pd()->src_md(), pd()->dst_md()}) {
        if (memory_desc_wrapper(md).size() == 0) return status::success;
    }

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
        pooling_fwd_kernel_vec_t pooling_fwd_kernel(pd()->conf_, cgh, ctx);

        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        int work_per_wg = wg_size * block_size;
        int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;
        int n_thr = n_wgs * wg_size;
        cgh.parallel_for(
                ::sycl::nd_range<1>(n_thr, wg_size), pooling_fwd_kernel);
    });
}

status_t ref_pooling_bwd_t::pd_t::init_conf() {
    conf_ = sycl_pooling_bwd_conf_t();
    conf_.ndims = ndims();
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.diff_src_md = xpu::sycl::md_t(diff_src_md(0));
    conf_.diff_dst_md = xpu::sycl::md_t(diff_dst_md(0));
    conf_.ws_md = !types::is_zero_md(workspace_md())
            ? xpu::sycl::md_t(workspace_md(0))
            : xpu::sycl::md_t {};
    conf_.zero_dims = has_zero_dim_memory();
    auto nelems_A = memory_desc_wrapper(diff_src_md(0)).nelems();
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;

    conf_.alg = desc()->alg_kind;
    conf_.MB = MB();
    conf_.OC = OC();
    conf_.OD = OD();
    conf_.OH = OH();
    conf_.OW = OW();
    conf_.ID = ID();
    conf_.IH = IH();
    conf_.IW = IW();
    conf_.KD = KD();
    conf_.KH = KH();
    conf_.KW = KW();
    conf_.SD = KSD();
    conf_.SH = KSH();
    conf_.SW = KSW();
    conf_.padF = padFront();
    conf_.padT = padT();
    conf_.padL = padL();
    conf_.DD = KDD();
    conf_.DH = KDH();
    conf_.DW = KDW();
    return status::success;
}

status_t ref_pooling_bwd_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<pooling_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_pooling_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    for (auto &md : {pd()->diff_src_md(), pd()->diff_dst_md()}) {
        if (memory_desc_wrapper(md).size() == 0) return status::success;
    }

    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto nelems_A = memory_desc_wrapper(pd()->diff_src_md(0)).nelems();
        pooling_bwd_kernel_vec_t pooling_bwd_kernel(pd()->conf_, cgh, ctx);

        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        int work_per_wg = wg_size * block_size;
        int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;
        int n_thr = n_wgs * wg_size;
        cgh.parallel_for(
                ::sycl::nd_range<1>(n_thr, wg_size), pooling_bwd_kernel);
    });
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
