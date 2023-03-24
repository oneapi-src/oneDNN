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

#include "gpu/sycl/ref_pooling.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "gpu/sycl/pooling_kernels.hpp"
#include "gpu/sycl/sycl_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;

status_t ref_pooling_fwd_t::pd_t::init_conf() {
    using namespace primitive_kind;
    conf_ = sycl_pooling_conf_t();
    conf_.ndims = ndims();
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.src_md = sycl_md_t(src_md(0));
    conf_.dst_md = sycl_md_t(dst_md(0));
    conf_.ws_md = !types::is_zero_md(workspace_md())
            ? sycl_md_t(workspace_md(0))
            : sycl_md_t {};
    conf_.zero_dims = has_zero_dim_memory();
    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        conf_.dst_dims[i] = dst_md()->dims[i];
    }
    conf_.dst_ndims = dst_md()->ndims;
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

    const auto *att = attr();
    const auto &attr_po = att->post_ops_;
    conf_.po_len = attr_po.len();
    for (auto i = 0; i < attr_po.len(); ++i) {
        if (attr_po.contain(binary, i)) {
            dnnl::impl::memory_desc_t mem = attr_po.entry_[i].binary.src1_desc;
            conf_.src1_md[i] = sycl_md_t(&mem);
        }
    }
    conf_.post_ops = sycl_post_ops_t(attr());
    return status::success;
}

status_t ref_pooling_fwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<pooling_fwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_pooling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto dst = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);
        auto ws = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_WORKSPACE);
        auto post_v0 = CTX_IN_SYCL_KERNEL_MEMORY(
                DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1);
        auto post_v1 = CTX_IN_SYCL_KERNEL_MEMORY(
                DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1);
        auto post_v2 = CTX_IN_SYCL_KERNEL_MEMORY(
                DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1);
        auto post_v3 = CTX_IN_SYCL_KERNEL_MEMORY(
                DNNL_ARG_ATTR_MULTIPLE_POST_OP(3) | DNNL_ARG_SRC_1);
        auto post_v4 = CTX_IN_SYCL_KERNEL_MEMORY(
                DNNL_ARG_ATTR_MULTIPLE_POST_OP(4) | DNNL_ARG_SRC_1);
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();
        pooling_fwd_kernel_vec_t pooling_fwd_kernel(pd()->conf_, src, dst, ws,
                post_v0, post_v1, post_v2, post_v3, post_v4);

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
    conf_ = sycl_pooling_conf_t();
    conf_.ndims = ndims();
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.diff_src_md = sycl_md_t(diff_src_md(0));
    conf_.diff_dst_md = sycl_md_t(diff_dst_md(0));
    conf_.ws_md = !types::is_zero_md(workspace_md())
            ? sycl_md_t(workspace_md(0))
            : sycl_md_t {};
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

status_t ref_pooling_bwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<pooling_bwd_kernel_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_pooling_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto diff_dst = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);
        auto diff_src = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);
        auto ws = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_WORKSPACE);
        auto nelems_A = memory_desc_wrapper(pd()->diff_src_md(0)).nelems();
        pooling_bwd_kernel_vec_t pooling_bwd_kernel(
                pd()->conf_, diff_dst, diff_src, ws);

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
} // namespace gpu
} // namespace impl
} // namespace dnnl
