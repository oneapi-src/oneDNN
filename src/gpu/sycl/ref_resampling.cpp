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

#include "gpu/sycl/ref_resampling.hpp"
#include "gpu/sycl/resampling_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

status_t ref_resampling_fwd_t::pd_t::init_conf() {
    conf_ = sycl_resampling_conf_t();

    conf_.src_dt = src_md(0)->data_type;
    conf_.dst_dt = dst_md()->data_type;

    conf_.block_size = 16;
    conf_.wg_size = 32;

    conf_.MB = MB();
    conf_.C = C();
    conf_.ID = ID();
    conf_.IH = IH();
    conf_.IW = IW();
    conf_.OD = OD();
    conf_.OH = OH();
    conf_.OW = OW();

    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        conf_.dst_dims[i] = dst_md()->dims[i];
    }
    conf_.dst_ndims = dst_md()->ndims;
    auto nelems_A = memory_desc_wrapper(src_md(0)).nelems();
    conf_.work_amount = nelems_A;
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;

    conf_.src_md = sycl_md_t(src_md(0));
    conf_.dst_md = sycl_md_t(dst_md());

    conf_.alg = desc()->alg_kind;
    const auto *att = attr();
    const auto &attr_po = att->post_ops_;
    conf_.po_len = attr_po.len();

    for (auto i = 0; i < attr_po.len(); ++i) {
        if (attr_po.contain(primitive_kind::binary, i)) {
            dnnl::impl::memory_desc_t mem = attr_po.entry_[i].binary.src1_desc;
            conf_.src1_md[i] = sycl_md_t(&mem);
        }
    }
    conf_.post_ops = sycl_post_ops_t(attr());
    return status::success;
}

status_t ref_resampling_fwd_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<resampling_kernel_fwd_vec_t>();
    return create_kernel(engine, kid, &kernel_);
}

status_t ref_resampling_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto dst_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);

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
        resampling_kernel_fwd_vec_t resampling_fwd_kernel(pd()->conf_,
                src_mem_arg, dst_mem_arg, post_v0, post_v1, post_v2, post_v3,
                post_v4);

        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        int work_per_wg = wg_size * block_size;
        int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;
        int n_thr = n_wgs * wg_size;
        cgh.parallel_for(
                ::sycl::nd_range<1>(n_thr, wg_size), resampling_fwd_kernel);
    });

    return status::success;
}

status_t ref_resampling_bwd_t::pd_t::init_conf() {
    conf_ = sycl_resampling_conf_t();

    conf_.diff_src_md = sycl_md_t(diff_src_md(0));
    conf_.diff_dst_md = sycl_md_t(diff_dst_md());

    conf_.src_dt = src_md(0)->data_type;
    conf_.dst_dt = dst_md()->data_type;
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.dst_ndims = dst_md()->ndims;
    auto nelems_A = memory_desc_wrapper(diff_src_md(0)).nelems();
    conf_.work_amount = nelems_A;
    int work_per_wg = conf_.wg_size * conf_.block_size;
    int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;
    conf_.n_thr = n_wgs * conf_.wg_size;
    conf_.alg = desc()->alg_kind;

    conf_.MB = MB();
    conf_.C = C();
    conf_.ID = ID();
    conf_.IH = IH();
    conf_.IW = IW();
    conf_.OD = OD();
    conf_.OH = OH();
    conf_.OW = OW();

    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        conf_.dst_dims[i] = dst_md()->dims[i];
    }
    return status::success;
}

status_t ref_resampling_bwd_t::init(engine_t *engine) {
    if (pd()->conf_.alg == alg_kind::resampling_nearest) {
        const auto kid = ::sycl::get_kernel_id<resampling_kernel_bwd_vec_t>();
        CHECK(create_kernel(engine, kid, &kernel_));
    } else {
        const auto kid = ::sycl::get_kernel_id<resampling_kernel_bwd_vec1_t>();
        CHECK(create_kernel(engine, kid, &kernel_));
    }
    return status::success;
}

status_t ref_resampling_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    return parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto dst_mem_arg = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_DST);
        auto src_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DIFF_SRC);

        auto nelems_A = memory_desc_wrapper(pd()->diff_src_md(0)).nelems();

        resampling_kernel_bwd_vec_t resampling_bwd_kernel(
                pd()->conf_, dst_mem_arg, src_mem_arg);

        resampling_kernel_bwd_vec1_t resampling_bwd_kernel1(
                pd()->conf_, dst_mem_arg, src_mem_arg);
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        int work_per_wg = wg_size * block_size;
        int n_wgs = (nelems_A + work_per_wg - 1) / work_per_wg;
        int n_thr = n_wgs * wg_size;
        if (pd()->conf_.alg == alg_kind::resampling_nearest) {
            cgh.parallel_for(
                    ::sycl::nd_range<1>(n_thr, wg_size), resampling_bwd_kernel);
        } else {
            cgh.parallel_for(::sycl::nd_range<1>(n_thr, wg_size),
                    resampling_bwd_kernel1);
        }
    });
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
