/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "gpu/sycl/ref_binary.hpp"
#include "gpu/sycl/binary_kernels.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace sycl {

using namespace impl::sycl;

status_t ref_binary_t::pd_t::init_conf() {
    conf_ = sycl_binary_conf_t();

    conf_.src0_md = sycl_md_t(src_md(0));
    conf_.src1_md = sycl_md_t(src_md(1));
    conf_.dst_md = sycl_md_t(dst_md());
    conf_.ndims = ndims();

    // XXX: should probably be tuned.
    conf_.block_size = 16;
    conf_.wg_size = 32;

    // TODO: uniform work groups are not supported for CUDA backend.
    // Need to find a way to circumvent it.
    if ((memory_desc_wrapper(src_md(0)).nelems() / conf_.block_size)
                    % conf_.wg_size
            != 0)
        return status::unimplemented;

    conf_.alg_kind = desc()->alg_kind;
    // Limitations:
    // - Runtime scales are not supported.
    // - Only common scale policy is supported.
    // XXX: oneDNN can stop supporting non-RT scales in 3.0 hence
    // this code can be thrown away in 3.0.
    conf_.src0_scale = attr()->scales_.get(DNNL_ARG_SRC_0).scales_[0];
    conf_.src1_scale = attr()->scales_.get(DNNL_ARG_SRC_1).scales_[0];
    conf_.do_scale_src0
            = !attr()->scales_.get(DNNL_ARG_SRC_0).has_default_values();
    conf_.do_scale_src1
            = !attr()->scales_.get(DNNL_ARG_SRC_1).has_default_values();
    conf_.is_tensor_op = is_tensor_op();
    for (size_t i = 0; i < sycl_md_t::max_dims; i++) {
        conf_.broadcast_dims[i] = broadcast_dims()[i];
    }

    conf_.post_ops = sycl_post_ops_t(attr());

    return status::success;
}

status_t ref_binary_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<binary_kernel_vec_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_binary_t::execute(const exec_ctx_t &ctx) const {
    const auto *src0 = CTX_IN_SYCL_STORAGE(DNNL_ARG_SRC_0);
    const auto *src1 = CTX_IN_SYCL_STORAGE(DNNL_ARG_SRC_1);
    auto *dst = CTX_OUT_SYCL_STORAGE(DNNL_ARG_DST);

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src0_mem_arg = src0->get_in_memory_arg(ctx.stream(), cgh);
        auto src1_mem_arg = src1->get_in_memory_arg(ctx.stream(), cgh);
        auto dst_mem_arg = dst->get_out_memory_arg(ctx.stream(), cgh);
        auto nelems_A = memory_desc_wrapper(pd()->src_md(0)).nelems();

        binary_kernel_vec_t binary_kernel(
                pd()->conf_, src0_mem_arg, src1_mem_arg, dst_mem_arg);
        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;
        cgh.parallel_for(::sycl::nd_range<1>(nelems_A / block_size, wg_size),
                binary_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace gpu
} // namespace impl
} // namespace dnnl
