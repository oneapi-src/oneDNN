/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#include "gpu/generic/sycl/ref_binary.hpp"
#include "gpu/generic/sycl/binary_kernels.hpp"
#include "gpu/generic/sycl/sycl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_binary_t::pd_t::init_conf() {
    conf_ = sycl_binary_conf_t();

    conf_.src0_md = xpu::sycl::md_t(src_md(0));
    conf_.src1_md = xpu::sycl::md_t(src_md(1));
    conf_.dst_md = xpu::sycl::md_t(dst_md());
    conf_.ndims = ndims();

    conf_.wk_size = memory_desc_wrapper(dst_md()).nelems();

    conf_.alg_kind = desc()->alg_kind;
    // Limitations:
    // - Only common scale policy is supported.
    conf_.do_scale_src0 = !attr()->scales_.has_default_values(DNNL_ARG_SRC_0);
    conf_.do_scale_src1 = !attr()->scales_.has_default_values(DNNL_ARG_SRC_1);
    conf_.is_tensor_op = is_tensor_op();
    for (size_t i = 0; i < xpu::sycl::md_t::max_dims; i++) {
        conf_.broadcast_dims0[i]
                = conf_.src0_md.dims()[i] == 1 && conf_.src1_md.dims()[i] != 1;
        conf_.broadcast_dims1[i]
                = conf_.src0_md.dims()[i] != 1 && conf_.src1_md.dims()[i] == 1;
    }

    conf_.post_ops = sycl_post_ops_t(attr(), dst_md());

    return status::success;
}

status_t ref_binary_t::init(impl::engine_t *engine) {
    if (memory_desc_wrapper(pd()->dst_md()).size() == 0) return status::success;

    const auto kid = ::sycl::get_kernel_id<binary_kernel_vec_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_binary_t::execute(const exec_ctx_t &ctx) const {
    if (memory_desc_wrapper(pd()->dst_md()).size() == 0) return status::success;

    ctx.zero_pad_output(DNNL_ARG_TO);

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        binary_kernel_vec_t binary_kernel(pd()->conf_, cgh, ctx);

        cgh.parallel_for(get_range(ctx, pd()->conf_.wk_size), binary_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
