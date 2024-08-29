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

#include "gpu/generic/sycl/ref_binary.hpp"
#include "gpu/generic/sycl/binary_kernels.hpp"

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

    // XXX: should probably be tuned.
    conf_.block_size = 16;
    conf_.wg_size = 32;

    conf_.wk_size = memory_desc_wrapper(dst_md()).nelems();

    conf_.alg_kind = desc()->alg_kind;
    // Limitations:
    // - Only common scale policy is supported.
    conf_.do_scale_src0
            = !attr()->scales_.get(DNNL_ARG_SRC_0).has_default_values();
    conf_.do_scale_src1
            = !attr()->scales_.get(DNNL_ARG_SRC_1).has_default_values();
    conf_.is_tensor_op = is_tensor_op();
    for (size_t i = 0; i < xpu::sycl::md_t::max_dims; i++) {
        conf_.broadcast_dims0[i]
                = conf_.src0_md.dims()[i] == 1 && conf_.src1_md.dims()[i] != 1;
        conf_.broadcast_dims1[i]
                = conf_.src0_md.dims()[i] != 1 && conf_.src1_md.dims()[i] == 1;
    }

    conf_.post_ops = sycl_post_ops_t(attr(), dst_md()->data_type);

    return status::success;
}

status_t ref_binary_t::init(impl::engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<binary_kernel_vec_t>();
    CHECK(create_kernel(engine, kid, &kernel_));
    return status::success;
}

status_t ref_binary_t::execute(const exec_ctx_t &ctx) const {

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        binary_kernel_vec_t binary_kernel(pd()->conf_, cgh, ctx);

        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        const int t_work = pd()->conf_.wk_size;
        const int wg_work = wg_size * block_size;
        const int wg_cnt = utils::div_up(t_work, wg_work);

        cgh.parallel_for(
                ::sycl::nd_range<1>(wg_cnt * wg_size, wg_size), binary_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
