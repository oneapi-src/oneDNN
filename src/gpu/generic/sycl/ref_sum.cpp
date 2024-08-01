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

#include "gpu/generic/sycl/ref_sum.hpp"
#include "gpu/generic/sycl/sum_kernels.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

status_t ref_sum_t::pd_t::init_conf() {
    conf_ = sycl_sum_conf_t();
    conf_.n = n_inputs();

    for (auto i = 0; i < conf_.n; ++i) {
        conf_.src_md[i] = xpu::sycl::md_t(src_md(i));
        conf_.src_scales[i] = scales()[i];
    }
    conf_.dst_md = xpu::sycl::md_t(dst_md());

    // XXX: should probably be tuned.
    conf_.block_size = 16;
    conf_.wg_size = 32;
    conf_.wk_size = memory_desc_wrapper(dst_md()).nelems();
    return status::success;
}

status_t ref_sum_t::init(engine_t *engine) {
    const auto kid = ::sycl::get_kernel_id<sum_kernel_vec_t>();
    CHECK(create_kernel(engine, kid, &kernel_));

    return status::success;
}

status_t ref_sum_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    parallel_for(ctx, kernel_, [&](::sycl::handler &cgh) {
        auto src0_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 0);
        auto src1_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 1);
        auto src2_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 2);
        auto src3_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 3);
        auto src4_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 4);
        auto src5_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 5);
        auto src6_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 6);
        auto src7_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 7);
        auto src8_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 8);
        auto src9_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 9);
        auto src10_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 10);
        auto src11_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 11);
        auto src12_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 12);
        auto src13_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 13);
        auto src14_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 14);
        auto src15_mem_arg
                = CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_MULTIPLE_SRC + 15);

        auto dst_mem_arg = CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);

        sum_kernel_vec_t sum_kernel(pd()->conf_, src0_mem_arg, src1_mem_arg,
                src2_mem_arg, src3_mem_arg, src4_mem_arg, src5_mem_arg,
                src6_mem_arg, src7_mem_arg, src8_mem_arg, src9_mem_arg,
                src10_mem_arg, src11_mem_arg, src12_mem_arg, src13_mem_arg,
                src14_mem_arg, src15_mem_arg, dst_mem_arg);

        const int block_size = pd()->conf_.block_size;
        const int wg_size = pd()->conf_.wg_size;

        const int t_work = pd()->conf_.wk_size;
        const int wg_work = wg_size * block_size;
        const int wg_cnt = utils::div_up(t_work, wg_work);

        cgh.parallel_for(
                ::sycl::nd_range<1>(wg_cnt * wg_size, wg_size), sum_kernel);
    });

    return status::success;
}

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
