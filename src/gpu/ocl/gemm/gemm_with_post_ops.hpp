/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GPU_OCL_GEMM_GEMM_WITH_POST_OPS_HPP
#define GPU_OCL_GEMM_GEMM_WITH_POST_OPS_HPP

#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gemm_with_post_ops_t : public gpu_gemm_t {
    using gpu_gemm_t::gpu_gemm_t;
    struct pd_t : public gpu_gemm_pd_t {

        DECLARE_COMMON_PD_T("ocl:gemm_with_po:any", gemm_with_post_ops_t);

        pd_t(const gemm_desc_t *adesc, const primitive_attr_t *attr,
                const hint_class *hint_fwd_pd)
            : gpu_gemm_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other) = default;

        status_t init(engine_t *engine);

        void init_scratchpad();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        bool use_scratchpad() const {
            return use_scratchpad_with_post_op_worker;
        }

        std::shared_ptr<primitive_desc_t> gemm_pd_;
        bool use_scratchpad_with_post_op_worker = false;
        bool use_reorder = false;
        compute::dispatch_t dispatch_;
        attr_info_t attr_info_;
    };

    status_t init(engine_t *engine) override {
        auto ret_status
                = create_nested_primitive(gemm_prim_, pd()->gemm_pd_, engine);
        CHECK(ret_status);
        primitive_attr_t attr;
        int threads_per_eu = 0;
        if (status::success
                == pd()->gemm_pd_->query(query::preferred_gpu_threads_per_eu, 0,
                        &threads_per_eu)) {
            CHECK(attr.set_gpu_attr(gpu_primitive_attr_t(threads_per_eu)));
        }
        compute::kernel_ctx_t kernel_ctx(&attr);
        ret_status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(ret_status);
        ret_status = create_kernel(
                engine, &post_process_kernel_, "gemm_post_ops", kernel_ctx);
        return ret_status;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> gemm_prim_;
    compute::kernel_t post_process_kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
