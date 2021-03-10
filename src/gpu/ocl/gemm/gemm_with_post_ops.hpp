/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_gemm_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gemm_with_post_ops_t : public gpu_gemm_t {
    struct pd_t : public gpu_gemm_pd_t {

        DECLARE_COMMON_PD_T("ocl:gemm:with_po", gemm_with_post_ops_t);

        pd_t(const gemm_desc_t *adesc, const primitive_attr_t *attr,
                const hint_class *hint_fwd_pd)
            : gpu_gemm_pd_t(adesc, attr, hint_fwd_pd) {}

        pd_t(const pd_t &other)
            : gpu_gemm_pd_t(other)
            , gemm_pd_(other.gemm_pd_->clone())
            , post_op_worker_pd_(other.post_op_worker_pd_->clone())
            , use_scratchpad_with_post_op_worker(
                      other.use_scratchpad_with_post_op_worker) {}

        status_t init(engine_t *engine);

        void init_scratchpad();

        bool use_scratchpad() const {
            return use_scratchpad_with_post_op_worker;
        }

        std::unique_ptr<primitive_desc_t> gemm_pd_;
        std::unique_ptr<primitive_desc_t> post_op_worker_pd_;
        bool use_scratchpad_with_post_op_worker = false;
    };

    gemm_with_post_ops_t(const pd_t *apd) : gpu_gemm_t(apd) {}

    status_t init(engine_t *engine) override {
        auto ret_status = pd()->gemm_pd_->create_primitive(gemm_prim_, engine);
        CHECK(ret_status);
        ret_status = pd()->post_op_worker_pd_->create_primitive(
                post_op_worker_prim_, engine);
        return ret_status;
    }

    status_t execute(const gemm_exec_ctx_t &ctx) const override;

protected:
    primitive_list_t nested_primitives() const override {
        return {gemm_prim_.get(), post_op_worker_prim_.get()};
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> gemm_prim_;
    std::shared_ptr<primitive_t> post_op_worker_prim_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
