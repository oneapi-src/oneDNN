/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_MATMUL_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_HPP

#include "gpu/gpu_matmul_pd.hpp"

#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/nvidia/cudnn_matmul_executor.hpp"
#include "gpu/nvidia/cudnn_matmul_impl.hpp"
#include "gpu/nvidia/cudnn_matmul_lt_impl.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_t : public gpu::primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public gpu_matmul_pd_t {
        using gpu_matmul_pd_t::gpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_matmul_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            data_type_t src_dt = src_md()->data_type;
            data_type_t dst_dt = dst_md()->data_type;
            data_type_t wei_dt = weights_md(0)->data_type;
            data_type_t bia_dt
                    = with_bias() ? weights_md(1)->data_type : data_type::f32;

            bool f32_case = utils::everyone_is(f32, src_dt, wei_dt, dst_dt);
            bool f16_case = utils::everyone_is(f16, src_dt, wei_dt, dst_dt);
            bool bf16_case = utils::everyone_is(bf16, src_dt, wei_dt, dst_dt);
            bool s8_case = utils::everyone_is(s8, src_dt, wei_dt)
                    && utils::one_of(dst_dt, s8, f32);

            auto *sycl_engine_impl
                    = utils::downcast<const xpu::sycl::engine_impl_t *>(
                            engine->impl());

            bool ok = is_dense_format_kind() && blocking_ok()
                    && attr()->has_default_values(
                            smask_t::scales_runtime | smask_t::post_ops)
                    && scales_ok() && attr_post_ops_ok(attr())
                    && IMPLICATION(bf16_case,
                            has_bf16_support(sycl_engine_impl->device()))
                    && set_default_formats()
                    && (f32_case || f16_case || bf16_case || s8_case)
                    && IMPLICATION(with_bias(),
                            (IMPLICATION(f32_case, utils::one_of(bia_dt, f32))
                                    && IMPLICATION(f16_case,
                                            utils::one_of(bia_dt, f16, f32))
                                    && IMPLICATION(bf16_case,
                                            utils::one_of(bia_dt, bf16, f32))
                                    && IMPLICATION(s8_case,
                                            utils::one_of(bia_dt, s8, f32))))
                    && !(with_bias() && s8_case);
            if (!ok) return status::unimplemented;

            if (src_md()->ndims > 3) return status::unimplemented;

            params_ = std::make_shared<cublas_params>();
            CHECK(params_->init(src_md(), weights_md(), dst_md(), weights_md(1),
                    attr(), batched(), with_bias()));

            if (!params_->has_runtime_params_) {
                auto scratchpad = scratchpad_registry().registrar();
                params_->init_scratchpad(dst_md(), scratchpad);
            }
            return status::success;
        }

        bool scales_ok() const {
            using namespace data_type;
            const auto &scales = attr()->scales_;
            const auto &supported_args
                    = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
            if (!scales.has_default_values(supported_args)) return false;
            // cuDNN does not support scaling per dimension.
            for (auto arg : supported_args) {
                auto &s = scales.get(arg);
                if (scales.get(arg).mask_ != 0
                        || !utils::one_of(
                                s.data_type_, s8, s32, f32, f16, bf16))
                    return false;
            }
            return true;
        }

        bool blocking_ok() const {
            std::vector<const memory_desc_t *> mds
                    = {src_md(), dst_md(), weights_md(0)};
            if (with_bias()) mds.push_back(weights_md(1));
            for (const memory_desc_t *md : mds) {
                memory_desc_wrapper mdw(md);
                if (mdw.is_blocking_desc()) {
                    if (mdw.blocking_desc().inner_nblks != 0) { return false; }
                }
            }
            return true;
        }

        std::shared_ptr<cublas_params> params_;
    };

    status_t init(impl::engine_t *engine) override {
        matmul_impl_.reset(new cudnn_matmul_impl_t());

        bool has_runtime_args = pd()->params_->has_runtime_params_;

        if (has_runtime_args) {
            executor_.reset(new cudnn_matmul_runtime_args_exec_t);
        } else {
            executor_.reset(new cudnn_matmul_exec_t);
            matmul_impl_->set_non_runtime_params(pd()->params_);
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

    std::shared_ptr<cudnn_matmul_impl_t> matmul_impl_;
    std::shared_ptr<cudnn_matmul_base_exec_t> executor_;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
