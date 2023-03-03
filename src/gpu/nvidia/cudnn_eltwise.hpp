/*******************************************************************************
* Copyright 2020-2023 Intel Corporation
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

#ifndef GPU_NVIDIA_SYCL_CUDA_ELTWISE_HPP
#define GPU_NVIDIA_SYCL_CUDA_ELTWISE_HPP

#include "common/eltwise_pd.hpp"
#include "common/primitive.hpp"
#include "gpu/nvidia/cudnn_eltwise_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_eltwise_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public eltwise_fwd_pd_t {
        using eltwise_fwd_pd_t::eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_eltwise_fwd_t);

        status_t init(engine_t *engine) {
            using namespace alg_kind;

            auto sycl_dev
                    = utils::downcast<impl::sycl::sycl_engine_base_t *>(engine)
                              ->device();

            bool ok = is_fwd()
                    // Supported algorithms
                    && utils::one_of(desc()->alg_kind, eltwise_relu,
                            eltwise_tanh, eltwise_elu, eltwise_logistic)
                    // Supported data types
                    && utils::one_of(src_md()->data_type, data_type::f32,
                            data_type::bf16, data_type::f16, data_type::s8)
                    && src_md()->data_type == dst_md()->data_type
                    && IMPLICATION(src_md()->data_type == data_type::bf16,
                            has_bf16_support(sycl_dev))
                    && IMPLICATION(desc()->alg_kind == eltwise_relu,
                            desc()->alpha == 0)
                    && attr()->has_default_values()
                    && set_default_formats_common()
                    && memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md())
                    // Eltwise does not support blocking
                    && src_md()->format_desc.blocking.inner_nblks == 0;
            if (!ok) return status::unimplemented;

            eltwise_fwd_impl_.reset(new cudnn_eltwise_fwd_impl_t());
            return eltwise_fwd_impl_->init(this);
        }
        std::shared_ptr<cudnn_eltwise_fwd_impl_t> eltwise_fwd_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct cudnn_eltwise_bwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public eltwise_bwd_pd_t {
        using eltwise_bwd_pd_t::eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_eltwise_bwd_t);

        status_t init(engine_t *) {
            using namespace alg_kind;
            bool ok = !is_fwd()
                    // Supported algorithms
                    && utils::one_of(desc()->alg_kind, eltwise_relu)
                    // Supported data types
                    && (utils::everyone_is(data_type::f32, data_md()->data_type,
                                diff_src_md()->data_type,
                                diff_dst_md()->data_type)
                            || utils::everyone_is(data_type::bf16,
                                    data_md()->data_type,
                                    diff_src_md()->data_type,
                                    diff_dst_md()->data_type))
                    && IMPLICATION(desc()->alg_kind == eltwise_relu,
                            desc()->alpha == 0)
                    && attr()->has_default_values()
                    && set_default_formats_common()
                    && memory_desc_wrapper(diff_src_md())
                            == memory_desc_wrapper(diff_dst_md())
                    // Eltwise does not support blocking
                    && src_md()->format_desc.blocking.inner_nblks == 0
                    && diff_dst_md()->format_desc.blocking.inner_nblks == 0;
            if (!ok) return status::unimplemented;

            eltwise_bwd_impl_.reset(new cudnn_eltwise_bwd_impl_t());
            return eltwise_bwd_impl_->init(this);
        }
        std::shared_ptr<cudnn_eltwise_bwd_impl_t> eltwise_bwd_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
