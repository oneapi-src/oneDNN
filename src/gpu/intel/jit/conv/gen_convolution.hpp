/*******************************************************************************
* Copyright 2021-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_CONV_GEN_CONVOLUTION_HPP
#define GPU_INTEL_JIT_CONV_GEN_CONVOLUTION_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "gpu/gpu_convolution_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class gen_convolution_t;
struct conv_pd_data_t;

class gen_convolution_fwd_t : public gpu_primitive_t {
public:
    friend gen_convolution_t;

    struct pd_t : public gpu_convolution_fwd_pd_t {
        friend gen_convolution_t;

        using gpu_convolution_fwd_pd_t::gpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("jit:ir", gen_convolution_fwd_t);

        status_t init(impl::engine_t *engine);

        std::shared_ptr<conv_pd_data_t> data;
    };

    using gpu_primitive_t::gpu_primitive_t;

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t init_res_storage(
            impl::engine_t *engine, gpu_resource_t *r) const override;

    std::shared_ptr<gen_convolution_t> impl_;
};

class gen_convolution_bwd_data_t : public gpu_primitive_t {
public:
    friend gen_convolution_t;

    struct pd_t : public gpu_convolution_bwd_data_pd_t {
        friend gen_convolution_t;

        using gpu_convolution_bwd_data_pd_t::gpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("jit:ir", gen_convolution_bwd_data_t);

        status_t init(impl::engine_t *engine);

        std::shared_ptr<conv_pd_data_t> data;
    };

    using gpu_primitive_t::gpu_primitive_t;

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t init_res_storage(
            impl::engine_t *engine, gpu_resource_t *r) const override;

    std::shared_ptr<gen_convolution_t> impl_;
};

class gen_convolution_bwd_weights_t : public gpu_primitive_t {
public:
    friend gen_convolution_t;

    struct pd_t : public gpu_convolution_bwd_weights_pd_t {
        friend gen_convolution_t;

        using gpu_convolution_bwd_weights_pd_t::
                gpu_convolution_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T("jit:ir", gen_convolution_bwd_weights_t);

        status_t init(impl::engine_t *engine);

        std::shared_ptr<conv_pd_data_t> data;
    };

    using gpu_primitive_t::gpu_primitive_t;

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    status_t init_res_storage(
            impl::engine_t *engine, gpu_resource_t *r) const override;

    std::shared_ptr<gen_convolution_t> impl_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
