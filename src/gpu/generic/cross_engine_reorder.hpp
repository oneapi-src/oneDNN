/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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

#ifndef GPU_GENERIC_CROSS_ENGINE_REORDER_HPP
#define GPU_GENERIC_CROSS_ENGINE_REORDER_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {

// Cross-engine reorder manages all reorders between GPU and CPU engines.
//
// For CPU -> GPU reorder, it includes 2 steps:
// 1. CPU -> GPU copying
// 2. GPU reorder
//
// For GPU -> CPU reorder, it includes 2 steps:
// 1. GPU reorder
// 2. GPU -> CPU copying
struct cross_engine_reorder_t : public gpu::primitive_t {
    using gpu::primitive_t::primitive_t;
    struct pd_t : public gpu_reorder_pd_t {
        using gpu_reorder_pd_t::gpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("ocl:cross_engine::any", cross_engine_reorder_t);

        status_t init(impl::engine_t *engine, impl::engine_t *src_engine,
                impl::engine_t *dst_engine);

        std::shared_ptr<primitive_desc_t> reorder_pd_;
        engine_kind_t reorder_engine_kind_ = engine_kind::gpu;
        bool do_reorder_ = true;

    private:
        void init_scratchpad(impl::engine_t *engine);
        DECLARE_GPU_REORDER_CREATE();
    };

    status_t init(impl::engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> reorder_;
    std::shared_ptr<impl::primitive_t> zp_precomp_conv_;
};

} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
