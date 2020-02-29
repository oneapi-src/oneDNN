/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef GPU_OCL_CROSS_ENGINE_REORDER_HPP
#define GPU_OCL_CROSS_ENGINE_REORDER_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_reorder_pd.hpp"
#include "gpu/ocl/ocl_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Cross-engine reorder manages all reorders between GPU and CPU engines.
//
// For CPU -> GPU reorder, it includes 2 steps:
// 1. CPU -> GPU copying
// 2. GPU reorder
//
// For GPU -> CPU reorder, it includes 2 steps:
// 1. GPU reorder
// 2. GPU -> CPU copying
struct cross_engine_reorder_t : public primitive_impl_t {
    struct pd_t : public reorder_pd_t {
        using reorder_pd_t::reorder_pd_t;

        pd_t(const pd_t &rhs)
            : reorder_pd_t(rhs)
            , reorder_(rhs.reorder_->clone())
            , reorder_engine_kind_(rhs.reorder_engine_kind_)
            , do_reorder_(rhs.do_reorder_) {}

        pd_t &operator=(const pd_t &rhs) {
            DNNL_SHORT_CIRCUIT_SELF_ASSIGN(rhs);
            reorder_pd_t::operator=(rhs);
            reorder_.reset(rhs.reorder_->clone());
            reorder_engine_kind_ = rhs.reorder_engine_kind_;
            do_reorder_ = rhs.do_reorder_;
            return *this;
        }

        DECLARE_COMMON_PD_T("ocl:cross_engine::any", cross_engine_reorder_t);

        DECLARE_GPU_REORDER_CREATE();

        status_t init();

        std::unique_ptr<primitive_desc_t> reorder_;
        engine_kind_t reorder_engine_kind_ = engine_kind::gpu;
        bool do_reorder_ = true;

    private:
        status_t init_scratchpad(
                memory_tracking::registrar_t &scratchpad) const;
    };

    virtual status_t init() override {
        status_t status;

        primitive_t *reorder_ptr;
        status = pd()->reorder_->create_primitive(&reorder_ptr);
        if (status != status::success) return status;

        reorder_.reset(reorder_ptr);

        return status::success;
    }

    cross_engine_reorder_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }

    std::unique_ptr<primitive_t> reorder_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
