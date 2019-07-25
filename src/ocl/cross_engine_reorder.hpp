/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef OCL_CROSS_ENGINE_REORDER_HPP
#define OCL_CROSS_ENGINE_REORDER_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/primitive_desc.hpp"
#include "common/utils.hpp"
#include "ocl/ocl_reorder_pd.hpp"
#include "ocl/ocl_utils.hpp"

namespace mkldnn {
namespace impl {
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
struct cross_engine_reorder_t : public primitive_t {
    struct pd_t : public reorder_pd_t {
        using reorder_pd_t::reorder_pd_t;

        pd_t(const pd_t &rhs)
            : reorder_pd_t(rhs)
            , gpu_reorder_pd_(rhs.gpu_reorder_pd_->clone()) {}

        DECLARE_COMMON_PD_T("ocl:cross_engine::any", cross_engine_reorder_t);

        DECLARE_OCL_REORDER_CREATE();

        status_t init() {
            bool args_ok = true
                    && utils::one_of(engine_kind::cpu, src_engine()->kind(),
                            dst_engine()->kind())
                    && utils::one_of(engine_kind::gpu, src_engine()->kind(),
                            dst_engine()->kind())
                    && (dst_engine()->kind() != src_engine()->kind());

            if (!args_ok)
                return status::unimplemented;

            auto *compute_engine = utils::downcast<compute::compute_engine_t *>(
                    dst_engine()->kind() == engine_kind::gpu ? dst_engine()
                                                             : src_engine());

            auto r_impls = engine_->get_reorder_implementation_list();
            const primitive_attr_t r_attr(*attr());
            for (auto r = r_impls; *r; ++r) {
                reorder_pd_t *r_pd = nullptr;
                if ((*r)(&r_pd, compute_engine, &r_attr, compute_engine,
                            src_md(), compute_engine, dst_md())
                        == status::success) {

                    r_pd->init_info();
                    gpu_reorder_pd_.reset(r_pd);
                    break;
                }
            }

            if (!gpu_reorder_pd_)
                return status::unimplemented;

            return status::success;
        }

        std::unique_ptr<primitive_desc_t> gpu_reorder_pd_;
    };

    virtual status_t init() override {
        status_t status;

        primitive_t *gpu_reorder_ptr;
        status = pd()->gpu_reorder_pd_->create_primitive(&gpu_reorder_ptr);
        if (status != status::success)
            return status;

        gpu_reorder_.reset(gpu_reorder_ptr);

        bool with_sum_ab = (pd()->alpha() != 1.0 || pd()->beta() != 0.0);
        do_reorder_ = with_sum_ab
                || memory_desc_wrapper(pd()->src_md())
                        != memory_desc_wrapper(pd()->dst_md());

        if (do_reorder_) {
            temp_buf.reset(new memory_t(engine(),
                    pd()->src_engine()->kind() == engine_kind::cpu
                            ? pd()->src_md()
                            : pd()->dst_md(),
                    memory_flags_t::alloc, nullptr));
            if (!temp_buf)
                return status::out_of_memory;
        }

        return status::success;
    }

    cross_engine_reorder_t(const pd_t *apd) : primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    std::unique_ptr<primitive_t> gpu_reorder_;
    std::unique_ptr<memory_t> temp_buf;
    bool do_reorder_;
};

} // namespace ocl
} // namespace impl
} // namespace mkldnn

#endif
