/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#ifndef GPU_GEN9_REDUCTION_HPP
#define GPU_GEN9_REDUCTION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_reduction_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

// Requirements for this implementation:
// - only N and C can be blocked (NCx or nCx)
// - C is blocked by 16 or 32
// - src/dst blocking structures have to match
// - either all or none of the HWD dims are being reduced
// - padded C is a multiple of 16

// This implementation combines any HWD dimensions to a single index,
// Leaving N, C, and HWD dimensions (plus blocked portions of C and
// possibly N). Each of these dimensions is broken into chunks, and a
// work item is assigned one of each of these chunks in the initial step.

// The final phase finishes the reduction by reducing all of the chunks
// belonging to reduction dimensions.

struct gen9_reduction_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_reduction_pd_t {
        using gpu_reduction_pd_t::gpu_reduction_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9", gen9_reduction_t);

        status_t init(engine_t *engine) {
            using smask_t = primitive_attr_t::skip_mask_t;
            bool ok = set_default_params() == status::success
                    && attr_.has_default_values(
                            smask_t::post_ops | smask_t::gpu_attr)
                    && !memory_desc_ndims_ok(src_md(), dst_md())
                    && post_ops_with_binary_ok(attr(), dst_md()->data_type, 5)
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;

            CHECK(init_conf(engine));
            init_scratchpad();

            return status::success;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        void init_scratchpad();

        reduction_conf_t conf;
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx(pd()->attr());

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        status = create_kernel(
                engine, &initial_kernel, "gen9_initial_reduce", kernel_ctx);
        CHECK(status);
        if (!pd()->conf.skip_final_phase) {
            status = create_kernel(
                    engine, &final_kernel, "gen9_final_reduce", kernel_ctx);
            CHECK(status);
        }

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_gen9(ctx);
    }
    static bool is_compatible(const memory_desc_wrapper &src_mdw,
            const memory_desc_wrapper &dst_mdw, reduction_conf_t *conf);

private:
    status_t execute_gen9(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t initial_kernel;
    compute::kernel_t final_kernel;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
