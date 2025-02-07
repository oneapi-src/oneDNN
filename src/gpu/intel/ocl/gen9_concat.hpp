/*******************************************************************************
* Copyright 2021-2025 Intel Corporation
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

#ifndef GPU_INTEL_OCL_GEN9_CONCAT_HPP
#define GPU_INTEL_OCL_GEN9_CONCAT_HPP

#include "common/engine.hpp"
#include "common/primitive.hpp"
#include "common/reorder.hpp"
#include "common/reorder_pd.hpp"
#include "common/stream.hpp"
#include "gpu/gpu_concat_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/ocl/utils.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct gen9_concat_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_concat_pd_t {
        pd_t(const primitive_attr_t *attr, const memory_desc_t *dst_md, int n,
                int concat_dim, const memory_desc_t *const *src_mds)
            : gpu_concat_pd_t(attr, dst_md, n, concat_dim, src_mds) {}

        pd_t(const pd_t &rhs) = default;
        ~pd_t() override = default;

        DECLARE_CONCAT_PD_T("gen9:any", gen9_concat_t);

        status_t init(impl::engine_t *engine) {

            using sm = primitive_attr_t::skip_mask_t;

            VDISPATCH_CONCAT(n_inputs() <= 16, VERBOSE_BAD_PARAM, "n_inputs");
            VDISPATCH_CONCAT(attr()->has_default_values(sm::scales_runtime),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONCAT_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONCAT(!memory_desc_ndims_ok(dst_md()), VERBOSE_BAD_NDIMS,
                    "dst", dst_md()->ndims);

            VDISPATCH_CONCAT_SC(init_conf(engine),
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "concat");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        concat_conf_t conf;

    protected:
        bool can_use_sub_group_size(
                const compute::compute_engine_t *compute_engine,
                int sub_group_size);
        int calculate_sub_group_size(
                const compute::compute_engine_t *compute_engine);
        std::pair<int, int> calculate_iter_dim_idx_chunk(int num_threads) const;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        status = create_kernel(engine, &kernel, "gen9_concat", kernel_ctx);
        CHECK(status);

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_concat(ctx);
    }

private:
    status_t execute_concat(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif //GPU_INTEL_OCL_GEN9_CONCAT_HPP
