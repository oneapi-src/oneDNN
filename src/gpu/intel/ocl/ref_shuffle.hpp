/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_REF_SHUFFLE_HPP
#define GPU_INTEL_OCL_REF_SHUFFLE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_shuffle_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_shuffle_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_shuffle_pd_t {
        using gpu_shuffle_pd_t::gpu_shuffle_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_shuffle_t);

        status_t init(impl::engine_t *engine) {
            using namespace format_tag;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto &md_src = is_fwd() ? src_md() : diff_src_md();
            const auto &md_dst = is_fwd() ? dst_md() : diff_dst_md();
            const memory_desc_wrapper src_d(md_src);
            const memory_desc_wrapper dst_d(md_dst);

            VDISPATCH_SHUFFLE(src_d.data_type() == dst_d.data_type(),
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_SHUFFLE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SHUFFLE(!memory_desc_ndims_ok(src_d.md_, dst_d.md_),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_SHUFFLE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SHUFFLE(
                    src_d == dst_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_SHUFFLE(IMPLICATION(src_md()->data_type == data_type::f16,
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_SHUFFLE_SC(init_conf(engine), "init_conf()");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        shuffle_conf_t conf;
        offsets_t off;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        CHECK(create_kernel(engine, &kernel_, "ref_shuffle", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_<format_tag::any>(ctx);
    }

private:
    template <format_tag_t tag>
    status_t execute_(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
