/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_OCL_GEN9_BATCH_NORMALIZATION_HPP
#define GPU_OCL_GEN9_BATCH_NORMALIZATION_HPP

#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_batch_normalization_pd.hpp"
#include "gpu/ocl/ocl_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_batch_normalization_fwd_t : public primitive_t {
    struct pd_t : public gpu_batch_normalization_fwd_pd_t {
        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : gpu_batch_normalization_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen9:blocked", gen9_batch_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            auto src_data_t = src_md()->data_type;
            auto dst_data_t = dst_md()->data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool ok = is_fwd()
                    && (utils::everyone_is(f16, src_data_t, dst_data_t)
                            || utils::everyone_is(bf16, src_data_t, dst_data_t)
                            || utils::everyone_is(f32, src_data_t, dst_data_t))
                    && attr()->has_default_values(attr_skip_mask)
                    && IMPLICATION(!attr()->has_default_values(),
                            attr()->post_ops_.len_ == 1 && with_relu_post_op())
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups);
            if (!ok) return status::unimplemented;

            if (is_training() && fuse_norm_relu()) init_default_ws(8);

            status_t status = init_conf(engine);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            return init_scratchpad(scratchpad);
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        status_t init_scratchpad(
                memory_tracking::registrar_t &scratchpad) const;

        bnorm_conf_t conf;
        offsets_t off;
    };

    status_t init(engine_t *engine) override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine);
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        std::vector<const char *> kernel_names
                = {"gen9_bnorm_fwd", nullptr, nullptr, nullptr, nullptr};
        if (pd()->conf.calculate_stats) {
            kernel_names[1] = "gen9_calc_mean";
            kernel_names[2] = "gen9_calc_variance";
            kernel_names[3] = "gen9_reduce_mean";
            kernel_names[4] = "gen9_reduce_variance";
        }

        std::vector<compute::binary_t> binaries;
        status = compute_engine->create_binaries(
                &binaries, kernel_names, kernel_ctx);
        CHECK(status);

        binary_ = binaries[0];
        calculate_mean_binary_ = binaries[1];
        calculate_variance_binary_ = binaries[2];
        reduce_mean_binary_ = binaries[3];
        reduce_variance_binary_ = binaries[4];

        return status::success;
    }

    gen9_batch_normalization_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t create_resource(
            engine_t *engine, resource_mapper_t &mapper) const override {
        if (mapper.has_resource(this)) return status::success;
        auto r = utils::make_unique<ocl_resource_t>();
        if (!r) return status::out_of_memory;
        CHECK(r->create_kernels_and_add(engine,
                {binary_, calculate_mean_binary_, calculate_variance_binary_,
                        reduce_mean_binary_, reduce_variance_binary_}));
        mapper.add(this, std::move(r));
        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::binary_t binary_;
    compute::binary_t calculate_mean_binary_;
    compute::binary_t reduce_mean_binary_;
    compute::binary_t calculate_variance_binary_;
    compute::binary_t reduce_variance_binary_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif // GPU_OCL_GEN9_BATCH_NORMALIZATION_HPP
