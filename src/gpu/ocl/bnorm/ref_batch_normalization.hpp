/*******************************************************************************
* Copyright 2019-2023 Intel Corporation
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

#ifndef GPU_OCL_REF_BATCH_NORMALIZATION_HPP
#define GPU_OCL_REF_BATCH_NORMALIZATION_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_batch_normalization_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct ref_batch_normalization_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_batch_normalization_fwd_pd_t {
        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : gpu_batch_normalization_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_batch_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            bool ok = is_fwd()
                    && utils::one_of(src_md()->data_type, f32, bf16, f16, s8)
                    && IMPLICATION(f16 == src_md()->data_type,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && src_md()->data_type == dst_md()->data_type
                    && IMPLICATION(src_md()->data_type == s8,
                            !is_training() && stats_is_src())
                    && check_scale_shift_data_type()
                    && attr()->has_default_values(attr_skip_mask)
                    && IMPLICATION(!attr()->has_default_values(),
                            attr()->post_ops_.len() == 1
                                    && with_relu_post_op(is_training()))
                    && set_default_formats_common()
                    && memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md())
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups);
            if (!ok) return status::unimplemented;

            if (is_training() && (fuse_norm_relu() || fuse_norm_add_relu()))
                CHECK(init_default_ws(8));

            init_conf(engine);
            init_scratchpad();

            return status::success;
        }

        void init_conf(engine_t *engine);
        void init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        void init_scratchpad();

        bnorm_conf_t conf;
        offsets_t off;
        compute::dispatch_t dispatch_calc_stat;
        compute::dispatch_t dispatch_reduce_stat;
        compute::dispatch_t dispatch;
    };

    status_t init(engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;
        compute::kernel_ctx_t kernel_ctx;

        pd()->init_kernel_ctx(kernel_ctx);

        std::vector<const char *> kernel_names
                = {"ref_bnorm_fwd", nullptr, nullptr, nullptr, nullptr};
        if (pd()->conf.calculate_stats) {
            kernel_names[1] = "calculate_mean";
            kernel_names[2] = "calculate_variance";
            kernel_names[3] = "reduce_mean";
            kernel_names[4] = "reduce_variance";
        }

        std::vector<compute::kernel_t> kernels;
        CHECK(create_kernels(engine, &kernels, kernel_names, kernel_ctx));

        kernel_ = kernels[0];
        calculate_mean_kernel_ = kernels[1];
        calculate_variance_kernel_ = kernels[2];
        reduce_mean_kernel_ = kernels[3];
        reduce_variance_kernel_ = kernels[4];

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
    compute::kernel_t calculate_mean_kernel_;
    compute::kernel_t reduce_mean_kernel_;
    compute::kernel_t calculate_variance_kernel_;
    compute::kernel_t reduce_variance_kernel_;
    compute::kernel_t calculate_mean_variance_kernel_;
};

struct ref_batch_normalization_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_batch_normalization_bwd_pd_t {
        pd_t(const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : gpu_batch_normalization_bwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_batch_normalization_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            bool ok = !is_fwd()
                    && utils::one_of(src_md()->data_type, f32, bf16, f16)
                    && IMPLICATION(f16 == src_md()->data_type,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16))
                    && src_md()->data_type == diff_src_md()->data_type
                    && diff_src_md()->data_type == diff_dst_md()->data_type
                    && check_scale_shift_data_type()
                    && attr()->has_default_values()
                    && set_default_formats_common()
                    && memory_desc_wrapper(diff_src_md())
                            == memory_desc_wrapper(diff_dst_md());
            if (!ok) return status::unimplemented;

            if (fuse_norm_relu() || fuse_norm_add_relu()) {
                CHECK(init_default_ws(8));
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            init_conf(engine);
            init_scratchpad();

            return status::success;
        }

        void init_conf(engine_t *engine);
        void init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        void init_scratchpad();

        bnorm_conf_t conf;
        offsets_t off;
        compute::dispatch_t dispatch_calc_stat;
        compute::dispatch_t dispatch_reduce_stat;
        compute::dispatch_t dispatch;
    };

    status_t init(engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;
        compute::kernel_ctx_t kernel_ctx;

        pd()->init_kernel_ctx(kernel_ctx);

        std::vector<const char *> kernel_names
                = {"ref_bnorm_bwd", "calculate_stats", "reduce_stats"};

        std::vector<compute::kernel_t> kernels;
        CHECK(create_kernels(engine, &kernels, kernel_names, kernel_ctx));

        kernel_ = kernels[0];
        calculate_stats_kernel_ = kernels[1];
        reduce_stats_kernel_ = kernels[2];

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
    compute::kernel_t calculate_stats_kernel_;
    compute::kernel_t reduce_stats_kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
