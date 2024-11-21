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

#ifndef GPU_INTEL_OCL_BNORM_SIMPLE_BNORM_HPP
#define GPU_INTEL_OCL_BNORM_SIMPLE_BNORM_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_batch_normalization_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct simple_batch_normalization_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_batch_normalization_fwd_pd_t {
        using gpu_batch_normalization_fwd_pd_t::
                gpu_batch_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:simple", simple_batch_normalization_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            VDISPATCH_BNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_BNORM(
                    utils::one_of(src_md()->data_type, f32, bf16, f16, s8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(IMPLICATION(f16 == src_md()->data_type,
                                    compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_BNORM(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_BNORM(IMPLICATION(src_md()->data_type == s8,
                                    !is_training() && stats_is_src()),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(check_scale_shift_data_type(),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale, shift or datatype configuration");
            VDISPATCH_BNORM(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    IMPLICATION(!attr()->has_default_values(),
                            attr()->post_ops_.len() == 1
                                    && with_relu_post_op(is_training())),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BNORM(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_BNORM(compute_engine->mayiuse(
                                    compute::device_ext_t::intel_subgroups),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");

            if (is_training() && (fuse_norm_relu() || fuse_norm_add_relu())) {
                VDISPATCH_BNORM_SC(init_default_ws(8), VERBOSE_WS_INIT);
            }

            VDISPATCH_BNORM_SC(init_conf(engine), "init_conf()");
            init_scratchpad();

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        void init_scratchpad();

        bnorm_conf_t conf;
        offsets_t off;
        compute::dispatch_t dispatch_calc_stat;
        compute::dispatch_t dispatch_reduce_stat;
        compute::dispatch_t dispatch;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;
        compute::kernel_ctx_t kernel_ctx;

        CHECK(pd()->init_kernel_ctx(kernel_ctx));

        std::vector<const char *> kernel_names
                = {"simple_bnorm_fwd", "simple_calculate_mean_variance"};

        std::vector<compute::kernel_t> kernels;
        CHECK(create_kernels(engine, &kernels, kernel_names, kernel_ctx));

        kernel_ = kernels[0];
        calculate_mean_variance_kernel_ = kernels[1];

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
    compute::kernel_t calculate_mean_variance_kernel_;
};

struct simple_batch_normalization_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_batch_normalization_bwd_pd_t {
        using gpu_batch_normalization_bwd_pd_t::
                gpu_batch_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:simple", simple_batch_normalization_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            VDISPATCH_BNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_BNORM(utils::one_of(src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BNORM(IMPLICATION(f16 == src_md()->data_type,
                                    compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_BNORM(src_md()->data_type == diff_src_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "diff_src");
            VDISPATCH_BNORM(
                    diff_src_md()->data_type == diff_dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "diff_src", "diff_dst");
            VDISPATCH_BNORM(check_scale_shift_data_type(),
                    VERBOSE_UNSUPPORTED_FEATURE,
                    "unsupported scale, shift or datatype configuration");
            VDISPATCH_BNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_BNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BNORM(memory_desc_wrapper(diff_src_md())
                            == memory_desc_wrapper(diff_dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");

            if (fuse_norm_relu() || fuse_norm_add_relu()) {
                VDISPATCH_BNORM_SC(init_default_ws(8), VERBOSE_WS_INIT);
                VDISPATCH_BNORM(compare_ws(hint_fwd_pd_), "compare_ws()");
            }

            VDISPATCH_BNORM_SC(init_conf(engine), "init_conf()");
            init_scratchpad();

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        void init_scratchpad();

        bnorm_conf_t conf;
        offsets_t off;
        compute::dispatch_t dispatch_calc_stat;
        compute::dispatch_t dispatch_reduce_stat;
        compute::dispatch_t dispatch;
    };

    status_t init(impl::engine_t *engine) override {
        if (pd()->has_zero_dim_memory()) return status::success;
        compute::kernel_ctx_t kernel_ctx;

        CHECK(pd()->init_kernel_ctx(kernel_ctx));

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
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
