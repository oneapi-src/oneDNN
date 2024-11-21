/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_INTEL_OCL_GEN9_POOLING_HPP
#define GPU_INTEL_OCL_GEN9_POOLING_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_pooling_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct gen9_pooling_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_pooling_fwd_pd_t {
        using gpu_pooling_fwd_pd_t::gpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9", gen9_pooling_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);
            auto src_data_t = src_md()->data_type;
            auto dst_data_t = dst_md()->data_type;
            auto acc_data_t = desc()->accum_data_type;

            VDISPATCH_POOLING_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(utils::one_of(desc()->prop_kind, forward_training,
                                      forward_inference),
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(utils::one_of(desc()->alg_kind, pooling_max,
                                      pooling_avg_include_padding,
                                      pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(
                    (utils::everyone_is(f32, src_data_t, dst_data_t, acc_data_t)
                            || utils::everyone_is(f16, src_data_t, dst_data_t)
                            || utils::everyone_is(bf16, src_data_t, dst_data_t)
                            || utils::everyone_is(u8, src_data_t, dst_data_t)
                            || utils::everyone_is(s8, src_data_t, dst_data_t)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(
                    IMPLICATION(utils::one_of(src_data_t, f16, s8, u8),
                            desc()->prop_kind == forward_inference),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_POOLING(
                    post_ops_with_binary_ok(attr(), dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_POOLING_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");
            VDISPATCH_POOLING(!utils::one_of(f64, src_data_t, dst_data_t),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_POOLING(compute_engine->mayiuse(
                                      compute::device_ext_t::intel_subgroups),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");
            VDISPATCH_POOLING(
                    IMPLICATION(src_data_t == f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short)),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            bool is_training = desc()->prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training)
                init_default_ws(s32);

            CHECK(init_conf(engine));

            // Required for storing spatial offsets into workspace for
            // pooling_max training.
            VDISPATCH_POOLING(conf.kd * conf.kh * conf.kw <= INT_MAX,
                    VERBOSE_OFFSET_DT_MISMATCH, "kernel spatial", "int");

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        pool_conf_t conf;
        offsets_t off;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(engine, &kernel_, "gen9_pooling_fwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct gen9_pooling_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_pooling_bwd_pd_t {
        using gpu_pooling_bwd_pd_t::gpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:gen9:any", gen9_pooling_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace alg_kind;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            VDISPATCH_POOLING_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(utils::one_of(desc()->prop_kind, backward_data),
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(utils::one_of(desc()->alg_kind, pooling_max,
                                      pooling_avg_include_padding,
                                      pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(
                    (utils::everyone_is(data_type::f32,
                             diff_dst_md()->data_type, diff_src_md()->data_type)
                            || utils::everyone_is(data_type::f16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type)
                            || utils::everyone_is(data_type::bf16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(
                    !utils::one_of(data_type::f64, diff_src_md()->data_type,
                            diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_POOLING(!is_dilated(), VERBOSE_UNSUPPORTED_FEATURE,
                    "does not support dilations");
            VDISPATCH_POOLING(
                    IMPLICATION(diff_src_md()->data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_POOLING(compute_engine->mayiuse(
                                      compute::device_ext_t::intel_subgroups),
                    VERBOSE_UNSUPPORTED_DEVICE_FEATURE, "subgroups");

            if (desc()->alg_kind == pooling_max) {
                init_default_ws(data_type::s32);
                VDISPATCH_POOLING(
                        compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);
            }

            CHECK(init_conf(engine));

            // Required for storing spatial offsets into workspace for
            // pooling_max training due to use of int type.
            VDISPATCH_POOLING(conf.kd * conf.kh * conf.kw <= INT_MAX,
                    VERBOSE_OFFSET_DT_MISMATCH, "kernel spatial", "int");

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        pool_conf_t conf;
        offsets_t off;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(engine, &kernel_, "gen9_pooling_bwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
