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

#ifndef GPU_INTEL_OCL_REF_POOLING_HPP
#define GPU_INTEL_OCL_REF_POOLING_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_pooling_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_pooling_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_pooling_fwd_pd_t {
        using gpu_pooling_fwd_pd_t::gpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref", ref_pooling_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            auto src_data_t = src_md()->data_type;
            auto dst_data_t = dst_md()->data_type;
            auto acc_data_t = desc()->accum_data_type;

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            const auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            VDISPATCH_POOLING_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_POOLING(utils::one_of(desc()->prop_kind, forward_training,
                                      forward_inference),
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(utils::one_of(desc()->alg_kind, pooling_max,
                                      pooling_avg_include_padding,
                                      pooling_avg_exclude_padding),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_POOLING(
                    IMPLICATION(utils::one_of(src_data_t, s8, u8, s32),
                            desc()->prop_kind == forward_inference),
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(IMPLICATION(src_data_t != dst_data_t,
                                      desc()->prop_kind == forward_inference),
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_POOLING(
                    IMPLICATION(src_data_t == bf16, src_data_t == dst_data_t),
                    VERBOSE_INCONSISTENT_DT, "src_data_t", "dst_data_t");
            VDISPATCH_POOLING(
                    IMPLICATION(utils::one_of(src_data_t, s8, u8),
                            utils::one_of(dst_data_t, s8, u8, f16, f32)),
                    VERBOSE_INCONSISTENT_DT, "src_data_t", "dst_data_t");
            VDISPATCH_POOLING(IMPLICATION(src_data_t == f16,
                                      utils::one_of(dst_data_t, s8, u8, f16)),
                    VERBOSE_INCONSISTENT_DT, "src_data_t", "dst_data_t");
            VDISPATCH_POOLING(IMPLICATION(src_data_t == f32,
                                      utils::one_of(dst_data_t, s8, u8, f32)),
                    VERBOSE_INCONSISTENT_DT, "src_data_t", "dst_data_t");
            VDISPATCH_POOLING(
                    IMPLICATION(utils::one_of(f32, src_data_t, dst_data_t),
                            acc_data_t == f32),
                    VERBOSE_INCONSISTENT_DT, "src_data_t", "dst_data_t");
            VDISPATCH_POOLING(IMPLICATION(utils::one_of(src_data_t, s8, u8)
                                              && dst_data_t != f32,
                                      acc_data_t == s32),
                    VERBOSE_INCONSISTENT_DT, "src_data_t", "dst_data_t");
            VDISPATCH_POOLING(
                    IMPLICATION(utils::one_of(f16, src_data_t, dst_data_t),
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_POOLING(
                    IMPLICATION(utils::one_of(f64, src_data_t, dst_data_t),
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_POOLING(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_POOLING(
                    post_ops_with_binary_ok(attr(), dst_md()->data_type, 5),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_POOLING_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_TAG);

            VDISPATCH_POOLING_SC(init_conf(engine), "init_conf()");

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training) {
                // Required for storing spatial offsets into workspace for
                // pooling_max training due to use of int type.
                VDISPATCH_POOLING(conf.kd * conf.kh * conf.kw <= INT_MAX,
                        VERBOSE_OFFSET_DT_MISMATCH, "kernel spatial", "int");
                init_default_ws(s32);
            }

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

        CHECK(create_kernel(engine, &kernel_, "ref_pooling_fwd", kernel_ctx));
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

struct ref_pooling_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_pooling_bwd_pd_t {
        using gpu_pooling_bwd_pd_t::gpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_pooling_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace alg_kind;

            const auto *compute_engine
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
                            || utils::everyone_is(data_type::bf16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type)
                            || (utils::everyone_is(data_type::f16,
                                        diff_dst_md()->data_type,
                                        diff_src_md()->data_type)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16))
                            || (utils::everyone_is(data_type::f64,
                                        diff_dst_md()->data_type,
                                        diff_src_md()->data_type)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp64))),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_POOLING(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_POOLING_SC(init_conf(engine), "init_conf()");

            if (desc()->alg_kind == pooling_max) {
                // Required for storing spatial offsets into workspace for
                // pooling_max training due to use of int type.
                VDISPATCH_POOLING(conf.kd * conf.kh * conf.kw <= INT_MAX,
                        VERBOSE_OFFSET_DT_MISMATCH, "kernel spatial", "int");
                init_default_ws(data_type::s32);
                VDISPATCH_POOLING(
                        compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);
            }

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

        CHECK(create_kernel(engine, &kernel_, "ref_pooling_bwd", kernel_ctx));
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

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
