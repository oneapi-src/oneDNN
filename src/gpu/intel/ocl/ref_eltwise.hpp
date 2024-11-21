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

#ifndef GPU_INTEL_OCL_REF_ELTWISE_HPP
#define GPU_INTEL_OCL_REF_ELTWISE_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

// Elementwise
struct ref_eltwise_conf_t {
    int ndims;
    int vector_size;
    bool with_zero_padding;
    data_type_t data_type;
    alg_kind_t alg;
    bool is_forward;
    int work_group_size;
    int sub_group_size;
    compute::dispatch_t dispatch;
    memory_desc_info_t data_md_info;
    memory_desc_info_t data_diff_md_info;

    attr_info_t attr_info;
};

struct ref_eltwise_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_eltwise_fwd_pd_t {
        using gpu_eltwise_fwd_pd_t::gpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_eltwise_fwd_t);

        status_t init(impl::engine_t *engine) {
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            const auto attr_skip_mask = primitive_attr_t::skip_mask_t::post_ops;

            using namespace alg_kind;
            VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");

            VDISPATCH_ELTWISE(!memory_desc_ndims_ok(dst_md()),
                    VERBOSE_BAD_NDIMS, "dst_md", dst_md()->ndims);
            VDISPATCH_ELTWISE(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(post_ops_with_binary_ok(
                                      attr(), dst_md()->data_type, MAX_NDIMS),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_ELTWISE_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_ELTWISE(IMPLICATION(src_md()->data_type == data_type::f64,
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(IMPLICATION(src_md()->data_type == data_type::f16,
                                      compute_engine->mayiuse(
                                              compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_ELTWISE_SC(init_conf(engine), "init_conf()");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        ref_eltwise_conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        CHECK(create_kernel(engine, &kernel_, "ref_eltwise_fwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward_dense(ctx);
    }

private:
    status_t execute_forward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_eltwise_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_eltwise_bwd_pd_t {
        using gpu_eltwise_bwd_pd_t::gpu_eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_eltwise_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            assert(engine->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            using namespace alg_kind;
            VDISPATCH_ELTWISE(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(!memory_desc_ndims_ok(data_md(), diff_dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "data_md", "diff_dst_md");
            VDISPATCH_ELTWISE(
                    utils::one_of(data_md()->data_type, data_type::f32,
                            data_type::f16, data_type::bf16, data_type::f64),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    utils::everyone_is(data_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(
                    IMPLICATION(data_md()->data_type == data_type::f64,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    IMPLICATION(data_md()->data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(memory_desc_wrapper(diff_dst_md())
                            == memory_desc_wrapper(diff_src_md()),
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");

            VDISPATCH_ELTWISE_SC(init_conf(engine), "init_conf()");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        ref_eltwise_conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        CHECK(create_kernel(engine, &kernel_, "ref_eltwise_bwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_dense(ctx);
    }

private:
    status_t execute_backward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
