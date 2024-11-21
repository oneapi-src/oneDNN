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

#ifndef GPU_INTEL_OCL_REF_PRELU_HPP
#define GPU_INTEL_OCL_REF_PRELU_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/reduction_pd.hpp"
#include "common/utils.hpp"
#include "gpu/gpu_prelu_pd.hpp"
#include "gpu/intel/gpu_primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

struct ref_prelu_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_prelu_fwd_pd_t {
        using gpu_prelu_fwd_pd_t::gpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T("prelu_ref:any", ref_prelu_fwd_t);

        status_t init(impl::engine_t *engine) {

            VDISPATCH_PRELU(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_PRELU(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_PRELU(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_PRELU(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_PRELU(
                    !memory_desc_ndims_ok(src_md(0), dst_md(0), weights_md(0)),
                    VERBOSE_INCONSISTENT_NDIMS, "src", "dst weights");
            VDISPATCH_PRELU(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");

            VDISPATCH_PRELU_SC(init_conf(engine), "init_conf()");
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        prelu_conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        status = create_kernel(engine, &kernel_, "ref_prelu_fwd", kernel_ctx);
        CHECK(status);

        return status::success;
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

struct ref_prelu_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_prelu_bwd_pd_t {
        using gpu_prelu_bwd_pd_t::gpu_prelu_bwd_pd_t;

        DECLARE_COMMON_PD_T("prelu_ref:any", ref_prelu_bwd_t);

        status_t init(impl::engine_t *engine) {

            VDISPATCH_PRELU(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_PRELU(
                    diff_dst_md()->data_type == diff_src_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");
            VDISPATCH_PRELU(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_PRELU(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_PRELU(!memory_desc_ndims_ok(diff_src_md(0),
                                    diff_dst_md(0), diff_weights_md(0)),
                    VERBOSE_INCONSISTENT_NDIMS, "diff_src",
                    "diff_dst diff_weights");
            VDISPATCH_PRELU(memory_desc_wrapper(diff_dst_md())
                            == memory_desc_wrapper(diff_src_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");

            VDISPATCH_PRELU_SC(init_conf(engine), "init_conf()");
            if (conf.reduce_diff_weights) {
                VDISPATCH_PRELU_SC(init_reduction(engine), "init_reduction()");
                init_scratchpad();
            }

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        void init_scratchpad();

        status_t init_reduction(impl::engine_t *engine) {
            reduction_desc_t rdesc;
            memory_desc_t red_diff_mem_desc(*src_md(0));
            red_diff_mem_desc.data_type = dnnl_f32;
            CHECK(reduction_desc_init(&rdesc,
                    dnnl_alg_kind_t::dnnl_reduction_sum, &red_diff_mem_desc,
                    diff_weights_md(0), 0, 0));
            primitive_attr_t reduction_attr(*attr());
            if (!reduction_attr.is_initialized()) return status::out_of_memory;
            primitive_desc_iterator_t it(
                    engine, (op_desc_t *)&rdesc, &reduction_attr, nullptr);
            if (!it.is_initialized()) return status::invalid_arguments;
            reduction_pd_ = *(++it);
            if (reduction_pd_)
                return status::success;
            else {
                return status::invalid_arguments;
            }
        }

        prelu_conf_t conf;
        std::shared_ptr<primitive_desc_t> reduction_pd_;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        status = create_kernel(engine, &kernel_, "ref_prelu_bwd", kernel_ctx);
        CHECK(status);

        if (pd()->conf.reduce_diff_weights) {
            CHECK(create_nested_primitive(
                    reduction_p_, pd()->reduction_pd_, engine));
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
    std::shared_ptr<impl::primitive_t> reduction_p_;
};

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
