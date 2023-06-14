/*******************************************************************************
* Copyright 2021-2023 Intel Corporation
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

#ifndef GPU_OCL_GEN9_GLOBAL_POOLING_HPP
#define GPU_OCL_GEN9_GLOBAL_POOLING_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/reduction_pd.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_pooling_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gen9_global_pooling_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_pooling_fwd_pd_t {
        pd_t(const pooling_desc_t *adesc, const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : gpu_pooling_fwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen9_global:any", gen9_global_pooling_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            auto src_data_t = src_md()->data_type;
            auto dst_data_t = dst_md()->data_type;

            bool ok = set_default_params() == status::success
                    && utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference)
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && !utils::one_of(f64, src_data_t, dst_data_t)
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training)
                init_default_ws(s32);

            CHECK(init_conf(engine));
            CHECK(init_reduction(engine));
            init_scratchpad();
            return status::success;
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        void init_scratchpad();

        status_t init_reduction(engine_t *engine) {
            using namespace alg_kind;

            reduction_desc_t rdesc;
            memory_desc_t red_src_mem_desc(*src_md(0));
            red_src_mem_desc.data_type = src_md()->data_type;
            CHECK(reduction_desc_init(&rdesc,
                    desc()->alg_kind == pooling_max
                            ? dnnl_alg_kind_t::dnnl_reduction_max
                            : dnnl_alg_kind_t::dnnl_reduction_mean,
                    &red_src_mem_desc, dst_md(0), 0, 0));
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

        std::shared_ptr<primitive_desc_t> reduction_pd_;
        pool_conf_t conf;
        offsets_t off;
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        using namespace alg_kind;
        // TODO: max-pooling requires workspace to track indices for training config
        if (pd()->desc()->alg_kind != pooling_max
                && !utils::one_of(data_type::f64, pd()->dst_md()->data_type,
                        pd()->src_md()->data_type)) {
            if (create_nested_primitive(
                        reduction_p_, pd()->reduction_pd_, engine)
                    == status::success) {
                return status::success;
            }
        }
        // fallback
        CHECK(create_kernel(
                engine, &kernel_, "gen9_global_pooling_fwd", kernel_ctx));
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
    std::shared_ptr<primitive_t> reduction_p_;
};

struct gen9_global_pooling_bwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_pooling_bwd_pd_t {
        pd_t(const pooling_desc_t *adesc, const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : gpu_pooling_bwd_pd_t(adesc, attr, hint_fwd_pd) {}

        DECLARE_COMMON_PD_T("ocl:gen9_global:any", gen9_global_pooling_bwd_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace alg_kind;
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine);

            auto diff_dst_dt = diff_dst_md()->data_type;
            auto diff_src_dt = diff_src_md()->data_type;

            bool ok = set_default_params() == status::success
                    && utils::one_of(desc()->prop_kind, backward_data)
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && (utils::everyone_is(data_type::f32,
                                diff_dst_md()->data_type,
                                diff_src_md()->data_type)
                            || utils::everyone_is(data_type::f16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type)
                            || utils::everyone_is(data_type::bf16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type))
                    && !utils::one_of(data_type::f64, diff_src_dt, diff_dst_dt)
                    && IMPLICATION(diff_src_md()->data_type == data_type::f16,
                            compute_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short))
                    && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == pooling_max) {
                init_default_ws(data_type::s32);
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            return init_conf(engine);
        }

        status_t init_conf(engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        pool_conf_t conf;
        offsets_t off;
    };

    status_t init(engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;
        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        CHECK(create_kernel(
                engine, &kernel_, "gen9_global_pooling_bwd", kernel_ctx));
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
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
