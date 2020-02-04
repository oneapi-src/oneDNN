/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GPU_OCL_REF_POOLING_HPP
#define GPU_OCL_REF_POOLING_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gpu_pooling_pd.hpp"
#include "gpu/ocl/ocl_stream.hpp"
#include "gpu/ocl/ocl_utils.hpp"
#include "gpu/ocl/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

status_t ref_pooling_init_conf(
        pool_conf_t &conf, const pooling_pd_t *_pd, offsets_t &off);
status_t ref_pooling_init_const_def(compute::kernel_ctx_t &kernel_ctx,
        const pool_conf_t &conf, const offsets_t &off);

struct ref_pooling_fwd_t : public primitive_impl_t {
    struct pd_t : public gpu_pooling_fwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : gpu_pooling_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conf_()
            , off_() {}

        DECLARE_COMMON_PD_T("ocl:ref", ref_pooling_fwd_t);

        status_t init() {
            using namespace data_type;
            using namespace prop_kind;
            using namespace alg_kind;
            assert(engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());
            auto src_data_t = src_md()->data_type;
            auto dst_data_t = dst_md()->data_type;
            auto acc_data_t = desc()->accum_data_type;

            bool ok = set_default_params() == status::success
                    && utils::one_of(desc()->prop_kind, forward_training,
                            forward_inference)
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && (utils::everyone_is(
                                f32, src_data_t, dst_data_t, acc_data_t)
                            || utils::everyone_is(
                                    f16, src_data_t, dst_data_t, acc_data_t)
                            || utils::everyone_is(bf16, src_data_t, dst_data_t)
                            || utils::everyone_is(u8, src_data_t, dst_data_t)
                            || utils::everyone_is(s8, src_data_t, dst_data_t))
                    && IMPLICATION(utils::one_of(src_data_t, f16),
                            desc()->prop_kind == forward_inference)
                    && IMPLICATION(src_data_t == u8 || src_data_t == s8,
                            desc()->accum_data_type == s32)
                    && attr()->has_default_values()
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups)
                    && IMPLICATION(src_data_t == f16,
                            true
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)
                                    && compute_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short));
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == forward_training;
            if (desc()->alg_kind == pooling_max && is_training)
                init_default_ws(s32);

            return ref_pooling_init_conf(conf_, this, off_);
        }
        pool_conf_t conf_;
        offsets_t off_;
    };

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());

        compute::kernel_ctx_t kernel_ctx;
        ref_pooling_init_const_def(kernel_ctx, pd()->conf_, pd()->off_);

        compute_engine->create_kernel(&kernel_, "ref_pooling_fwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    ref_pooling_fwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

struct ref_pooling_bwd_t : public primitive_impl_t {
    struct pd_t : public gpu_pooling_bwd_pd_t {
        pd_t(engine_t *engine, const pooling_desc_t *adesc,
                const primitive_attr_t *attr,
                const pooling_fwd_pd_t *hint_fwd_pd)
            : gpu_pooling_bwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conf_()
            , off_() {}

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_pooling_bwd_t);

        status_t init() {
            using namespace prop_kind;
            using namespace alg_kind;
            assert(engine()->kind() == engine_kind::gpu);
            auto *compute_engine
                    = utils::downcast<compute::compute_engine_t *>(engine());

            bool ok = set_default_params() == status::success
                    && utils::one_of(desc()->prop_kind, backward_data)
                    && utils::one_of(desc()->alg_kind, pooling_max,
                            pooling_avg_include_padding,
                            pooling_avg_exclude_padding)
                    && (utils::everyone_is(data_type::f32,
                                diff_dst_md()->data_type,
                                diff_src_md()->data_type)
                            || utils::everyone_is(data_type::bf16,
                                    diff_dst_md()->data_type,
                                    diff_src_md()->data_type))
                    && attr()->has_default_values()
                    && compute_engine->mayiuse(
                            compute::device_ext_t::intel_subgroups);
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == pooling_max) {
                init_default_ws(data_type::s32);
                if (!compare_ws(hint_fwd_pd_)) return status::unimplemented;
            }

            return ref_pooling_init_conf(conf_, this, off_);
        }
        pool_conf_t conf_;
        offsets_t off_;
    };

    status_t init() override {
        auto *compute_engine
                = utils::downcast<compute::compute_engine_t *>(engine());

        compute::kernel_ctx_t kernel_ctx;
        ref_pooling_init_const_def(kernel_ctx, pd()->conf_, pd()->off_);

        compute_engine->create_kernel(&kernel_, "ref_pooling_bwd", kernel_ctx);
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    ref_pooling_bwd_t(const pd_t *apd) : primitive_impl_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_impl_t::pd(); }
    compute::kernel_t kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
