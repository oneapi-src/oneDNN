/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef CPU_X64_JIT_UNI_INSTANCE_NORMALIZATION_HPP
#define CPU_X64_JIT_UNI_INSTANCE_NORMALIZATION_HPP

#include "common/primitive.hpp"

#include "cpu/cpu_group_normalization_pd.hpp"

#include "cpu/x64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_uni_instance_normalization_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public cpu_group_normalization_fwd_pd_t {
        using cpu_group_normalization_fwd_pd_t::
                cpu_group_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                "jit_instance:uni", jit_uni_instance_normalization_fwd_t);

        status_t init(engine_t *engine);

        int nthr_; // To not exceed the limit in execute used for set up.
    };

    status_t init(engine_t *engine) override {
        CHECK(safe_ptr_assign(kernel_, kernel_base_t::create(pd())));
        CHECK(safe_ptr_assign(kernel_mean_, kernel_stat_base_t::create(pd())));
        CHECK(safe_ptr_assign(
                kernel_var_, kernel_stat_base_t::create(pd(), true)));
        if (kernel_) CHECK(kernel_->create_kernel());
        if (kernel_mean_) CHECK(kernel_mean_->create_kernel());
        if (kernel_var_) CHECK(kernel_var_->create_kernel());
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

    struct kernel_base_t {
        virtual void operator()(const void *src, void *dst, const float *scale,
                const float *shift, const float *mean, const float *var,
                const float *src_scales, const float *dst_scales,
                const void *post_ops_binary_rhs_arg_vec,
                const size_t block_size) const = 0;
        static kernel_base_t *create(const group_normalization_pd_t *pd);
        virtual status_t create_kernel() = 0;
        virtual ~kernel_base_t() = default;

    protected:
        kernel_base_t(const group_normalization_pd_t *pd) : pd_(pd) {}

        // `pd_` is needed to access its members (such as `attr()`) in
        // `generate()` call.
        const group_normalization_pd_t *pd_;
    };

    struct kernel_stat_base_t {
        virtual void operator()(
                const void *src, float *mean, size_t block_size) const = 0;
        virtual void operator()(const void *src, const float *mean, float *var,
                size_t block_size) const = 0;
        static kernel_stat_base_t *create(
                const group_normalization_pd_t *pd, bool compute_var = false);
        virtual status_t create_kernel() = 0;
        virtual ~kernel_stat_base_t() = default;
    };

protected:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<kernel_base_t> kernel_;
    std::unique_ptr<kernel_stat_base_t> kernel_mean_;
    std::unique_ptr<kernel_stat_base_t> kernel_var_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
