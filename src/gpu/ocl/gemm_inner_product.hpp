/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef GPU_OCL_GEMM_INNER_PRODUCT_HPP
#define GPU_OCL_GEMM_INNER_PRODUCT_HPP

#include <assert.h>
#include <string>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_iterator.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gemm_inner_product_fwd_t : public gpu_primitive_t {
    struct pd_t : public gpu_inner_product_fwd_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_fwd_pd_t(adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : gpu_inner_product_fwd_pd_t(rhs) {
            gemm_pd_.reset(rhs.gemm_pd_->clone());
            attr_info_ = rhs.attr_info_;
        }
        ~pd_t() = default;

        DECLARE_COMMON_PD_T("ocl:gemm", gemm_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace data_type;

            assert(engine->kind() == engine_kind::gpu);

            attr_info_ = attr_info_t::create(attr());

            bool ok = is_fwd() && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && utils::one_of(true,
                            expect_data_types(f16, f16, f16, f16, f16),
                            expect_data_types(f32, f32, f32, f32, f32))
                    && attr()->post_ops_.len() <= 2
                    && IMPLICATION(attr()->post_ops_.len() == 2,
                            attr()->post_ops_.find(dnnl_sum) == 0)

                    && dense_consistency_check(src_md(), weights_md(), dst_md())
                    && dense_gemm_consistency_check(
                            src_md(), weights_md(), dst_md());
            if (!ok) return status::unimplemented;

            memory_desc_t a_md, b_md, c_md;
            init_2d_desc(&a_md, src_md());
            init_2d_desc(&b_md, weights_md(), true);
            init_2d_desc(&c_md, dst_md());
            bool gemm_ok = status::success
                    == create_gemm_pd(gemm_pd_, engine, &a_md, &b_md, &c_md,
                            weights_md(1), desc()->accum_data_type, attr(),
                            true);
            if (!gemm_ok) return status::unimplemented;
            init_scratchpad();

            return status::success;
        }

        attr_info_t attr_info_ = {};
        std::unique_ptr<primitive_desc_t> gemm_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    gemm_inner_product_fwd_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(gemm_, engine);
        if (gemm_status != status::success) return gemm_status;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {gemm_.get()};
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<primitive_t> gemm_;
};

struct gemm_inner_product_bwd_data_t : public gpu_primitive_t {
    struct pd_t : public gpu_inner_product_bwd_data_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : gpu_inner_product_bwd_data_pd_t(rhs) {
            gemm_pd_.reset(rhs.gemm_pd_->clone());
        }
        ~pd_t() = default;

        DECLARE_COMMON_PD_T("ocl:gemm", gemm_inner_product_bwd_data_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            assert(engine->kind() == engine_kind::gpu);

            bool ok = this->desc()->prop_kind == backward_data
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && utils::one_of(weights_md()->data_type, f32, bf16)
                    && utils::one_of(diff_src_md()->data_type, f32, bf16)
                    && utils::one_of(diff_dst_md()->data_type, f32, bf16)
                    && attr()->has_default_values()
                    && dense_consistency_check(
                            diff_src_md(), weights_md(), diff_dst_md())
                    && dense_gemm_consistency_check(
                            diff_src_md(), weights_md(), diff_dst_md());
            if (!ok) return status::unimplemented;

            memory_desc_t a_md, b_md, c_md;
            init_2d_desc(&a_md, diff_dst_md());
            init_2d_desc(&b_md, weights_md());
            init_2d_desc(&c_md, diff_src_md());

            bool gemm_ok = status::success
                    == create_gemm_pd(gemm_pd_, engine, &a_md, &b_md, &c_md,
                            &glob_zero_md, desc()->accum_data_type, attr(),
                            true);
            if (!gemm_ok) return status::unimplemented;
            init_scratchpad();

            return status::success;
        }

        std::unique_ptr<primitive_desc_t> gemm_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    gemm_inner_product_bwd_data_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(gemm_, engine);
        return gemm_status;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {gemm_.get()};
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<primitive_t> gemm_;
};

struct gemm_inner_product_bwd_weights_t : public gpu_primitive_t {
    using gpu_ip_bwd_weights_pd_t = gpu_inner_product_bwd_weights_pd_t;
    struct pd_t : public gpu_ip_bwd_weights_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_ip_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) : gpu_ip_bwd_weights_pd_t(rhs) {
            gemm_pd_.reset(rhs.gemm_pd_->clone());
        }
        ~pd_t() = default;

        DECLARE_COMMON_PD_T("gemm:ocl", gemm_inner_product_bwd_weights_t);

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            assert(engine->kind() == engine_kind::gpu);

            bool ok = this->desc()->prop_kind == backward_weights
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && utils::one_of(diff_weights_md()->data_type, f32, bf16)
                    && utils::one_of(src_md()->data_type, f32, bf16)
                    && utils::one_of(diff_dst_md()->data_type, f32, bf16)
                    && attr()->has_default_values()
                    && dense_consistency_check(
                            src_md(), diff_weights_md(), diff_dst_md())
                    && dense_gemm_consistency_check(
                            src_md(), diff_weights_md(), diff_dst_md());
            if (!ok) return status::unimplemented;

            memory_desc_t a_md, b_md, c_md;
            if (wei_tr()) {
                init_2d_desc(&a_md, src_md(), true);
                init_2d_desc(&b_md, diff_dst_md());
                init_2d_desc(&c_md, diff_weights_md(), true);
            } else {
                init_2d_desc(&a_md, diff_dst_md(), true);
                init_2d_desc(&b_md, src_md());
                init_2d_desc(&c_md, diff_weights_md());
            }

            bool gemm_ok = false;
            gemm_ok = status::success
                    == create_gemm_pd(gemm_pd_, engine, &a_md, &b_md, &c_md,
                            &glob_zero_md, desc()->accum_data_type, attr());

            if (!gemm_ok) return status::unimplemented;
            init_scratchpad();

            return status::success;
        }

        bool wei_tr() const {
            const auto &wmd = *this->diff_weights_md();
            return wmd.format_desc.blocking.strides[0] == 1;
        }

        std::unique_ptr<primitive_desc_t> gemm_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    gemm_inner_product_bwd_weights_t(const pd_t *apd) : gpu_primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        status_t gemm_status = pd()->gemm_pd_->create_primitive(gemm_, engine);
        if (gemm_status != status::success) return gemm_status;

        if (pd()->with_bias()) {
            compute::kernel_ctx_t kernel_ctx;

            kernel_ctx.set_data_type(pd()->src_md()->data_type);
            def_data_type(
                    kernel_ctx, pd()->diff_weights_md(1)->data_type, "BIA");
            kernel_ctx.define_int("MB", pd()->MB());
            kernel_ctx.define_int("OC", pd()->OC());

            create_kernel(engine, &bias_kernel_,
                    "gemm_inner_product_backward_weights_bias", kernel_ctx);
            if (!bias_kernel_) return status::runtime_error;
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

protected:
    primitive_list_t nested_primitives() const override {
        return {gemm_.get()};
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> gemm_;
    compute::kernel_t bias_kernel_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
