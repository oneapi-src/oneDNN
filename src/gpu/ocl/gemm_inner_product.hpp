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

#ifndef GPU_OCL_GEMM_INNER_PRODUCT_HPP
#define GPU_OCL_GEMM_INNER_PRODUCT_HPP

#include <assert.h>
#include <string>

#include "common/c_types_map.hpp"
#include "common/gemm_utils.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "gpu/compute/compute.hpp"
#include "gpu/gemm/gpu_gemm.hpp"
#include "gpu/gpu_inner_product_pd.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/gpu_primitive_attr.hpp"
#include "gpu/gpu_reduction_pd.hpp"
#include "gpu/gpu_resource.hpp"
#include "gpu/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace ocl {

struct gemm_inner_product_fwd_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_inner_product_fwd_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_fwd_pd_t(adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) = default;
        ~pd_t() = default;

        DECLARE_COMMON_PD_T(gemm_pd_->name(), gemm_inner_product_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using namespace prop_kind;
            using namespace data_type;
            assert(engine->kind() == engine_kind::gpu);

            const auto attr_skip_mask
                    = primitive_attr_t::skip_mask_t::scales_runtime
                    | primitive_attr_t::skip_mask_t::post_ops;

            bool ok = is_fwd() && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && dense_consistency_check(src_md(), weights_md(), dst_md())
                    && dense_gemm_consistency_check(
                            src_md(), weights_md(), dst_md())
                    && attr()->has_default_values(attr_skip_mask)
                    && post_ops_with_binary_ok(
                            attr(), desc()->dst_desc.data_type)
                    && attr_.set_default_formats(dst_md(0)) == status::success;
            if (!ok) return status::unimplemented;

            attr_info_ = attr_info_t::create(attr());

            memory_desc_t a_md, b_md, c_md;
            CHECK(init_2d_desc(&a_md, src_md()));
            CHECK(init_2d_desc(&b_md, weights_md(), true));
            CHECK(init_2d_desc(&c_md, dst_md()));
            primitive_attr_t gemm_attr = *attr();
            auto wei_mask = gemm_attr.scales_.get(DNNL_ARG_WEIGHTS).mask_;
            if (wei_mask == 1) //transpose mask for gemm
                CHECK(gemm_attr.scales_.set(
                        DNNL_ARG_WEIGHTS, 1 << (b_md.ndims - 1)));
            else if (wei_mask != 0)
                return status::unimplemented;
            bool gemm_ok = status::success
                    == create_gemm_pd(gemm_pd_, engine, &a_md, &b_md, &c_md,
                            weights_md(1), desc()->accum_data_type, &gemm_attr,
                            true);
            if (!gemm_ok) return status::unimplemented;

            init_scratchpad();

            return status::success;
        }

        attr_info_t attr_info_ = {};
        std::shared_ptr<primitive_desc_t> gemm_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    status_t init(engine_t *engine) override {
        return create_nested_primitive(gemm_, pd()->gemm_pd_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<primitive_t> gemm_;
};

struct gemm_inner_product_bwd_data_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    struct pd_t : public gpu_inner_product_bwd_data_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_inner_product_bwd_data_pd_t(adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) = default;
        ~pd_t() = default;

        DECLARE_COMMON_PD_T(gemm_pd_->name(), gemm_inner_product_bwd_data_t);

        bool has_type(data_type_t v) const {
            return utils::one_of(v, weights_md()->data_type,
                    diff_src_md()->data_type, diff_dst_md()->data_type);
        }

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            assert(engine->kind() == engine_kind::gpu);

            bool ok = this->desc()->prop_kind == backward_data
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && (!(has_type(f16) && has_type(bf16)))
                    && utils::one_of(diff_src_md()->data_type, f32, bf16, f16)
                    && utils::one_of(diff_dst_md()->data_type, f32, bf16, f16)
                    && attr()->has_default_values()
                    && dense_consistency_check(
                            diff_src_md(), weights_md(), diff_dst_md())
                    && dense_gemm_consistency_check(
                            diff_src_md(), weights_md(), diff_dst_md());
            if (!ok) return status::unimplemented;

            memory_desc_t a_md, b_md, c_md;
            CHECK(init_2d_desc(&a_md, diff_dst_md()));
            CHECK(init_2d_desc(&b_md, weights_md()));
            CHECK(init_2d_desc(&c_md, diff_src_md()));

            bool gemm_ok = status::success
                    == create_gemm_pd(gemm_pd_, engine, &a_md, &b_md, &c_md,
                            &glob_zero_md, desc()->accum_data_type, attr(),
                            true);
            if (!gemm_ok) return status::unimplemented;
            init_scratchpad();

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> gemm_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    gemm_pd_->scratchpad_registry());
        }
    };

    status_t init(engine_t *engine) override {
        return create_nested_primitive(gemm_, pd()->gemm_pd_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::shared_ptr<primitive_t> gemm_;
};

struct gemm_inner_product_bwd_weights_t : public gpu_primitive_t {
    using gpu_primitive_t::gpu_primitive_t;
    using gpu_ip_bwd_weights_pd_t = gpu_inner_product_bwd_weights_pd_t;
    struct pd_t : public gpu_ip_bwd_weights_pd_t {
        pd_t(const inner_product_desc_t *adesc, const primitive_attr_t *attr,
                const inner_product_fwd_pd_t *hint_fwd_pd)
            : gpu_ip_bwd_weights_pd_t(adesc, attr, hint_fwd_pd) {}
        pd_t(const pd_t &rhs) = default;

        ~pd_t() = default;

        DECLARE_COMMON_PD_T(gemm_pd_->name(), gemm_inner_product_bwd_weights_t);

        bool has_type(data_type_t v) const {
            return utils::one_of(v, diff_weights_md()->data_type,
                    src_md()->data_type, diff_dst_md()->data_type);
        }

        status_t init(engine_t *engine) {
            using namespace prop_kind;
            using namespace data_type;

            assert(engine->kind() == engine_kind::gpu);
            bool ok = this->desc()->prop_kind == backward_weights
                    && set_default_params() == status::success
                    && !has_zero_dim_memory()
                    && (!(has_type(f16) && has_type(bf16)))
                    && utils::one_of(
                            diff_weights_md()->data_type, f32, bf16, f16)
                    && utils::one_of(src_md()->data_type, f32, bf16, f16)
                    && utils::one_of(diff_dst_md()->data_type, f32, bf16, f16)
                    && attr()->has_default_values()
                    && dense_consistency_check(
                            src_md(), diff_weights_md(), diff_dst_md())
                    && dense_gemm_consistency_check(
                            src_md(), diff_weights_md(), diff_dst_md());
            if (!ok) return status::unimplemented;

            memory_desc_t a_md, b_md, c_md;
            if (wei_tr()) {
                CHECK(init_2d_desc(&a_md, src_md(), true));
                CHECK(init_2d_desc(&b_md, diff_dst_md()));
                CHECK(init_2d_desc(&c_md, diff_weights_md(), true));
            } else {
                CHECK(init_2d_desc(&a_md, diff_dst_md(), true));
                CHECK(init_2d_desc(&b_md, src_md()));
                CHECK(init_2d_desc(&c_md, diff_weights_md()));
            }
            bool gemm_ok = false;
            auto reduce_bias = sum_ab::sum_none;
            if (with_bias())
                reduce_bias = wei_tr() ? sum_ab::sum_b_col : sum_ab::sum_a_row;
            gemm_ok = status::success
                    == create_gemm_pd(gemm_pd_, engine, &a_md, &b_md, &c_md,
                            &glob_zero_md, desc()->accum_data_type, attr(),
                            true, reduce_bias,
                            desc()->diff_bias_desc.data_type);

            //fused bias reduction not supported, apply in separate kernel
            if (with_bias() && !gemm_ok) {
                gemm_ok = status::success
                        == create_gemm_pd(gemm_pd_, engine, &a_md, &b_md, &c_md,
                                &glob_zero_md, desc()->accum_data_type, attr());
                if (!gemm_ok) return status::unimplemented;
                memory_desc_t reduction_dst_md, reduction_bias_md;
                //Set ndims to 3 in order to explicitly specify blocked format
                //so that it will go to optimized reduction implementation.
                reduction_bias_md.ndims = 3;
                reduction_bias_md.dims[0] = 1;
                reduction_bias_md.dims[1] = diff_bias_md_.dims[0];
                reduction_bias_md.dims[2] = 1;
                bool use_blocked = OC() % 16 == 0;
                CHECK(memory_desc_init_by_tag(reduction_bias_md,
                        reduction_bias_md.ndims, reduction_bias_md.dims,
                        diff_bias_md_.data_type,
                        use_blocked ? format_tag::aBc16b : format_tag::abc));
                reduction_dst_md = *diff_dst_md();
                reduction_dst_md.ndims = 3;
                reduction_dst_md.dims[2] = 1;
                CHECK(memory_desc_init_by_tag(reduction_dst_md,
                        reduction_dst_md.ndims, reduction_dst_md.dims,
                        diff_dst_md_.data_type,
                        use_blocked ? format_tag::aBc16b : format_tag::abc));
                reduction_desc_t reduction_d;
                CHECK(reduction_desc_init(&reduction_d,
                        dnnl::impl::alg_kind::reduction_sum, &reduction_dst_md,
                        &reduction_bias_md, 0.0f, 0.0f));
                primitive_attr_t reduction_attr = *attr();
                int threads_per_eu;
                auto status
                        = gemm_pd_->query(query::preferred_gpu_threads_per_eu,
                                0, &threads_per_eu);
                if (status == status::success)
                    CHECK(reduction_attr.set_gpu_attr(
                            gpu_primitive_attr_t(threads_per_eu)));
                primitive_desc_iterator_t it(engine, (op_desc_t *)&reduction_d,
                        &reduction_attr, nullptr);
                if (!it.is_initialized()) return status::out_of_memory;
                reduction_pd_ = *(++it);
                if (!reduction_pd_) return status::unimplemented;
            }
            if (!gemm_ok) return status::unimplemented;
            init_scratchpad();
            return status::success;
        }

        bool wei_tr() const {
            const auto &wmd = *this->diff_weights_md();
            return wmd.format_desc.blocking.strides[0] == 1;
        }

        std::shared_ptr<primitive_desc_t> gemm_pd_;
        std::shared_ptr<primitive_desc_t> reduction_pd_;

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested_multiple,
                    gemm_pd_->scratchpad_registry());
            if (with_bias() && reduction_pd_)
                scratchpad.book(memory_tracking::names::key_nested_multiple + 1,
                        reduction_pd_->scratchpad_registry());
        }
    };

    status_t init(engine_t *engine) override {
        CHECK(create_nested_primitive(gemm_, pd()->gemm_pd_, engine));
        if (pd()->with_bias() && pd()->reduction_pd_)
            CHECK(create_nested_primitive(
                    reduction_, pd()->reduction_pd_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_weights(ctx);
    }

private:
    status_t execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<primitive_t> gemm_;
    std::shared_ptr<primitive_t> reduction_;
};

} // namespace ocl
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
