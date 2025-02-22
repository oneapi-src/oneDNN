/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef CPU_X64_MATMUL_INNER_PRODUCT_HPP
#define CPU_X64_MATMUL_INNER_PRODUCT_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/matmul_pd.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

#include "cpu/cpu_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

status_t create_matmul_pd(std::shared_ptr<primitive_desc_t> &matmul_pd,
        engine_t *engine, const memory_desc_t *a_md, const memory_desc_t *b_md,
        const memory_desc_t *c_md, const memory_desc_t *ip_bia_md,
        const memory_desc_t *reduce_md, const primitive_attr_t *attr);

status_t init_matmul_md(memory_desc_t &mm_md, const memory_desc_t &ip_md,
        format_tag_t tag, bool swap_dims = false);

status_t set_training_formats(memory_desc_t *src_md, memory_desc_t *wei_md,
        memory_desc_t *bias_md, memory_desc_t *dst_md);

struct matmul_inner_product_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T((matmul_pd_ ? matmul_pd_->name() : "matmul"),
                matmul_inner_product_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            const auto src_dt = invariant_src_md()->data_type;
            const auto wei_dt = invariant_wei_md()->data_type;
            const auto dst_dt = invariant_dst_md()->data_type;
            const bool is_int8 = utils::one_of(src_dt, u8, s8) && wei_dt == s8
                    && utils::one_of(dst_dt, u8, s8, s32, f32, bf16);

            auto skip_mask = skip_mask_t::post_ops | skip_mask_t::sum_dt
                    | skip_mask_t::fpmath_mode;
            if (is_int8) skip_mask |= skip_mask_t::scales;

            // This implementation is currently enabled only for inference.
            VDISPATCH_INNER_PRODUCT(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_INNER_PRODUCT(attr()->has_default_values(skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);

            if (get_prop_kind() == prop_kind::forward_training) {
                VDISPATCH_INNER_PRODUCT_SC(
                        set_training_formats(
                                &src_md_, &weights_md_, &bias_md_, &dst_md_),
                        VERBOSE_UNSUPPORTED_TAG);
            }

            VDISPATCH_INNER_PRODUCT_SC(
                    init_matmul_params(engine), "init_matmul_params");
            init_scratchpad();

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> matmul_pd_;

    private:
        int get_k_blk(format_tag_t tag) const;
        status_t init_matmul_params(engine_t *engine);

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    matmul_pd_->scratchpad_registry());
        }
    };

    status_t init(impl::engine_t *engine) override {
        CHECK(pd()->matmul_pd_->create_primitive(matmul_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_;
};

struct matmul_inner_product_bwd_data_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_inner_product_bwd_data_pd_t {
        using cpu_inner_product_bwd_data_pd_t::cpu_inner_product_bwd_data_pd_t;

        DECLARE_COMMON_PD_T((matmul_pd_ ? matmul_pd_->name() : "matmul"),
                matmul_inner_product_bwd_data_t);

        bool has_type(data_type_t v) const {
            return utils::one_of(v, weights_md()->data_type,
                    diff_src_md()->data_type, diff_dst_md()->data_type);
        }

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            const auto diff_src_dt = invariant_src_md()->data_type;
            const auto diff_dst_dt = invariant_dst_md()->data_type;
            const auto wei_dt = invariant_wei_md()->data_type;

            const bool is_f32
                    = utils::everyone_is(f32, diff_src_dt, wei_dt, diff_dst_dt);

            VDISPATCH_INNER_PRODUCT(mayiuse(avx2), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_INNER_PRODUCT(IMPLICATION(!is_f32, mayiuse(avx512_core)),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_INNER_PRODUCT(get_prop_kind() == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT_SC(set_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_INNER_PRODUCT(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_INNER_PRODUCT(utils::one_of(diff_dst_dt, f32, bf16, f16,
                                            f8_e5m2, f8_e4m3),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(wei_dt == diff_dst_dt,
                    VERBOSE_INCONSISTENT_DT, "weights", "diff_dst");
            VDISPATCH_INNER_PRODUCT(
                    utils::one_of(diff_src_dt, f32, diff_dst_dt),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(
                    attr()->has_default_values(skip_mask_t::fpmath_mode),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_INNER_PRODUCT_SC(
                    init_matmul_params(engine), "init_matmul_params");
            init_scratchpad();

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> matmul_pd_;

    private:
        status_t init_matmul_params(engine_t *engine);
        status_t set_formats() {
            return set_training_formats(
                    &diff_src_md_, &weights_md_, nullptr, &diff_dst_md_);
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    matmul_pd_->scratchpad_registry());
        }
    };

    status_t init(impl::engine_t *engine) override {
        return pd()->matmul_pd_->create_primitive(matmul_, engine);
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_;
};

struct matmul_inner_product_bwd_weights_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_inner_product_bwd_weights_pd_t {
        using cpu_inner_product_bwd_weights_pd_t::
                cpu_inner_product_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T((matmul_pd_ ? matmul_pd_->name() : "matmul"),
                matmul_inner_product_bwd_weights_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            const auto src_dt = invariant_src_md()->data_type;
            const auto diff_wei_dt = invariant_wei_md()->data_type;
            const auto diff_dst_dt = invariant_dst_md()->data_type;
            const auto diff_bia_dt = invariant_bia_md()->data_type;

            const bool is_f32
                    = utils::everyone_is(f32, src_dt, diff_wei_dt, diff_dst_dt)
                    && IMPLICATION(with_bias(), diff_bia_dt == f32);

            VDISPATCH_INNER_PRODUCT(mayiuse(avx2), VERBOSE_UNSUPPORTED_ISA);
            VDISPATCH_INNER_PRODUCT(IMPLICATION(!is_f32, mayiuse(avx512_core)),
                    VERBOSE_UNSUPPORTED_DT_CFG);

            VDISPATCH_INNER_PRODUCT(
                    get_prop_kind() == prop_kind::backward_weights,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT_SC(set_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_INNER_PRODUCT(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_INNER_PRODUCT(
                    utils::one_of(src_dt, f32, bf16, f16, f8_e5m2, f8_e4m3),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(diff_dst_dt == src_dt,
                    VERBOSE_INCONSISTENT_DT, "diff_dst", "src");
            VDISPATCH_INNER_PRODUCT(utils::one_of(diff_wei_dt, f32, src_dt),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_INNER_PRODUCT(
                    attr()->has_default_values(skip_mask_t::fpmath_mode),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_INNER_PRODUCT_SC(
                    init_matmul_params(engine), "init_matmul_params");
            init_scratchpad();

            return status::success;
        }

        std::shared_ptr<primitive_desc_t> matmul_pd_;

    private:
        status_t init_matmul_params(engine_t *engine);
        status_t set_formats() {
            return set_training_formats(
                    &src_md_, &diff_weights_md_, &diff_bias_md_, &diff_dst_md_);
        }

        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_nested,
                    matmul_pd_->scratchpad_registry());
        }
    };

    status_t init(impl::engine_t *engine) override {
        CHECK(pd()->matmul_pd_->create_primitive(matmul_, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> matmul_;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
