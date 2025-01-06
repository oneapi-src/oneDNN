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

#include "cpu/cpu_inner_product_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

status_t create_matmul_pd(std::shared_ptr<primitive_desc_t> &matmul_pd,
        engine_t *engine, const memory_desc_t *a_md, const memory_desc_t *b_md,
        const memory_desc_t *c_md, const memory_desc_t *ip_bia_md,
        const primitive_attr_t *attr);

status_t init_matmul_md(memory_desc_t &mm_md, const memory_desc_t &ip_md,
        format_tag_t tag, bool swap_dims = false);

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
            if (is_int8) skip_mask |= skip_mask_t::scales_runtime;

            // This implementation is currently enabled only for inference.
            VDISPATCH_INNER_PRODUCT(
                    get_prop_kind() == prop_kind::forward_inference,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_INNER_PRODUCT(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
            VDISPATCH_INNER_PRODUCT(attr()->has_default_values(skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
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

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
