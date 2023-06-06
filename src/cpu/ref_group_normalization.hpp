/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef CPU_REF_GROUP_NORMALIZATION_HPP
#define CPU_REF_GROUP_NORMALIZATION_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"

#include "cpu/cpu_group_normalization_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_group_normalization_fwd_t : public primitive_t {
    struct pd_t : public cpu_group_normalization_fwd_pd_t {
        using cpu_group_normalization_fwd_pd_t::
                cpu_group_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_group_normalization_fwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            VDISPATCH_GNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_GNORM(
                    utils::one_of(src_md()->data_type, f32, bf16, f16, s8, u8)
                            && platform::has_data_type_support(
                                    src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(
                    utils::one_of(dst_md()->data_type, f32, bf16, f16, s8, u8)
                            && platform::has_data_type_support(
                                    dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(
                    attr()->has_default_values(skip_mask_t::scales_runtime)
                            && attr_scales_ok(),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_GNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
    };

    using primitive_t::primitive_t;

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct ref_group_normalization_bwd_t : public primitive_t {
    struct pd_t : public cpu_group_normalization_bwd_pd_t {
        using cpu_group_normalization_bwd_pd_t::
                cpu_group_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_group_normalization_bwd_t);

        status_t init(engine_t *engine) {
            using namespace data_type;

            VDISPATCH_GNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);

            VDISPATCH_GNORM(
                    utils::one_of(src_md()->data_type, f32, bf16, f16, s8, u8)
                            && platform::has_data_type_support(
                                    src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(utils::one_of(diff_dst_md()->data_type, f32, bf16,
                                    f16, s8, u8)
                            && platform::has_data_type_support(
                                    diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(utils::one_of(diff_src_md()->data_type, f32, bf16,
                                    f16, s8, u8)
                            && platform::has_data_type_support(
                                    diff_src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_GNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_GNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
    };

    ref_group_normalization_bwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
