/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2022 Codeplay Software Limited
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

#ifndef GPU_AMD_MIOPEN_REDUCTION_HPP
#define GPU_AMD_MIOPEN_REDUCTION_HPP

#include <miopen/miopen.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/reduction_pd.hpp"
#include "gpu/amd/miopen_reduction_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_stream.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_reduction_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public reduction_pd_t {
        using reduction_pd_t::reduction_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_reduction_t);
        status_t init(engine_t *engine) {
            using namespace data_type;

            const bool ok = (set_default_params() == status::success)
                    && attr()->has_default_values()
                    && utils::one_of(
                            src_md()->data_type, data_type::f32, data_type::f16)
                    && utils::one_of(
                            dst_md()->data_type, data_type::f32, data_type::f16)
                    && check_format()
                    && utils::one_of(desc()->alg_kind, alg_kind::reduction_max,
                            alg_kind::reduction_min, alg_kind::reduction_sum,
                            alg_kind::reduction_mul, alg_kind::reduction_mean,
                            alg_kind::reduction_norm_lp_sum,
                            alg_kind::reduction_norm_lp_power_p_sum)
                    && IMPLICATION(
                            desc()->alg_kind == alg_kind::reduction_norm_lp_sum,
                            desc()->p == 2)
                    && IMPLICATION(desc()->alg_kind
                                    == alg_kind::reduction_norm_lp_power_p_sum,
                            desc()->p == 1)
                    && desc()->eps == 0.f;

            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            reduction_impl_.reset(new miopen_reduction_impl_t());
            auto status = reduction_impl_->init(this);

            if (status == status::success)
                reduction_impl_->create_and_set_workspace(this, engine);
            return status;
        }

        bool check_for_zero_dims() const {
            return has_zero_dims(src_md()->dims, src_md()->ndims)
                    || has_zero_dims(dst_md()->dims, dst_md()->ndims);
        }

        bool check_format() const {
            // Only abx formats are supported
            return (memory_desc_wrapper(src_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,
                            format_tag::abcd, format_tag::abcde)
                    && memory_desc_wrapper(dst_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,
                            format_tag::abcd, format_tag::abcde));
        }

        std::shared_ptr<miopen_reduction_impl_base_t> reduction_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace amd
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
