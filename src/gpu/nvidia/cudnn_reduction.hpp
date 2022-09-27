/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020 Codeplay Software Limited
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

#ifndef GPU_NVIDIA_CUDNN_REDUCTION_HPP
#define GPU_NVIDIA_CUDNN_REDUCTION_HPP

#include "cudnn.h"

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/reduction_pd.hpp"
#include "gpu/nvidia/cudnn_reduction_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_reduction_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public reduction_pd_t {
        using reduction_pd_t::reduction_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_reduction_t);
        status_t init(engine_t *engine) {
            using namespace data_type;

            const bool ok = (set_default_params() == status::success)
                    && attr()->has_default_values()
                    && utils::one_of(src_md()->data_type, data_type::f32,
                            data_type::f16, data_type::s8)
                    && utils::one_of(dst_md()->data_type, data_type::f32,
                            data_type::f16, data_type::s8)
                    && check_no_blocking() && check_format()
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

            reduction_impl_.reset(new cudnn_reduction_impl_t());
            auto status = reduction_impl_->init(this);

            if (status == status::success)
                reduction_impl_->create_and_set_workspace(this, engine);
            return status;
        }

        bool check_for_zero_dims() const {
            return has_zero_dims(src_md()->dims, src_md()->ndims)
                    || has_zero_dims(dst_md()->dims, dst_md()->ndims);
        }

        bool check_no_blocking() const {
            // Blocking is not supported by cudnnReduceTensor, return false if
            // any blocks are present.
            return src_md(0)->format_desc.blocking.inner_nblks
                    + src_md(1)->format_desc.blocking.inner_nblks
                    + dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        bool check_format() const {
            // cudnnReduceTensor produces incorrect results when src format
            // is not nchw or its derivatives.
            return memory_desc_wrapper(src_md()).matches_one_of_tag(
                    format_tag::a, format_tag::ab, format_tag::abc,
                    format_tag::abcd, format_tag::abcde);
        }

        std::shared_ptr<cudnn_reduction_impl_base_t> reduction_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
