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

#ifndef GPU_AMD_MIOPEN_SOFTMAX_HPP
#define GPU_AMD_MIOPEN_SOFTMAX_HPP

#include "hip/hip_runtime.h"

#include "miopen/miopen.h"

#include "common/primitive.hpp"
#include "common/softmax_pd.hpp"
#include "gpu/amd/miopen_softmax_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_softmax_fwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public softmax_fwd_pd_t {
        using softmax_fwd_pd_t::softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_softmax_fwd_t);

        status_t init(engine_t *) {
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper dst_d(dst_md());
            bool ok = is_fwd()
                    && utils::one_of(
                            src_d.data_type(), data_type::f32, data_type::f16)
                    && attr()->has_default_values()
                    && set_default_formats() == status::success
                    && src_d.is_plain() && dst_d.is_plain() && dst_d == src_d
                    && axis() == 1 && check_format();

            if (!ok) return status::unimplemented;

            softmax_impl_.reset(new miopen_softmax_fwd_impl_t());

            return softmax_impl_->init(this);
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

        std::shared_ptr<miopen_softmax_impl_base_t> softmax_impl_;
    };

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

struct miopen_softmax_bwd_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public softmax_bwd_pd_t {
        using softmax_bwd_pd_t::softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_softmax_bwd_t);

        status_t init(engine_t *) {
            const memory_desc_wrapper diff_src_d(diff_src_md());
            const memory_desc_wrapper diff_dst_d(diff_dst_md());
            const memory_desc_wrapper dst_d(dst_md());

            bool ok = !is_fwd()
                    && utils::one_of(
                            dst_d.data_type(), data_type::f32, data_type::f16)
                    && attr()->has_default_values()
                    && set_default_formats() == status::success
                    && dst_d.is_plain() && diff_dst_d.is_plain()
                    && diff_src_d.is_plain() && diff_src_d == diff_dst_d
                    && diff_src_d == dst_d && axis() == 1 && check_format();
            if (!ok) return status::unimplemented;

            softmax_impl_.reset(new miopen_softmax_bwd_impl_t());

            return softmax_impl_->init(this);
        }
        bool check_format() const {
            // Only abx formats are supported
            return (memory_desc_wrapper(dst_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,
                            format_tag::abcd, format_tag::abcde)
                    && memory_desc_wrapper(diff_src_md())
                               .matches_one_of_tag(format_tag::a,
                                       format_tag::ab, format_tag::abc,
                                       format_tag::abcd, format_tag::abcde)
                    && memory_desc_wrapper(diff_dst_md())
                               .matches_one_of_tag(format_tag::a,
                                       format_tag::ab, format_tag::abc,
                                       format_tag::abcd, format_tag::abcde));
        }

        std::shared_ptr<miopen_softmax_impl_base_t> softmax_impl_;
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
