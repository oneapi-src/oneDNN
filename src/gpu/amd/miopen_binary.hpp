/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

#ifndef GPU_AMD_MIOPEN_BINARY_HPP
#define GPU_AMD_MIOPEN_BINARY_HPP

#include "common/binary_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/amd/miopen_binary_impl.hpp"
#include "gpu/amd/sycl_hip_engine.hpp"
#include "gpu/amd/sycl_hip_utils.hpp"
#include <miopen/miopen.h>

namespace dnnl {
namespace impl {
namespace gpu {
namespace amd {

struct miopen_binary_t : public primitive_t {
    using primitive_t::primitive_t;

    struct pd_t : public binary_pd_t {
        using binary_pd_t::binary_pd_t;

        DECLARE_COMMON_PD_T("hip:miopen:any", miopen_binary_t);

        status_t init(impl::engine_t *) {
            using namespace data_type;

            bool ok = (set_default_params() == status::success)
                    && check_data_types() && check_no_blocking()
                    && check_broadcast()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::scales_runtime)
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask())
                    && check_format();

            if (!ok) return status::unimplemented;

            if (check_for_zero_dims()) return status::success;

            binary_impl_.reset(new miopen_binary_impl_t());

            return binary_impl_->init(this);
        }

        bool check_for_zero_dims() const {
            return has_zero_dims(src_md(0)->dims, src_md(0)->ndims)
                    || has_zero_dims(src_md(1)->dims, src_md(1)->ndims)
                    || has_zero_dims(dst_md()->dims, dst_md()->ndims);
        }

        bool check_scales_mask() const {
            for (const auto &s : attr()->scales_.scales_) {
                if (s.second.mask_ != 0) return false;
            }
            return true;
        }

        bool check_no_blocking() const {
            // Blocking is not supported by MIOPENOpTensor, return false if any
            // blocks are present
            return src_md(0)->format_desc.blocking.inner_nblks
                    + src_md(1)->format_desc.blocking.inner_nblks
                    + dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        bool check_broadcast() const {
            // Source 0 broadcast is not supported
            const int ndims = nstl::min(src_md(0)->ndims, src_md(1)->ndims);
            for (int dim_idx = 0; dim_idx < ndims; dim_idx++) {
                if (src_md(0)->dims[dim_idx] == 1
                        && src_md(0)->dims[dim_idx] != src_md(1)->dims[dim_idx])
                    return false;
            }
            return true;
        }

        bool check_format() const {
            // Only abx formats are supported
            return (memory_desc_wrapper(src_md(0)).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,
                            format_tag::abcd, format_tag::abcde)
                    && memory_desc_wrapper(src_md(1)).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,
                            format_tag::abcd, format_tag::abcde)
                    && memory_desc_wrapper(dst_md()).matches_one_of_tag(
                            format_tag::a, format_tag::ab, format_tag::abc,
                            format_tag::abcd, format_tag::abcde));
        }

        bool check_data_types() const {
            using namespace data_type;
            data_type_t input1_type = src_md(0)->data_type;
            data_type_t input2_type = src_md(1)->data_type;
            data_type_t input_type = input1_type;
            data_type_t output_type = dst_md()->data_type;
            bool inputs_same = (input1_type == input2_type);

            switch (output_type) {
                case f32: return inputs_same && (input_type == f32);
                case f16: return inputs_same && (input_type == f16);
                case s32: return inputs_same && (input_type == s32);
                default: break;
            }

            return false;
        }
        std::shared_ptr<miopen_binary_impl_base_t> binary_impl_;
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
