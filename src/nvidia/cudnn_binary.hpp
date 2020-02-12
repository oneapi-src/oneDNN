/***************************************************************************
 *  Copyright 2020 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 **************************************************************************/

#ifndef CUDNN_BINARY_HPP
#define CUDNN_BINARY_HPP

#include "cudnn.h"

#include <CL/sycl.hpp>

#include "common/binary_pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "nvidia/cudnn_binary_impl.hpp"
#include "nvidia/sycl_cuda_engine.hpp"
#include "nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace cuda {

struct cudnn_binary_t : public primitive_t {

    struct pd_t : public binary_pd_t {
        using binary_pd_t::binary_pd_t;

        DECLARE_COMMON_PD_T("cuda:cudnn:any", cudnn_binary_t);

        status_t init(engine_t *) {
            using namespace data_type;
            if (check_for_zero_dims()) { return status::success; };
            bool ok = true && (set_default_params() == status::success)
                    && check_data_types() && check_no_blocking()
                    && IMPLICATION(
                            utils::one_of(src_md(0)->data_type, f32, f16),
                            attr()->has_default_values())
                    && IMPLICATION(utils::one_of(src_md(0)->data_type, s8),
                            attr()->has_default_values(
                                    primitive_attr_t::skip_mask_t::scales))
                    && IMPLICATION(!attr()->scales_.has_default_values(),
                            check_scales_mask());

            if (!ok) return status::unimplemented;

            binary_impl.reset(new cudnn_binary_impl_t());

            return binary_impl->init(this);
        }

        bool check_for_zero_dims() {
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

        bool check_no_blocking() {
            // Blocking is not supported by cudnnOpTensor, return false if any
            // blocks are present
            return src_md(0)->format_desc.blocking.inner_nblks
                    + src_md(1)->format_desc.blocking.inner_nblks
                    + dst_md()->format_desc.blocking.inner_nblks
                    == 0;
        }

        bool check_data_types() {
            using namespace data_type;
            bool inputs_same = src_md(0)->data_type == src_md(1)->data_type;
            dnnl_data_type_t input_type = src_md(0)->data_type;
            dnnl_data_type_t output_type = dst_md()->data_type;

            switch (output_type) {
                case f32:
                    return inputs_same
                            && (input_type == f32 || input_type == s8
                                    || input_type == f16);
                case f16:
                    return inputs_same
                            && (input_type == f32 || input_type == f16);
                case s8:
                    return inputs_same
                            && (input_type == f32 || input_type == s8);
            }
            return false;
        }
        std::shared_ptr<cudnn_binary_impl_base_t> binary_impl;
    };

    cudnn_binary_t(const pd_t *apd) : primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cuda
} // namespace impl
} // namespace dnnl

#endif // CUDNN_BINARY_HPP
