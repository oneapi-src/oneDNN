/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef OCL_CONVOLUTION_PD_HPP
#define OCL_CONVOLUTION_PD_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "ocl/ocl_engine.hpp"

namespace dnnl {
namespace impl {
namespace ocl {

struct ocl_convolution_fwd_pd_t : public convolution_fwd_pd_t {
    using convolution_fwd_pd_t::convolution_fwd_pd_t;

protected:
    bool post_ops_ok(const primitive_attr_t *attr) const {
        const auto &p = attr->post_ops_;

        auto is_eltwise
                = [&](int idx) { return p.entry_[idx].is_eltwise(false); };
        auto is_sum = [&](int idx) { return p.entry_[idx].is_sum(false); };

        bool is_int8 = utils::one_of(
                invariant_src_md()->data_type, data_type::s8, data_type::u8);
        switch (p.len_) {
            case 0: return true; // no post_ops
            case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
            case 2:
                // sum -> eltwise (or eltwise -> sum for int8)
                return (is_sum(0) && is_eltwise(1))
                        || (is_int8 && is_eltwise(0) && is_sum(1));
            default: return false;
        }

        return false;
    }
};

struct ocl_convolution_bwd_data_pd_t : public convolution_bwd_data_pd_t {
    using convolution_bwd_data_pd_t::convolution_bwd_data_pd_t;
};

struct ocl_convolution_bwd_weights_pd_t : public convolution_bwd_weights_pd_t {
    using convolution_bwd_weights_pd_t::convolution_bwd_weights_pd_t;
};

} // namespace ocl
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
