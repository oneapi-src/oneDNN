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

#ifndef GPU_GPU_ZERO_POINTS_CONV_HPP
#define GPU_GPU_ZERO_POINTS_CONV_HPP

#include "common/primitive_desc.hpp"

namespace dnnl {
namespace impl {
namespace gpu {

status_t create_zp_precompute_conv_pd(std::shared_ptr<primitive_desc_t> &retn,
        dnnl::impl::engine_t *eng, const primitive_attr_t &attr,
        const memory_desc_t *wei, const dim_t *idhw, const dim_t *odhw,
        const dim_t *pdhw, const dim_t *ddhw, data_type_t out_type,
        prop_kind_t prop, bool has_offset0 = true);

} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
