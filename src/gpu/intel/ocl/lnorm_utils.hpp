/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gpu/intel/compute/dispatch_reusable.hpp"

#include <cassert>
#include <cstddef>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace ocl {

namespace lnorm_dims {
constexpr dim_idx_t mb = 0;
constexpr dim_idx_t ic = 1;
constexpr dim_idx_t sp0 = 2;
constexpr dim_idx_t sp1 = 3;
constexpr dim_idx_t sp2 = 4;
}; // namespace lnorm_dims

static std::vector<dim_idx_t> get_dims(size_t ndims, bool for_stats = false) {
    assert(ndims > 1 && ndims < 6);
    // The last logical dimension is not included in lnorm stats
    if (for_stats) ndims--;
    std::vector<dim_idx_t> ret(ndims);
    uint8_t idx = 0;
    ret[idx++] = lnorm_dims::mb;
    if (ndims >= 2) ret[idx++] = lnorm_dims::ic;
    if (ndims >= 3) ret[idx++] = lnorm_dims::sp0;
    if (ndims >= 4) ret[idx++] = lnorm_dims::sp1;
    if (ndims >= 5) ret[idx++] = lnorm_dims::sp2;
    return ret;
}

static compute::named_buffer_t get_ss_buffer(
        const memory_desc_t *md, dim_idx_t dim) {
    if (types::is_zero_md(md)) {
        // Scale/shift are unused. We need to construct a buffer that will not be dispatched to
        compute::named_buffer_t ret("SS");
        ret.data_type = data_type::f32; // Anything but undef
        return ret;
    } else {
        return compute::named_buffer_t("SS", *md, {dim});
    }
}

} // namespace ocl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
