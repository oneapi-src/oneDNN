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

#include "gpu/intel/compute/utils.hpp"
#include "common/verbose.hpp"

#include <limits>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace compute {

void check_global_range(const compute::range_t &range) {
    bool exceeds_32bit = false;
    const size_t u32_max = std::numeric_limits<uint32_t>::max();
    for (size_t i = 0; i < range.ndims(); i++) {
        if (range[i] > u32_max) {
            exceeds_32bit = true;
            break;
        }
    }
    if (exceeds_32bit) {
        VERROR(common, runtime,
                "global work size exceeds the 32-bit limit. Potential "
                "correctness issues may arise due to driver limitation");
    }
}

} // namespace compute
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
