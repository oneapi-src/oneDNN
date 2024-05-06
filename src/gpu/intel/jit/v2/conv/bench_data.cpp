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

#include "gpu/intel/jit/v2/conv/bench_data.hpp"

#include <sstream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {
namespace conv {

std::string bench_data_t::str() const {
    std::ostringstream oss;
    for (int i = 0; i < size(); i++) {
        if (i > 0) oss << std::endl;
        oss << "bench," << prbs[i].csv_str() << "," << times[i];
    }
    return oss.str();
}

} // namespace conv
} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
