/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace experimental {

// Bnorm expermental feature: calculate mean & variance in single pass over
// input tensor. Improves performance by 25-33% but uses numerically unstable
// formula.
bool DNNL_API use_bnorm_stats_one_pass() {
#ifdef DNNL_EXPERIMENTAL
    static const bool stats_onepass_algo
            = getenv_int_user("EXPERIMENTAL_BNORM_STATS_ONE_PASS", 1);
#else
    static const bool stats_onepass_algo = false;
#endif
    return stats_onepass_algo;
}

} // namespace experimental
} // namespace impl
} // namespace dnnl
