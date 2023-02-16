/*******************************************************************************
 * Copyright 2020-2023 Intel Corporation
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

#include "quantize_info.hpp"
#include <compiler/ir/sc_expr.hpp>
namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

static std::vector<union_val> get_padded_union_val_vector(
        const std::vector<int> &input, unsigned padded) {
    if (input.empty() || padded <= 0) { return std::vector<union_val>(); }
    std::vector<union_val> results(padded);
    for (unsigned i = 0; i < padded; i++) {
        if (i < input.size()) {
            results[i] = (int64_t)input[i];
        } else {
            results[i] = (int64_t)0;
        }
    }
    return results;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
