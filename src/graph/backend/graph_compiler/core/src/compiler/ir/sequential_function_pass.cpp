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

#include "sequential_function_pass.hpp"
#include <memory>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
sequential_function_pass_t::sequential_function_pass_t(
        std::vector<std::unique_ptr<function_pass_t>> &&passes)
    : passes_(std::move(passes)) {}

sequential_function_pass_t::sequential_function_pass_t(
        sequential_function_pass_t &&other)
    : passes_(std::move(other.passes_)) {}

func_c sequential_function_pass_t::operator()(func_c f) {
    for (auto &p : passes_) {
        f = (*p)(f);
    }
    return f;
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
