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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SEQUENTIAL_FUNCTION_PASS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_SEQUENTIAL_FUNCTION_PASS_HPP

#include <utility>
#include <vector>
#include "function_pass.hpp"
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
class sequential_function_pass_t : public function_pass_t {
public:
    std::vector<function_pass_ptr> passes_;
    sequential_function_pass_t(std::vector<function_pass_ptr> &&passes);
    sequential_function_pass_t(sequential_function_pass_t &&other);
    func_c operator()(func_c f) override;
    template <typename... Args>
    sequential_function_pass_t(Args &&...args) {
        utils::args_to_vector<function_pass_ptr>(passes_, std::move(args)...);
    }
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
