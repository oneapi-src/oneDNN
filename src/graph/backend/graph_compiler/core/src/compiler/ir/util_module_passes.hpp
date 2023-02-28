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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_UTIL_MODULE_PASSES_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_UTIL_MODULE_PASSES_HPP

#include <utility>
#include <vector>
#include "function_pass.hpp"
#include "module_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

// The pass to wrap function pass to module pass. It will run the function pass
// on each of the function in the input module
class module_function_pass_t : public module_pass_t {
public:
    function_pass_ptr impl_;
    module_function_pass_t(function_pass_ptr impl);
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
    SC_DECL_PASS_INFO_FUNC();
    // makes a module_function_pass_t from a function_pass_t
    // T should be a function_pass_t class
    template <typename T, typename... Args>
    static module_pass_ptr make(Args &&...args) {
        return utils::make_unique<module_function_pass_t>(
                utils::make_unique<T>(std::forward<Args>(args)...));
    }
};

// Sequentially run module_passes on the input module. The output of the current
// pass will be the input of the next pass
class SC_INTERNAL_API sequential_module_pass_t : public module_pass_t {
public:
    std::vector<module_pass_ptr> passes_;
    sequential_module_pass_t(std::vector<module_pass_ptr> &&passes);
    sequential_module_pass_t(sequential_module_pass_t &&other);
    template <typename... Args>
    sequential_module_pass_t(Args &&...args) {
        utils::args_to_vector<module_pass_ptr>(passes_, std::move(args)...);
    }
    const_ir_module_ptr operator()(const_ir_module_ptr f) override;
};

class ir_visitor_t;
// dispatch the global variables and functions in the module on the visitor,
// returns a new module with updated members
const_ir_module_ptr dispatch_module_on_visitor(
        ir_visitor_t *vis, const const_ir_module_ptr &f);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
