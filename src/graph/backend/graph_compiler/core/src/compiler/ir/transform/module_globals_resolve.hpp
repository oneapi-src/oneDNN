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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_MODULE_GLOBALS_RESOLVE_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_MODULE_GLOBALS_RESOLVE_HPP

#include "../module_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace attr_keys {
constexpr const char *static_global = "static_global";
constexpr const char *module_global_offset = "module_global_offset";
} // namespace attr_keys
/**
 * Resolves global variables and tensors to module_globals_pointer + offsets.
 * Also modifies all function arguments defined in the current module by
 * appending an additional argument for module_globals_pointer
 * */
class module_globals_resolver_t : public module_pass_t {
public:
    const_ir_module_ptr operator()(const_ir_module_ptr m) override;
    SC_DECL_PASS_INFO_FUNC();
};
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
