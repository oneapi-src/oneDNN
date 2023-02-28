/*******************************************************************************
 * Copyright 2022-2023 Intel Corporation
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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_SSA_TRANSFORM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_SSA_TRANSFORM_HPP

#include <compiler/ir/function_pass.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Converts the IR to SSA (Single-Static-Assignment) form.
 * 1. It flattens the exprs and makes sure that, for any expr node, its
 * sub-nodes must be constants or var nodes
 * 2. Any expr node must assigned to a var_node in define_node
 * 3. The assign nodes to local variables will be replaced by defining new
 * variables. And we make sure a local variable is assigned only once (at the
 * define_node, Single-Static-Assignment). Note that the assignments to global
 * variable or local/global tensor elements are unchanged
 * 4. expr_base::ssa_data_ will be available after this pass
 * */
class ssa_transform_t : public function_pass_t {
public:
    func_c operator()(func_c f) override;
    stmt_c operator()(stmt_c f);
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
