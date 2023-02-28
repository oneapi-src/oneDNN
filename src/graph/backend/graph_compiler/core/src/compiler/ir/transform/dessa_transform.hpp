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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_DESSA_TRANSFORM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_DESSA_TRANSFORM_HPP

#include <compiler/ir/function_pass.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Converts the IR from SSA (Single-Static-Assignment) form to normal IR for
 * codegen. It mainly does two things:
 * 1. Move the var define nodes to an appropriate scope which covers all uses.
 * Reason: in SSA form, we allow a variable to live across the scopes. e.g. a
 * var can be defined in "then" block of if-else and be used after if-else ends.
 * We need to move the def of the var to a parent scope to make the IR valid. We
 * also need to take the PHI node's inputs into consideration when selecting
 * where to put the definition
 * 2. Resolve PHI nodes. There are two kinds of PHI nodes, which are 1) PHI in
 * for-loops which depends on the value of a var of the previous iteration 2)
 * PHI unrelated with loops. If all/none of the PHI node inputs depend on the
 * values defined in the current for-loop, the PHI is of kind 1), otherwise it
 * is of kind 2). For kind 1), we need a "shadow" copy for the var in the loop
 * to solve the problems like "lost copy" or "swap problem". For kind 2), we
 * just need to add an assign node for each of the inputs of PHI
 * */
class dessa_transform_t : public function_pass_t {
public:
    func_c operator()(func_c f) override;
    SC_DECL_PASS_INFO_FUNC();
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
