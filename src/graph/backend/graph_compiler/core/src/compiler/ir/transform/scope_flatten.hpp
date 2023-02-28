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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_SCOPE_FLATTEN_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_SCOPE_FLATTEN_HPP

#include <vector>
#include "../function_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * Merge one nested stmts_node_t scope into parent scope:
 * Before:
 * {
 *  { // <------stmt_index pointing here
 *      AAA
 *  }
 *  BBB
 * }
 * After:
 * {
 *  AAA
 *  BBB
 * }
 * @param seq the parent scope
 * @param stmt_index the child stmts node index to merge within parent scope. If
 *  it is not an stmts, do nothing. If stmt_index < 0, will try to flatten all
 *  stmts in the parent scope (not recursively)
 * */
void scope_flatten(std::vector<stmt> &seq, int stmt_index);
void scope_flatten(const stmt &seq, int stmt_index);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
