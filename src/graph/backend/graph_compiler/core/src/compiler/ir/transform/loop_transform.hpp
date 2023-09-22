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
#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_LOOP_TRANSFORM_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_LOOP_TRANSFORM_HPP
#include <vector>
#include <compiler/ir/sc_stmt.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
/**
 * Removes redundant loops with parallel attribute. Will reserve the outmost
 * one loop with parallel.
 * @param body the stmts for parallel remove
 * @param ignore_nested_parallel decides whether finally remove the parallel
 * loop which also owns `num_threads_` field
 * */
void remove_parallel(stmt body, bool ignore_nested_parallel = false);

void remove_parallel(func_t body, bool ignore_nested_parallel = false);

/**
 * Collect loops inside this body. Won't recurisvely look into loop body.
 * For example.
 *  for()     # loop 1
 *    for()   # loop 2
 *  for()     # loop 3
 *    for()   # loop 4
 * Only loop 1 and 3 are returned.
 * @param body the stmts for collection
 * */
std::vector<for_loop> collect_loops(stmt body);

/**
 * Collect nested loops inside this body.
 * For example.
 *  for()     # loop 1
 *    for()   # loop 2
 *      for() # loop 3
 *      for() # loop 4
 *
 * Only loop 1 and 2 are returned because loop 3 and 4 are not nested loop.
 *
 * @param body the stmts for collection
 * */
std::vector<for_loop> collect_nested_loops(stmt body);

/**
 * Collect all loops inside this body recursively
 * */
std::vector<for_loop> collect_all_loops(const stmt &body);

// get inner for_loop
for_loop get_inner_for_loop(const for_loop_node_t *f);

// get last for_loop in body
for_loop get_last_loop_in_body(const stmt &body);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
