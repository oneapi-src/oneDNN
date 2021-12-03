/*******************************************************************************
 * Copyright 2020-2021 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_LOWERING_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_LOWERING_HPP
#include <vector>
#include "graph.hpp"

namespace sc {

/**
 * Generates the ir_module_t from the OP graph
 * @param ctx the context
 * @param graph the graph
 * @param args optional order of the arguments of the generated IR function.
 * Should all be input_op or output_op. If empty, use default internal order
 * */
SC_API ir_module_ptr lower_graph(
        context_ptr ctx, sc_graph_t &graph, const std::vector<sc_op_ptr> &args);
} // namespace sc

#endif
