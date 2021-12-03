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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_PASS_PASS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_PASS_PASS_HPP

#include <ios>
#include <string>
#include "../graph.hpp"
#include <unordered_map>
namespace sc {
// Used in `graph_constant_input_folding`, stores as an attribute
// "constant" in nodes which describe current node's outputs status.
namespace const_kind {
// means non constant node
constexpr int not_const = 0;
// means intermediate constant node, with constant inputs and its
// outputs will be used to calculate local/global constant node in init stage.
constexpr int local_const = 1;
// means final constant node, with constant inputs and its outputs will be
// lowerred to global variables and be used as inputs in non-const nodes.
constexpr int global_const = 2;
} // namespace const_kind

SC_INTERNAL_API void print_graph(const sc_graph_t &mgr, std::ostream &os,
        bool print_shape = false, bool print_attr = false);

void visualize(const std::string &filename, const sc_graph_t &opmg);

SC_INTERNAL_API void save_graph_to_json(
        const sc_graph_t &graph, std::ostream &os);
SC_INTERNAL_API sc_graph_t load_graph_from_json(std::istream &is);

SC_INTERNAL_API sc_graph_t copy_graph(const sc_graph_t &graph);

SC_API bool check_graph_connection(sc_graph_t &graph);
bool check_graph_config(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());
/**
 * Folding nodes with constant attribute from inputs in graph.
 * In this pass, we only propagate "constant" states from constant op and input
 * op which marked `constant`. Actual folding will occur in lowering.
 * */
SC_INTERNAL_API void graph_constant_input_folding(
        sc_graph_t &graph, const context_ptr &ctx = get_default_context());

/**
 * Compares the graphs.
 * @param lhs the left hand side graph
 * @param rhs the right hand side graph
 * @param first_diff_lhs outputs the first different LHS Op
 * @param first_diff_rhs outputs the first different RHS Op
 * @param lhs_rhs_input_mapping {the left graph input op id, the right graph
 * input op id} mapping
 * @return true if the the graphs are the same
 * */
SC_INTERNAL_API bool compare_graph(sc_op_ptr &first_diff_lhs,
        sc_op_ptr &first_diff_rhs, const sc_graph_t &lhs, const sc_graph_t &rhs,
        const std::unordered_map<int, int> &lhs_rhs_input_mapping = {});

/**
 * Compares the graphs.
 * @param lhs the left hand side graph
 * @param rhs the right hand side graph
 * @param lhs_rhs_input_mapping {the left graph input op id, the right graph
 * input op id} mapping
 * @return true if the the graphs are the same
 * */
SC_INTERNAL_API bool compare_graph(const sc_graph_t &lhs, const sc_graph_t &rhs,
        const std::unordered_map<int, int> &lhs_rhs_input_mapping = {});

} // namespace sc

#endif
