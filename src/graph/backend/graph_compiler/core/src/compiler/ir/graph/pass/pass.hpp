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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_PASS_PASS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_PASS_PASS_HPP

#include <functional>
#include <ios>
#include <string>
#include "../graph.hpp"
#include <unordered_map>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
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
        bool print_shape = false, bool print_attr = false,
        bool print_name = false, bool print_stride = false);

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
 * Do the same as graph_constant_input_folding, except that it also try to share
 * the constant buffer with other graphs. This pass should be put after all
 * other graph_constant_input_folding
 * */
SC_INTERNAL_API void graph_constant_input_folding_and_share_constants(
        sc_graph_t &mgr, const context_ptr &ctx);
/**
 * Mark the elementwise op with padded input/output could use output mask(not
 * mask load/store) or not. The op could use mask when its direct uses have
 * reduce or memory movement semantics.
 */
SC_INTERNAL_API void padded_mask_mark(
        sc_graph_t &graph, const context_ptr &ctx);
/**
 * Enable/Disable some latest optimizations like image-affinity according to the
 * compiler opt level.
 */
SC_INTERNAL_API void intrusive_opt_level(
        sc_graph_t &graph, const context_ptr &ctx);

// find the graph in cached code. If a matched graph is found, the
// compiler_driver/graph_driver can skip the compilation and reuse the code
SC_INTERNAL_API void graph_code_cache(sc_graph_t &mgr, const context_ptr &ctx);

/**
 * Compares the graphs.
 * @param lhs the left hand side graph
 * @param rhs the right hand side graph
 * @param first_diff_lhs outputs the first different LHS Op
 * @param first_diff_rhs outputs the first different RHS Op
 * @param lhs_rhs_input_mapping {the left graph input op id, the right graph
 * input op id} mapping
 * @param filter the filter function for op attr, @see
 * sc_graph_t::compare_contents
 * @return true if the the graphs are the same
 * */
SC_INTERNAL_API bool compare_graph(sc_op_ptr &first_diff_lhs,
        sc_op_ptr &first_diff_rhs, const sc_graph_t &lhs, const sc_graph_t &rhs,
        const std::unordered_map<int, int> &lhs_rhs_input_mapping = {},
        const std::function<bool(const sc_op *, const std::string &)> &filter
        = nullptr);

/**
 * Compares the graphs.
 * @param lhs the left hand side graph
 * @param rhs the right hand side graph
 * @param lhs_rhs_input_mapping {the left graph input op id, the right graph
 * input op id} mapping
 * @param filter the filter function for op attr, @see
 * sc_graph_t::compare_contents
 * @return true if the the graphs are the same
 * */
SC_INTERNAL_API bool compare_graph(const sc_graph_t &lhs, const sc_graph_t &rhs,
        const std::unordered_map<int, int> &lhs_rhs_input_mapping = {},
        const std::function<bool(const sc_op *, const std::string &)> &filter
        = nullptr);

namespace runtime {
struct dynamic_tensor_t;
}
/**
 * The api for dynamic infer shape.
 * At runtime, given input real shapes of partition, infer the output shapes.
 * After calling this api, the sizes of output buffers are known for allocation.
 * @param graph cached raw graph after translation(after graph inline and
 * constant optimization) but not graph driver. We need original graph cache to
 * preserve all semantics of ops
 * @param ins a pointer to dynamic tensor vector of inputs, contains real shape
 * info.
 * @param outs a pointer to dynamic tensor vector of outputs, the real shape
 * need to be infered.
 * @param num_ins, the length of input dynamic tensor vector, should be equal to
 * number of inputs in graph.
 * @param num_outs, the length of output dynamic tensor vector, should be equal
 * to number of outputs in graph.
 * */
SC_API void dynamic_infer_shape_by_graph(sc_graph_t &graph,
        runtime::dynamic_tensor_t **ins, runtime::dynamic_tensor_t **outs,
        size_t num_ins, size_t num_outs);

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
