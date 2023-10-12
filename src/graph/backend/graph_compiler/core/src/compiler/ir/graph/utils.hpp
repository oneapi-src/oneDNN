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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_UTILS_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_UTILS_HPP

#include <memory>
#include <string>
#include <vector>
#include <compiler/ir/graph/graph.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
// do divide and ceil on expr.
expr divide_and_ceil(const expr &, const expr &);
namespace graph {
sc_graph_t make_single_op_graph(const std::string &opname,
        const std::vector<graph_tensor_ptr> &inputs,
        const std::vector<graph_tensor_ptr> &outputs = {},
        const any_map_t &attr = {});

expr tensor_detail_to_ir_tensor(
        sc_graph_t &graph, const std::string &name, const logical_tensor_t &);
std::vector<expr> tensor_detail_to_ir_tensor(sc_graph_t &graph,
        const std::string &name_prefix, const std::vector<logical_tensor_t> &);
std::vector<expr> tensor_detail_to_ir_tensor(sc_graph_t &graph,
        const std::string &name_prefix, const std::vector<graph_tensor_ptr> &);
std::string decay_quantized_op_name(const std::string &op_name);

// get logical_tensor_t from logical_tensor_t
void get_logical_tensors(
        ltensors *ins, const std::vector<graph_tensor_ptr> &flts);

// marks the "read_buffer" or "write_buffer" attributes on each expr in `args`.
// If `is_read` mark them read. Otherwise, mark them write.
void mark_read_or_write_buffers(std::vector<expr> &args, bool is_read);

/**
 * Creates an empty function declaration for an op. The function will have the
 * name `op::op_name_`. Its arguments are composed of the Op's outputs and then
 * the Op's inputs
 * @param op the op
 * @param ins the vector will be set with the input args of the generated IR
 * function
 * @param outs the vector will be set with the output args of the generated IR
 * functio
 * @return the generated IR function for the Op. Its body is an empty stmts node
 * */
func_t create_func_decl_for_op(
        sc_op *op, std::vector<expr> &ins, std::vector<expr> &outs);

/**
 * Creates an empty query format function declaration for an op. The function
 * will have the name `query_format_op::op_name_`. Its arguments are composed of
 * the Op's outputs and then the Op's inputs and their formats.
 * @param op the op
 * @param ins the vector of input args
 * @param ori_ins the vector of origin input args
 * @param outs the vector of output args
 * @param in_fmts the vector of input format args
 * @param ori_in_fmts the vector of origin input format args
 * @param out_fmts the vector of output format
 * @param out_sizes the size tensor of output
 * @param kernel the kernel tensor
 * @return the generated IR function for the Op. Its body is an empty stmts node
 * */
func_t create_query_func_decl_for_op(sc_op *op, std::vector<expr> &ins,
        std::vector<expr> &ori_ins, std::vector<expr> &outs,
        std::vector<expr> &in_fmts, std::vector<expr> &ori_in_fmts,
        std::vector<expr> &out_fmts, std::vector<expr> &out_sizes,
        expr &kernel);

ltensors extract_detail_from_tensors(
        const std::vector<std::shared_ptr<graph_tensor>> &);

/**
 * Checks whether lhs_shape could be identical to rhs_shape
 * considering dynamic shape use cases
 * */
bool check_shape_equal(const sc_dims &lhs_shape, const sc_dims &rhs_shape);

/**
 * Checks whether lhs and rhs matches for both plain_dims and dtype
 * considering dynamic shape use cases
 * */
void check_logical_tensor_shape_dtype_identical(
        const logical_tensor_t &lhs, const logical_tensor_t &rhs);
} // namespace graph
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
