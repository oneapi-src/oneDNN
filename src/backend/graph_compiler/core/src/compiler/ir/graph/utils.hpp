/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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

#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_UTILS_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_UTILS_HPP

#include <memory>
#include <string>
#include <vector>
#include <compiler/ir/graph/graph.hpp>

namespace sc {
namespace graph {
expr tensor_detail_to_ir_tensor(
        const std::string &name, const logical_tensor_t &);
expr tensor_detail_to_ir_tensor(
        const std::string &name, const graph_tensor_ptr &, gt2buf_map &);
std::vector<expr> tensor_detail_to_ir_tensor(
        const std::string &name_prefix, const std::vector<logical_tensor_t> &);
std::vector<expr> tensor_detail_to_ir_tensor(
        const std::string &name_prefix, const std::vector<graph_tensor_ptr> &);
std::vector<expr> tensor_detail_to_ir_tensor(const std::string &name_prefix,
        const std::vector<graph_tensor_ptr> &tsrs, gt2buf_map &g2b_map);
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

ltensors extract_detail_from_tensors(
        const std::vector<std::shared_ptr<graph_tensor>> &);
} // namespace graph
} // namespace sc

#endif
