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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_COMMIT_OP_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_OPS_TEMPLATES_COMMIT_OP_HPP

#include <string>
#include <vector>
#include <compiler/ir/graph/graph.hpp>
#include <compiler/ir/graph/tensor_slice.hpp>
#include <util/array_ref.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
/**
 * @brief Commit the code of an Op into the current IR builder. Use op_inputs,
 * op_outputs and attr to define the computation of the Op and the
 * shape/layout/stride info. And the Op's Tensor IR code will be committed on
 * the tensor slices in in_slice and out_slice.
 * @note The base tensor of the in_slice and out_slice should be of the same
 * shape of blocking dims specified by arguments op_inputs, op_outputs.
 * Otherwise, wrong code may be generated. The users may manually add the attr
 * "tensor_shrinker_attrs::may_shrink" on the local temp tensors to make them
 * smaller for better performance. @see tensor_shrinker_t
 *
 * @param ctx the context
 * @param opname the name of the Op
 * @param in_slice the input tensor slices in Tensor IR
 * @param out_slice the output tensor slices in Tensor IR
 * @param op_inputs the input graph tensor for the Op
 * @param op_outputs the output graph tensor for the Op
 * @param attr the attributes of the Op
 */
void commit_op(const context_ptr &ctx, const std::string &opname,
  array_ref<tensor_slice> in_slice, array_ref<tensor_slice> out_slice,
  const std::vector<graph_tensor_ptr> &op_inputs,
  const std::vector<graph_tensor_ptr> &op_outputs = {},
  const any_map_t &attr = {});
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
