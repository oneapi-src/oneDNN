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

#include "commit_op.hpp"
#include <string>
#include <vector>
#include <compiler/ir/graph/fusible_op.hpp>
#include <compiler/ir/graph/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {
namespace ops {
void commit_op(const context_ptr &ctx, const std::string &opname,
  array_ref<tensor_slice> in_slice, array_ref<tensor_slice> out_slice,
  const std::vector<graph_tensor_ptr> &inputs,
  const std::vector<graph_tensor_ptr> &outputs, const any_map_t &attr) {
  auto graph = graph::make_single_op_graph(opname, inputs, outputs, attr);
  COMPILE_ASSERT(graph.ops_.size() == 3UL && graph.ops_[1]->isa<fusible_op_t>(),
    "commit_op only supports fusible op");
  auto op = graph.ops_[1]->stc_cast<fusible_op_t>();
  std::vector<tensor_slice> out_copy = out_slice.as_vector();
  std::vector<tensor_slice *> out;
  out.reserve(out_slice.size());
  for (auto &v : out_copy) {
    out.push_back(&v);
  }

  std::vector<const tensor_slice *> in;
  in.reserve(in_slice.size());
  for (auto &v : in_slice) {
    in.push_back(&v);
  }
  op->compute_block(ctx, out, in);
}
} // namespace ops
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
