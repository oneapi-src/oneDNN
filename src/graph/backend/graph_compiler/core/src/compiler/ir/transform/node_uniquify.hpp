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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_NODE_UNIQUIFY_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_TRANSFORM_NODE_UNIQUIFY_HPP

#include "../function_pass.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

/**
 * A simple pass to walk through the IR DAG, and convert it into a "tree". If a
 * node is not var_node or tensor_node and has multiple parent nodes, this pass
 * will deep copy it and replace it. After this pass, each node except vars and
 * tensors should have its own unique parent. Note that var_nodes and
 * tensor_nodes are not touched and they may still have multiple uses
 * */
class node_uniquifier_t : public function_pass_t {
public:
    func_c operator()(func_c f) override;
    expr_c operator()(expr_c f);
    stmt_c operator()(stmt_c f);
};

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
