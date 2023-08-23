/*******************************************************************************
 * Copyright 2021-2023 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_COMPILER_GRAPH_HPP
#define BACKEND_GRAPH_COMPILER_COMPILER_GRAPH_HPP

#include <string>
#include <vector>
#include <unordered_map>

#include "graph/interface/graph.hpp"
#include "graph/interface/op.hpp"
#include "utils.hpp"

#include "compiler/ir/graph/fusible_op.hpp"
#include "compiler/ir/graph/graph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace compiler_impl {

class compiler_graph_impl_t : public gc::sc_graph_t {
public:
    compiler_graph_impl_t() = default;

    // convert onednn graph op to compiler backend op
    gc::sc_op_ptr make_backend_op(const op_t *aop,
            const std::vector<gc::graph_tensor_ptr> &producer_lt,
            const std::vector<gc::graph_tensor_ptr> &consumer_lt);

    // convert onednn graph dtype to compiler backend dtype
    static inline gc::sc_data_type_t convert_data_type(data_type_t dtype);

    // convert partition's input logical tensor to compiler backend input node
    gc::sc_op_ptr make_compiler_backend_input(
            const graph::logical_tensor_t &lt, const size_t &partition_id);

    // get compiler backend ops
    const std::vector<gc::sc_op_ptr> get_compiler_backend_ops() { return ops_; }

    // onednn graph op id -> compiler backend op pointer
    std::unordered_map<int, gc::sc_op_ptr> op_mapping_;

    // convert onednn graph op attrs to compiler backend op attrs
    gc::any_map_t convert_op_attrs(const std::unordered_map<graph::op_attr_t,
            graph::utils::attribute_value_t> &attrs);

    // convert onednn graph logical tensor to backend graph tensor
    static gc::graph_tensor_ptr convert_logical_tensor(
            const graph::logical_tensor_t &lt);

    // return whether an op is supported by compiler backend or not
    static bool is_supported_op(op_kind_t name);
};

} // namespace compiler_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
