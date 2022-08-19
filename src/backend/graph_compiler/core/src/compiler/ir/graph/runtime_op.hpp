/*******************************************************************************
 * Copyright 2022 Intel Corporation
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
#ifndef BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_RUNTIME_OP_HPP
#define BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_RUNTIME_OP_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph.hpp"
#include <util/reflection.hpp>
#include <util/utils.hpp>

namespace sc {
struct runtime_extra_info_t {
    std::vector<graph_tensor_ptr> in_ltsrs_;
    std::vector<graph_tensor_ptr> out_ltsrs_;
    std::vector<expr> attrs_;
};

// The kind of op determind their calculation during runtime not compile-time.
// May rely on runtime format/size/shape. Mainly for dynamic shape, like
// `shape_of_tensor`. If we want to get a padded plain shape of a tensor, the
// blocks are unknown util runtime query.

class runtime_op_t : public sc_op {
public:
    runtime_op_t() = default;

    runtime_op_t(const std::string &op_name,
            const std::vector<graph_tensor_ptr> &producer_lt,
            const std::vector<graph_tensor_ptr> &consumer_lt,
            const any_map_t &attrs)
        : sc_op(op_name, producer_lt, consumer_lt, attrs) {}

    void query_format(context_ptr ctx,
            std::vector<std::vector<format_stride_pair>> &supported_ins,
            std::vector<std::vector<format_stride_pair>> &supported_outs)
            override {};

    // Get extra related lower infos, return infos like related logical tensors.
    // The tensors may not just the input/output of current node. And extra
    // inner attributes of op.
    virtual runtime_extra_info_t get_extra_lower_infos(
            sc_graph_t &graph, ir_module_ptr &m)
            = 0;
};
} // namespace sc

#endif
