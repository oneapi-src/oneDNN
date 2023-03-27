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

#ifndef GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_LOWERING_HPP
#define GRAPH_BACKEND_GRAPH_COMPILER_CORE_SRC_COMPILER_IR_GRAPH_LOWERING_HPP
#include <string>
#include <vector>
#include "graph.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

namespace graph {
std::string get_tensor_name(graph_tensor *t, sc_op *linked_output);
}

expr call_op_dynamic_query_function(
        const sc_op_ptr &op, const std::vector<expr> &args);

struct tsr_info_t {
    expr tensor_;
    expr placeholder_;
    expr format_;
    expr size_;
    int count_ = 0;
    tsr_info_t() = default;
    tsr_info_t(const expr &tensor, const expr &placeholder, const expr &format,
            const expr &size)
        : tensor_(tensor)
        , placeholder_(placeholder)
        , format_(format)
        , size_(size) {}
};

enum info_etype_t { real_tensor, placeholder, format, out_size };

/**
 * Generates the ir_module_t from the OP graph
 * @param ctx the context
 * @param graph the graph
 * @param args optional order of the arguments of the generated IR function.
 * Should all be input_op or output_op. If empty, use default internal order
 * @param mark_as_main mark the main entry function with "is_main" attr
 * */
SC_API ir_module_ptr lower_graph(context_ptr ctx, sc_graph_t &graph,
        const std::vector<sc_op_ptr> &args, bool mark_as_main = true);
} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
