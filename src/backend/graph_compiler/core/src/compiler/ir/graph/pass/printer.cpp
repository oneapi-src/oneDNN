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

#include <fstream>
#include "../fusible_op.hpp"
#include "../graph_op.hpp"
#include "../visitor.hpp"
#include "pass.hpp"
#include <compiler/ir/graph/lowering.hpp>
#include <util/utils.hpp>

namespace dnnl {
namespace impl {
namespace graph {
namespace gc {

SC_INTERNAL_API void print_graph(const sc_graph_t &mgr, std::ostream &os,
        bool print_shape, bool print_attr, bool print_name, bool print_stride) {
    std::unordered_map<graph_tensor_ptr, int> tsr_idx;
    auto get_tensor_id = [&](const graph_tensor_ptr &t) {
        auto itr = tsr_idx.find(t);
        if (itr != tsr_idx.end()) { return itr->second; }
        int ret = tsr_idx.size();
        tsr_idx[t] = ret;
        return ret;
    };
    auto print_tensor_name = [&](const graph_tensor_ptr &t) {
        if (print_name) {
            sc_op *linked_output = nullptr;
            for (auto &use : t->uses_) {
                if (use.second->isa<output_op>()) {
                    linked_output = use.second.get();
                    break;
                }
            }
            auto name = graph::get_tensor_name(t.get(), linked_output);
            if (!name.empty()) {
                os << name;
                return;
            }
        }
        os << 'v' << get_tensor_id(t);
    };
    auto print_tensor_list = [&](const std::vector<graph_tensor_ptr> &list,
                                     bool p_shape, bool p_stride) {
        bool is_first_input = true;
        for (const auto &tsr : list) {
            if (!is_first_input) {
                os << ", ";
            } else {
                is_first_input = false;
            }
            print_tensor_name(tsr);
            if (p_shape) {
                os << ": " << tsr->details_.dtype_ << '[';
                bool is_first_shape = true;
                for (auto dim : tsr->details_.get_blocking_dims()) {
                    if (!is_first_shape) {
                        os << ", ";
                    } else {
                        is_first_shape = false;
                    }
                    os << dim;
                }
                os << ']';
            }
            if (p_stride) {
                os << ", "
                   << (tsr->details_.is_dense() ? "dense:" : "strided:");
                for (auto stride : tsr->details_.get_strides()) {
                    os << "s";
                    os << stride;
                }
            }
        }
    };
    std::vector<graph_tensor_ptr> inputs;
    std::vector<graph_tensor_ptr> outputs;
    for (auto &v : mgr.ops_) {
        if (v->isa<input_op>()) {
            inputs.insert(inputs.end(), v->get_outputs().begin(),
                    v->get_outputs().end());
        } else if (v->isa<output_op>()) {
            outputs.insert(outputs.end(), v->get_inputs().begin(),
                    v->get_inputs().end());
        }
    }
    std::string graph_name;
    if (print_name) {
        graph_name = mgr.attrs_.get_or_else<std::string>("temp.name", "graph");
    } else {
        graph_name = "graph";
    }
    os << graph_name << '(';
    print_tensor_list(inputs, print_shape, print_stride);
    os << ") -> [";
    print_tensor_list(outputs, print_shape, print_stride);
    os << "] {\n";

    op_visitor_t::dfs_topology_sort().visit_graph(
            mgr, [&](op_visitor_t *vis, const sc_op_ptr &node) {
                if (!node->isa<input_op>() && !node->isa<output_op>()) {
                    print_indents(os, 1);

                    os << '[';
                    print_tensor_list(
                            node->get_outputs(), print_shape, print_stride);
                    os << "] = " << node->op_name_;
                    os << '(';
                    if (auto con_node = node->dyn_cast<constant_op_t>()) {
                        os << utils::print_vector(
                                con_node->get_constant_blocking_dims());
                    } else {
                        print_tensor_list(node->get_inputs(), false, false);
                    }
                    os << ')' << '\n';
                }
            });
    os << '}' << '\n';
}

} // namespace gc
} // namespace graph
} // namespace impl
} // namespace dnnl
